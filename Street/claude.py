import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import timm  # 我们将使用这个库加载更多先进模型
from tqdm.auto import tqdm
import random
import numpy as np
from PIL import Image
from glob import glob
import json
import math
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练

# 复用baseline中的数据目录设置
dataset_path = "./dataset"
data_dir = {
    'train_data': f'{dataset_path}/mchar_train/',
    'val_data': f'{dataset_path}/mchar_val/',
    'test_data': f'{dataset_path}/mchar_test_a/',
    'train_label': f'{dataset_path}/mchar_train.json',
    'val_label': f'{dataset_path}/mchar_val.json',
    'submit_file': f'{dataset_path}/mchar_sample_submit_A.csv'
}

# 改进的配置类
class ImprovedConfig:
    batch_size = 32  # 降低批量大小以适应更大的模型
    lr = 5e-4  # 降低学习率以获得更稳定的训练
    weight_decay = 1e-2  # 增加权重衰减以减少过拟合
    class_num = 11  # 与原始相同，包括空类别
    eval_interval = 1
    checkpoint_interval = 1
    print_interval = 50
    checkpoints = './claude'  # 新的检查点目录
    pretrained = None
    start_epoch = 0
    epochs = 40  # 增加训练轮数
    smooth = 0.1
    focal_loss_gamma = 2.0  # Focal Loss参数
    focal_loss_alpha = 0.25  # Focal Loss参数
    tta_enabled = True  # 测试时增强
    tta_times = 5  # 测试时增强次数
    mixed_precision = True  # 使用混合精度训练
    model_name = 'efficientnet_b0'  # 使用EfficientNet-B0作为基础模型
    attention = True  # 启用注意力机制
    sequence_modeling = True  # 启用顺序建模（GRU）
    
config = ImprovedConfig()

# 确保检查点目录存在
os.makedirs(config.checkpoints, exist_ok=True)

# 设置随机种子以获得可重复的结果
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything()

# 增强的数据增强策略
class ImprovedDigitsDataset(Dataset):
    def __init__(self, mode='train', size=(128, 224), aug=True):
        super(ImprovedDigitsDataset, self).__init__()
        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        self.batch_count = 0
        
        # 加载数据
        if mode == 'test':
            self.imgs = glob(data_dir['test_data'] + '*.png')
            self.labels = None
        else:
            labels = json.load(open(data_dir['%s_label' % mode], 'r'))
            imgs = glob(data_dir['%s_data' % mode] + '*.png')
            self.imgs = [(img, labels[os.path.split(img)[-1]]) for img in imgs \
                         if os.path.split(img)[-1] in labels]
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None
        
        img = Image.open(img)
        
        # 获取适合当前模式的变换
        transform = self.get_transforms(self.mode if self.aug else 'val')
        
        if self.mode != 'test':
            return transform(img), torch.tensor(
                label['label'][:4] + (4 - len(label['label'])) * [10]).long()
        else:
            return transform(img), self.imgs[idx]
    
    def __len__(self):
        return len(self.imgs)
    
    def get_transforms(self, mode='train'):
        if mode == 'train':
            return transforms.Compose([
                # 方法1: 使用RandomResizedCrop直接替代Resize和RandomCrop
                transforms.RandomResizedCrop((128, 224)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
            ])
        else:  # test/val mode
            return transforms.Compose([
                transforms.Resize((128, 224)),  # 确保测试/验证图像也是正确大小
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))
        
        self.batch_count += 1
        return torch.stack(imgs).float(), torch.stack(labels) if self.mode != 'test' else labels

# 注意力机制模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# Focal Loss实现
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
    
    def forward(self, logits, targets):
        # 计算交叉熵
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        
        # 计算focal loss
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        loss = -self.alpha * (1 - prob) ** self.gamma * log_prob * targets_one_hot
        
        loss = loss.sum(dim=-1)
        if self.size_average == 'mean':
            return loss.mean()
        elif self.size_average == 'sum':
            return loss.sum()
        else:
            return loss

# 改进的模型实现
class ImprovedDigitsModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', class_num=11, use_attention=True, use_sequence=True):
        super(ImprovedDigitsModel, self).__init__()
        
        # 使用timm库加载预训练模型
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        
        # 获取特征维度
        dummy_input = torch.randn(1, 3, 128, 224)
        features = self.backbone(dummy_input)
        feature_dim = features[-1].shape[1]  # 获取最后一层的通道数
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(feature_dim)
        
        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        
        # 序列建模
        self.use_sequence = use_sequence
        if use_sequence:
            self.gru = nn.GRU(feature_dim, 512, bidirectional=True, batch_first=True)
            hidden_size = 512 * 2  # 双向GRU
        else:
            hidden_size = feature_dim
        
        # 数字分类头
        self.digit_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, class_num) for _ in range(4)
        ])
    
    def forward(self, x):
        # 特征提取 - 使用最后一层特征
        features = self.backbone(x)[-1]
        
        # 应用注意力机制
        if self.use_attention:
            features = self.attention(features)
        
        # 全局平均池化
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        
        # 序列建模
        if self.use_sequence:
            features = features.unsqueeze(1).repeat(1, 4, 1)  # [batch_size, seq_len=4, features]
            seq_features, _ = self.gru(features)
            
            # 分类
            digits = tuple(classifier(seq_features[:, i]) for i, classifier in enumerate(self.digit_classifiers))
        else:
            # 不使用序列建模，直接分类
            digits = tuple(classifier(features) for classifier in self.digit_classifiers)
        
        return digits

# 训练器类
class ImprovedTrainer:
    def __init__(self, val=True):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_set = ImprovedDigitsDataset(mode='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, 
                                      num_workers=8, pin_memory=True, persistent_workers=True,
                                      drop_last=True, collate_fn=self.train_set.collect_fn)
        
        if val:
            self.val_loader = DataLoader(ImprovedDigitsDataset(mode='val', aug=False), 
                                        batch_size=config.batch_size, num_workers=8,
                                        pin_memory=True, drop_last=False, persistent_workers=True)
        else:
            self.val_loader = None
        
        self.model = ImprovedDigitsModel(
            model_name=config.model_name,
            class_num=config.class_num,
            use_attention=config.attention,
            use_sequence=config.sequence_modeling
        ).to(self.device)
        
        # 使用Focal Loss替换原有的标签平滑交叉熵
        self.criterion = FocalLoss(gamma=config.focal_loss_gamma, 
                                  alpha=config.focal_loss_alpha).to(self.device)
        
        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # OneCycleLR学习率调度
        steps_per_epoch = len(self.train_loader)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=config.epochs,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1000
        )
        
        # 混合精度训练
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        
        self.best_acc = 0
        self.best_checkpoint_path = ""
        
        # 加载预训练模型
        if config.pretrained is not None:
            self.load_model(config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print(f'Load model from {config.pretrained}, Eval Acc: {acc * 100:.2f}')
    
    def train(self):
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch {epoch+1}/{config.epochs}")
            train_acc = self.train_epoch(epoch)
            
            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                if self.val_loader is not None:
                    val_acc = self.eval()
                    print(f"Epoch {epoch+1} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc * 100:.2f}%")
                    
                    # 保存最佳模型
                    if val_acc > self.best_acc:
                        save_path = os.path.join(config.checkpoints, 
                                               f'improved-{config.model_name}-epoch-{epoch+1}-acc-{val_acc * 100:.2f}.pth')
                        self.save_model(save_path)
                        print(f'{save_path} saved successfully...')
                        self.best_acc = val_acc
                        self.best_checkpoint_path = save_path
    
    def train_epoch(self, epoch):
        total_loss = 0
        corrects = 0
        samples = 0
        tbar = tqdm(self.train_loader)
        self.model.train()
        
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            batch_size = img.size(0)
            samples += batch_size
            
            # 混合精度训练
            if config.mixed_precision:
                with torch.amp.autocast('cuda'):
                    pred = self.model(img)
                    loss = sum(self.criterion(pred[j], label[:, j]) for j in range(4))
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = sum(self.criterion(pred[j], label[:, j]) for j in range(4))
                loss.backward()
                self.optimizer.step()
            
            # 更新学习率
            self.lr_scheduler.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            temp = torch.stack([
                pred[0].argmax(1) == label[:, 0],
                pred[1].argmax(1) == label[:, 1],
                pred[2].argmax(1) == label[:, 2],
                pred[3].argmax(1) == label[:, 3],
            ], dim=1)
            corrects += torch.all(temp, dim=1).sum().item()
            
            lr = self.optimizer.param_groups[0]['lr']
            tbar.set_description(
                f'loss: {total_loss/(i+1):.4f}, acc: {corrects * 100 / samples:.3f}, lr: {lr:.6f}'
            )
        
        return corrects * 100 / samples
    
    def eval(self):
        self.model.eval()
        corrects = 0
        samples = 0
        
        with torch.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                batch_size = img.size(0)
                samples += batch_size
                
                # 如果启用TTA，则进行测试时增强
                if config.tta_enabled:
                    pred = self.test_time_augmentation(img)
                else:
                    pred = self.model(img)
                
                temp = torch.stack([
                    pred[0].argmax(1) == label[:, 0],
                    pred[1].argmax(1) == label[:, 1],
                    pred[2].argmax(1) == label[:, 2],
                    pred[3].argmax(1) == label[:, 3],
                ], dim=1)
                corrects += torch.all(temp, dim=1).sum().item()
                tbar.set_description(f'Val Acc: {corrects * 100 / samples:.2f}')
        
        self.model.train()
        return corrects / samples
    
    def test_time_augmentation(self, img):
        """测试时增强，对同一张图像进行多次预测并取平均"""
        self.model.eval()
        
        # 原始图像预测
        with torch.no_grad():
            base_pred = self.model(img)
        
        # 如果不需要额外增强，直接返回
        if config.tta_times <= 1:
            return base_pred
        
        # 准备TTA变换
        tta_transforms = [
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.02, 0.02)),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
            transforms.GaussianBlur(3)
        ]
        
        # 存储所有预测结果
        all_preds = [base_pred]
        
        # 应用TTA并预测
        for _ in range(config.tta_times - 1):
            # 随机选择一个变换
            augment = random.choice(tta_transforms)
            aug_img = torch.stack([augment(x) for x in img])
            
            with torch.no_grad():
                aug_pred = self.model(aug_img)
                all_preds.append(aug_pred)
        
        # 平均所有预测结果
        final_preds = []
        for digit_pos in range(4):
            digit_preds = [p[digit_pos] for p in all_preds]
            avg_pred = torch.mean(torch.stack(digit_preds), dim=0)
            final_preds.append(avg_pred)
        
        return tuple(final_preds)
    
    def save_model(self, save_path, save_opt=False, save_config=False):
        dicts = {}
        dicts['model'] = self.model.state_dict()
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
        if save_config:
            dicts['config'] = {s: getattr(config, s) for s in dir(config) if not s.startswith('_')}
        torch.save(dicts, save_path)
    
    def load_model(self, load_path, changed=False, save_opt=False, save_config=False):
        dicts = torch.load(load_path)
        if not changed:
            self.model.load_state_dict(dicts['model'])
        
        if save_opt and 'opt' in dicts:
            self.optimizer.load_state_dict(dicts['opt'])
        
        if save_config and 'config' in dicts:
            for k, v in dicts['config'].items():
                setattr(config, k, v)

# 预测函数
def parse2class(prediction):
    """将模型预测转换为字符串结果"""
    ch1, ch2, ch3, ch4 = prediction
    char_list = [str(i) for i in range(10)]
    char_list.append('')
    ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)
    ch1, ch2, ch3, ch4 = [char_list[i.item()] for i in ch1], [char_list[i.item()] for i in ch2], \
                    [char_list[i.item()] for i in ch3], [char_list[i.item()] for i in ch4] 
    res = [c1+c2+c3+c4 for c1, c2, c3, c4 in zip(ch1, ch2, ch3, ch4)]             
    return res

def write2csv(results, csv_path):
    """将结果写入CSV文件"""
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: x.split('/')[-1] if '/' in x else x.split('\\')[-1])
    save_name = csv_path
    df.to_csv(save_name, sep=',', index=None)
    print(f'Results saved to {save_name}')

def predicts(model_path, csv_path):
    """使用模型进行预测并保存结果"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        ImprovedDigitsDataset(mode='test', aug=False), 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=8, 
        pin_memory=True, 
        drop_last=False,
        persistent_workers=True
    )
    
    # 载入模型
    model = ImprovedDigitsModel(
        model_name=config.model_name,
        class_num=config.class_num,
        use_attention=config.attention,
        use_sequence=config.sequence_modeling
    ).to(device)
    
    dicts = torch.load(model_path)
    model.load_state_dict(dicts['model'])
    print(f'Load model from {model_path} successfully')
    
    # 进行预测
    results = []
    tbar = tqdm(test_loader)
    model.eval()
    
    with torch.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.to(device)
            
            # 使用测试时增强
            if config.tta_enabled:
                pred = test_time_augmentation(model, img, n_augmentations=config.tta_times)
            else:
                pred = model(img)
            
            results += [[name, code] for name, code in zip(img_names, parse2class(pred))]
    
    # 排序并写入结果
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results

def test_time_augmentation(model, img, n_augmentations=5):
    """测试时增强函数"""
    # 原始图像预测
    pred = model(img)
    all_preds = [pred]
    
    # TTA变换
    tta_transforms = [
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.02, 0.02)),
        transforms.ColorJitter(brightness=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
        transforms.GaussianBlur(3)
    ]
    
    # 应用TTA并预测
    for _ in range(n_augmentations):
        # 随机选择一个变换
        augment = random.choice(tta_transforms)
        aug_img = torch.stack([augment(x) for x in img])
        
        aug_pred = model(aug_img)
        all_preds.append(aug_pred)
    
    # 平均所有预测结果
    final_preds = []
    for digit_pos in range(4):
        digit_preds = [p[digit_pos] for p in all_preds]
        avg_pred = torch.mean(torch.stack(digit_preds), dim=0)
        final_preds.append(avg_pred)
    
    return tuple(final_preds)

# 运行训练
if __name__ == '__main__':
    trainer = ImprovedTrainer(val=True)
    trainer.train()
    acc = trainer.eval()
    print(f'Final validation accuracy: {acc * 100:.2f}%')
    
    # 生成提交文件
    if trainer.best_checkpoint_path:
        predicts(trainer.best_checkpoint_path, "improved_result.csv")