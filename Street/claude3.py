# 只是删掉了下载的部分；并调整了train_epoch中的顺序，因为报warning。结果还是报warning。那就不用这份代码了。

# %%
import pandas as pd
import os
from glob import glob
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import torch as t
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import torch.nn.functional as F
import json
from torchsummary import summary
import random
import timm
# 修复导入路径
from torch.amp import autocast, GradScaler

dataset_path = "./dataset"

# 全局数据目录，下载好了
data_dir = {
    "train_data": f"{dataset_path}/mchar_train/",
    "val_data": f"{dataset_path}/mchar_val/",
    "test_data": f"{dataset_path}/mchar_test_a/",
    "train_label": f"{dataset_path}/mchar_train.json",
    "val_label": f"{dataset_path}/mchar_val.json",
    "submit_file": f"{dataset_path}/mchar_sample_submit_A.csv",
}

# %%

# 改进配置参数
class Config:
    batch_size = 48  # 减小批量大小以适应更大的模型
    lr = 5e-4  # 调整学习率
    momentum = 0.9
    weights_decay = 1e-3  # 增加权重衰减
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 1  # 每个epoch都保存检查点
    print_interval = 50
    checkpoints = './checkpoints3'
    pretrained = None 
    start_epoch = 0
    epoches = 25 # 15  # 增加训练轮数
    smooth = 0.2  # 增加标签平滑系数
    erase_prob = 0.5
    model_name = 'efficientnet_b3'  # 使用EfficientNet-B3
    mixed_precision = True  # 启用混合精度训练
    num_workers = 4  # 减少worker数量避免问题

config = Config()


# %%

# 增强版数据集类
class DigitsDataset(Dataset):
    """
    DigitsDataset with advanced augmentation
    """
    def __init__(self, mode='train', size=(128, 224), aug=True):
        super(DigitsDataset, self).__init__()
        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        self.batch_count = 0
        global data_dir
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
        
        # 基础变换
        trans0 = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # 大小调整变换
        trans1 = [
            transforms.Resize((128, self.width)),
        ]
        
        # 增强的数据增强
        if self.aug and self.mode == 'train':
            trans1.extend([
                transforms.RandomRotation(10),  # 适度的旋转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增强的颜色抖动
                transforms.RandomGrayscale(0.1),
                transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.RandomPosterize(bits=4, p=0.2),
                transforms.RandomAutocontrast(p=0.2),
            ])
        
        trans1.extend(trans0)
        
        # 应用变换
        img = transforms.Compose(trans1)(img)
        
        # 随机擦除 - 训练时的额外增强
        if self.aug and self.mode == 'train':
            img = transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.0))(img)
        
        if self.mode != 'test':
            return img, t.tensor(label['label'][:4] + (4 - len(label['label'])) * [10]).long()
        else:
            return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))

        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels) 

# %%

# 注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 改进的模型架构
class ImprovedModel(nn.Module):
    def __init__(self, class_num=11, model_name='efficientnet_b3', dropout=0.2):
        super(ImprovedModel, self).__init__()
        
        # 使用timm库加载预训练模型
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=False)
        
        # 获取特征维度
        if 'efficientnet' in model_name:
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in model_name:
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # 共享特征提取层后添加SE注意力
        self.attention = SEBlock(feature_dim)
        
        # 特征融合层
        self.neck = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 分支层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024, class_num)
        self.fc2 = nn.Linear(1024, class_num)
        self.fc3 = nn.Linear(1024, class_num)
        self.fc4 = nn.Linear(1024, class_num)

    def forward(self, img):
        # 特征提取
        feat = self.backbone(img)
        
        if isinstance(feat, t.Tensor):
            # 如果是Tensor，直接处理
            feat = feat.view(feat.size(0), -1)
        else:
            # 如果是多尺度特征，取最后一个
            feat = feat[-1]
            feat = nn.AdaptiveAvgPool2d(1)(feat)
            feat = feat.view(feat.size(0), -1)
        
        # 应用注意力
        if feat.dim() == 4:  # 如果是4D tensor
            feat = self.attention(feat)
            feat = t.flatten(feat, 1)
        
        # 特征融合
        feat = self.neck(feat)
        
        # 分支预测
        c1 = self.fc1(self.dropout(feat))
        c2 = self.fc2(self.dropout(feat))
        c3 = self.fc3(self.dropout(feat))
        c4 = self.fc4(self.dropout(feat))
        
        return c1, c2, c3, c4

# %%

# 添加Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = t.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 改进的标签平滑损失
class LabelSmoothEntropy(nn.Module):
    def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
        super(LabelSmoothEntropy, self).__init__()
        self.size_average = size_average
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, preds, targets):
        lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[1] - 1)
        smoothed_lb = t.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)
        log_soft = F.log_softmax(preds, dim=1)
        if self.class_weights is not None:
            loss = -log_soft * smoothed_lb * self.class_weights[None, :]
        else:
            loss = -log_soft * smoothed_lb
        loss = loss.sum(1)
        if self.size_average == 'mean':
            return loss.mean()
        elif self.size_average == 'sum':
            return loss.sum()
        else:
            return loss

# 组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, smooth=0.1, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.smooth_loss = LabelSmoothEntropy(smooth=smooth)
        self.focal_loss = FocalLoss(gamma=2)
        self.alpha = alpha
        
    def forward(self, preds, targets):
        return self.alpha * self.smooth_loss(preds, targets) + (1 - self.alpha) * self.focal_loss(preds, targets)

class Trainer:
    def __init__(self, val=True):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        self.train_set = DigitsDataset(mode='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, 
                                       num_workers=config.num_workers,
                                       pin_memory=True, persistent_workers=True, \
                                       drop_last=True, collate_fn=self.train_set.collect_fn)
        if val:
            self.val_loader = DataLoader(DigitsDataset(mode='val', aug=False), batch_size=config.batch_size, \
                                         num_workers=config.num_workers, pin_memory=True, drop_last=False, persistent_workers=True)
        else:
            self.val_loader = None

        # 使用改进的模型
        self.model = ImprovedModel(config.class_num, model_name=config.model_name, dropout=0.2).to(self.device)
        
        # 使用组合损失函数
        self.criterion = CombinedLoss(smooth=config.smooth, alpha=0.7).to(self.device)
        
        # 使用AdamW优化器
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weights_decay)
        
        # 使用OneCycleLR调度器
        self.lr_scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=config.lr,
            epochs=config.epoches,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000
        )
        
        # 修复GradScaler使用
        self.scaler = GradScaler() if config.mixed_precision else None
        
        self.best_acc = 0
        self.best_checkpoint_path = ""
        
        # 载入预训练模型
        if config.pretrained is not None:
            self.load_model(config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print('Load model from %s, Eval Acc: %.2f' % (config.pretrained, acc * 100))

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            print(f"Epoch {epoch+1}/{config.epoches}")
            train_acc = self.train_epoch(epoch)
            
            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                if self.val_loader is not None:
                    val_acc = self.eval()
                    print(f'Validation accuracy: {val_acc:.4f}')
                    
                    # 保存最优模型
                    if val_acc > self.best_acc:
                        os.makedirs(config.checkpoints, exist_ok=True)
                        save_path = os.path.join(config.checkpoints,
                                              f'epoch-{config.model_name}-{epoch+1}-acc-{val_acc*100:.2f}.pth')
                        self.save_model(save_path)
                        print('%s saved successfully...' % save_path)
                        self.best_acc = val_acc
                        self.best_checkpoint_path = save_path

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            
            # 混合精度训练
            if config.mixed_precision:
                with t.amp.autocast('cuda'):
                    self.optimizer.zero_grad()
                    pred = self.model(img)
                    loss = (
                        self.criterion(pred[0], label[:, 0]) +
                        self.criterion(pred[1], label[:, 1]) +
                        self.criterion(pred[2], label[:, 2]) +
                        self.criterion(pred[3], label[:, 3])
                    )
                
                # 混合精度反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 修改：先更新优化器，再更新学习率
                self.lr_scheduler.step()
            else:
                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = (
                    self.criterion(pred[0], label[:, 0]) +
                    self.criterion(pred[1], label[:, 1]) +
                    self.criterion(pred[2], label[:, 2]) +
                    self.criterion(pred[3], label[:, 3])
                )
                loss.backward()
                
                # 梯度裁剪
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                
                # 修改：先更新优化器，再更新学习率
                self.lr_scheduler.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            temp = t.stack([
                pred[0].argmax(1) == label[:, 0],
                pred[1].argmax(1) == label[:, 1],
                pred[2].argmax(1) == label[:, 2],
                pred[3].argmax(1) == label[:, 3],
            ], dim=1)
            
            corrects += t.all(temp, dim=1).sum().item()
            current_acc = corrects * 100 / ((i + 1) * config.batch_size)
            
            tbar.set_description(
                f'loss: {loss.item():.3f}, acc: {current_acc:.3f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}'
            )
            
        epoch_acc = corrects * 100 / (len(self.train_loader) * config.batch_size)
        return epoch_acc

    def eval(self):
        self.model.eval()
        corrects = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                
                pred = self.model(img)
                
                temp = t.stack([
                    pred[0].argmax(1) == label[:, 0],
                    pred[1].argmax(1) == label[:, 1],
                    pred[2].argmax(1) == label[:, 2],
                    pred[3].argmax(1) == label[:, 3],
                ], dim=1)
                
                corrects += t.all(temp, dim=1).sum().item()
                current_acc = corrects * 100 / ((i + 1) * len(img))
                tbar.set_description(f'Val Acc: {current_acc:.2f}')
        
        val_acc = corrects / (len(self.val_loader.dataset))
        return val_acc

    def save_model(self, save_path, save_opt=True, save_config=True):
        dicts = {}
        dicts['model'] = self.model.state_dict()
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}
        t.save(dicts, save_path)

    def load_model(self, load_path, changed=False, save_opt=False, save_config=False):
        dicts = t.load(load_path)
        if not changed:
            self.model.load_state_dict(dicts['model'])

        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])

        if save_config:
            for k, v in dicts['config'].items():
                config.__setattr__(k, v)

# %%
def parse2class(prediction):
    """
    Params:
    prediction(tuple of tensor): 
    """
    ch1, ch2, ch3, ch4 = prediction
    char_list = [str(i) for i in range(10)]
    char_list.append('')
    ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)
    ch1, ch2, ch3, ch4 = [char_list[i.item()] for i in ch1], [char_list[i.item()] for i in ch2], \
                    [char_list[i.item()] for i in ch3], [char_list[i.item()] for i in ch4] 
    res = [c1+c2+c3+c4 for c1, c2, c3, c4 in zip(ch1, ch2, ch3, ch4)]             
    return res

def write2csv(results,csv_path):
    """
    results(list):
    """
    # 定义输出文件
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: x.split('/')[-1])
    save_name = csv_path
    df.to_csv(save_name, sep=',', index=None)
    print('Results saved to %s'%save_name)

def predicts(model_path, csv_path):
    test_loader = DataLoader(DigitsDataset(mode='test', aug=False), batch_size=config.batch_size, shuffle=False,\
                    num_workers=config.num_workers, pin_memory=True, drop_last=False,persistent_workers=True)
    results = []
    res_path = model_path
    
    # 使用相同模型架构
    res_net = ImprovedModel(config.class_num, model_name=config.model_name).cuda()
    res_net.load_state_dict(t.load(res_path)['model'])
    print('Load model from %s successfully' % model_path)
    
    tbar = tqdm(test_loader)
    res_net.eval()
    with t.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.cuda()
            pred = res_net(img)
            results += [[name, code] for name, code in zip(img_names, parse2class(pred))]
    
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results

# %%
if __name__ == '__main__':
    print('__name__:', __name__)
    
    # 只在主进程中执行数据下载
    # data_dir = download_dataset()
    
    # 开始训练
    trainer = Trainer()
    trainer.train()
    acc = trainer.eval()
    print('Local validation accuracy: %.2f%%' % (acc * 100))
    
    # 生成预测结果
    predicts(trainer.best_checkpoint_path, "claude3_result.csv")
