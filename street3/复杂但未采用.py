# %%
import pandas as pd
import os
from glob import glob
import json
from PIL import Image
import torch as t
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torch.nn.functional as F
import numpy as np
import json
import random
import timm
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
    batch_size = 32  # 减小批量大小以适应更大的模型
    lr = 3e-4  # 降低学习率，提高稳定性
    momentum = 0.9
    weights_decay = 2e-3  # 增加权重衰减，减轻过拟合
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 1
    print_interval = 50
    checkpoints = './checkpoints7'
    pretrained = None 
    start_epoch = 0
    epoches = 35  # 增加训练轮数
    smooth = 0.15  # 调整标签平滑系数
    erase_prob = 0.5
    model_name = 'efficientnet_b4'  # 升级到更强大的模型
    mixed_precision = True
    num_workers = 4
    cutmix_prob = 0.5  # 添加CutMix增强
    mixup_prob = 0.3   # 添加MixUp增强
    weight_alpha = 0.6  # 损失函数权重

config = Config()


# %%

# 实现CutMix和MixUp数据增强
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # 均匀分布中采样中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 边界框坐标
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# 更强的数据集类
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
            
        # 添加自适应增强
        self.epoch = 0
        
    def update_epoch(self, epoch):
        """更新当前epoch，用于自适应增强"""
        self.epoch = epoch

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
        
        # 增强的数据增强 - 针对字符识别优化
        if self.aug and self.mode == 'train':
            aug_intensity = min(1.0, 0.5 + self.epoch * 0.02)  # 随着训练进行增加增强强度
            
            trans1.extend([
                # 透视变换 - 特别适合文本识别
                transforms.RandomPerspective(distortion_scale=0.2 * aug_intensity, p=0.5),
                
                # 保持字符基本形状的小角度旋转
                transforms.RandomRotation(10 * aug_intensity),
                
                # 颜色变化
                transforms.ColorJitter(
                    brightness=0.3 * aug_intensity, 
                    contrast=0.3 * aug_intensity, 
                    saturation=0.2 * aug_intensity, 
                    hue=0.1 * aug_intensity
                ),
                
                # 随机灰度
                transforms.RandomGrayscale(p=0.1 * aug_intensity),
                
                # 仿射变换 - 适合字符识别
                transforms.RandomAffine(
                    degrees=8 * aug_intensity, 
                    translate=(0.15 * aug_intensity, 0.15 * aug_intensity), 
                    scale=(0.85, 1.15), 
                    shear=8 * aug_intensity
                ),
                
                # 模拟低分辨率/模糊 - 常见于真实场景
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0 * aug_intensity)),
                
                # 后处理效果
                transforms.RandomPosterize(bits=4, p=0.2 * aug_intensity),
                transforms.RandomAutocontrast(p=0.3 * aug_intensity),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2 * aug_intensity),
                
                # 模拟阴影效果
                transforms.RandomEqualize(p=0.1 * aug_intensity),
            ])
        
        trans1.extend(trans0)
        
        # 应用变换
        img = transforms.Compose(trans1)(img)
        
        # 随机擦除 - 训练时的额外增强
        if self.aug and self.mode == 'train':
            # 动态调整擦除概率
            erase_prob = min(0.4, 0.2 + self.epoch * 0.01)
            img = transforms.RandomErasing(
                p=erase_prob, 
                scale=(0.02, 0.15), 
                ratio=(0.3, 3.0), 
                value='random'
            )(img)
        
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
                self.width = random.choice(range(224, 256, 8))  # 更细粒度的宽度变化

        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels) 

# %%

# 注意力模块增强版
class CBAM(nn.Module):
    """通道与空间注意力模块"""
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        
        # 空间注意力
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # 空间注意力
        avg_out = t.mean(x, dim=1, keepdim=True)
        max_out, _ = t.max(x, dim=1, keepdim=True)
        spatial_inp = t.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv(spatial_inp)
        
        return x * spatial_out

# 改进的模型架构
class ImprovedModel(nn.Module):
    def __init__(self, class_num=11, model_name='efficientnet_b4', dropout=0.25):
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
            
        # 增强版注意力机制
        self.attention = CBAM(feature_dim)
        
        # 特征融合层 - 增加非线性表示能力
        self.neck = nn.Sequential(
            nn.Linear(feature_dim, 1536),
            nn.BatchNorm1d(1536),
            nn.SiLU(inplace=True),  # SiLU (Swish) 激活函数
            nn.Dropout(dropout),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.8)
        )
        
        # 分支层 - 添加BatchNorm提高稳定性
        self.dropout = nn.Dropout(dropout * 0.5)  # 不同层使用不同dropout率
        
        # 专用于每个位置的特征提取
        self.position_embedding = nn.Parameter(t.randn(4, 256))
        
        # 为每个数字位置创建专门分支
        self.fc_common = nn.Linear(1024, 256)
        
        self.digit_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, class_num)
            ) for _ in range(4)
        ])

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
        
        # 共享特征处理
        common_feat = self.fc_common(self.dropout(feat))  # [B, 256]
        
        # 分支预测 - 为每个位置添加位置编码
        outputs = []
        for i, classifier in enumerate(self.digit_classifiers):
            # 添加位置信息
            position_feat = common_feat + self.position_embedding[i]
            outputs.append(classifier(position_feat))
        
        return tuple(outputs)

# 加权标签平滑损失
class WeightedLabelSmoothEntropy(nn.Module):
    def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
        super(WeightedLabelSmoothEntropy, self).__init__()
        self.size_average = size_average
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, preds, targets):
        # 为标签10（空）提供更高的权重
        if self.class_weights is None:
            weights = t.ones(preds.shape[1], device=preds.device)
            weights[10] = 1.5  # 给空白类更高权重
        else:
            weights = self.class_weights
        
        lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[1] - 1)
        smoothed_lb = t.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)
        log_soft = F.log_softmax(preds, dim=1)
        loss = -log_soft * smoothed_lb * weights[None, :]
        loss = loss.sum(1)
        
        if self.size_average == 'mean':
            return loss.mean()
        elif self.size_average == 'sum':
            return loss.sum()
        else:
            return loss

# 改进版Focal Loss
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 针对数字识别的类别不平衡处理
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = t.exp(-ce_loss)
        
        # 根据类别调整alpha
        alpha = t.ones_like(targets, dtype=t.float32) * self.alpha
        alpha = t.where(targets == 10, alpha * 1.5, alpha)  # 空类权重调整
        
        focal_loss = alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, smooth=0.1, alpha=0.6):
        super(CombinedLoss, self).__init__()
        self.smooth_loss = WeightedLabelSmoothEntropy(smooth=smooth)
        self.focal_loss = ImprovedFocalLoss(gamma=2.5)  # 增加gamma值
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
        self.model = ImprovedModel(config.class_num, model_name=config.model_name, dropout=0.25).to(self.device)
        
        # 使用组合损失函数
        self.criterion = CombinedLoss(smooth=config.smooth, alpha=config.weight_alpha).to(self.device)
        
        # 使用AdamW优化器，添加更高的epsilon稳定训练
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, 
                              weight_decay=config.weights_decay, eps=1e-7)
        
        # 使用余弦退火学习率调度
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.epoches//3,  # 第一次重启周期
            T_mult=2,  # 每次重启后周期长度翻倍
            eta_min=config.lr/100  # 最小学习率
        )
        
        # 混合精度训练
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

    def apply_cutmix_or_mixup(self, images, labels):
        """应用CutMix或MixUp数据增强"""
        batch_size = images.size(0)
        
        # 随机决定是否应用CutMix
        if random.random() < config.cutmix_prob:
            # 生成混合权重
            lam = np.random.beta(1.0, 1.0)
            rand_index = t.randperm(batch_size).to(self.device)
            
            # 混合标签
            mixed_labels = labels[rand_index]
            
            # 应用CutMix
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # 调整混合系数
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            
            return images, labels, mixed_labels, lam
            
        # 随机决定是否应用MixUp
        elif random.random() < config.mixup_prob:
            # 生成混合权重
            lam = np.random.beta(1.5, 1.5)
            rand_index = t.randperm(batch_size).to(self.device)
            
            # 混合图像
            mixed_images = lam * images + (1 - lam) * images[rand_index]
            mixed_labels = labels[rand_index]
            
            return mixed_images, labels, mixed_labels, lam
        
        return images, labels, None, None

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            print(f"Epoch {epoch+1}/{config.epoches}")
            
            # 更新数据集的epoch信息，用于自适应增强
            self.train_set.update_epoch(epoch)
            
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
            
            # 更新学习率
            self.lr_scheduler.step()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            
            # 应用CutMix或MixUp
            img, target_a, target_b, lam = self.apply_cutmix_or_mixup(img, label)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if config.mixed_precision:
                with t.amp.autocast('cuda'):
                    pred = self.model(img)
                    
                    # 如果应用了CutMix/MixUp，使用混合损失
                    if target_b is not None:
                        loss = 0
                        for j in range(4):
                            loss += lam * self.criterion(pred[j], target_a[:, j]) + \
                                    (1 - lam) * self.criterion(pred[j], target_b[:, j])
                    else:
                        loss = sum(self.criterion(pred[j], label[:, j]) for j in range(4))
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪 - 防止梯度爆炸
                self.scaler.unscale_(self.optimizer)
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(img)
                
                # 如果应用了CutMix/MixUp，使用混合损失
                if target_b is not None:
                    loss = 0
                    for j in range(4):
                        loss += lam * self.criterion(pred[j], target_a[:, j]) + \
                                (1 - lam) * self.criterion(pred[j], target_b[:, j])
                else:
                    loss = sum(self.criterion(pred[j], label[:, j]) for j in range(4))
                
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 只计算没有混合的样本准确率，或只用target_a
            if target_b is None or True:
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

def test_time_augmentation(model, img, n_augment=5):
    """测试时增强，提高预测准确率"""
    model.eval()
    
    # 原始预测
    with t.no_grad():
        original_pred = model(img)
    
    # TTA变换
    tta_transforms = [
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
    ]
    
    all_preds = [original_pred]
    
    # 应用TTA并收集预测结果
    for i in range(n_augment):
        # 随机选择变换
        trans = random.choice(tta_transforms)
        aug_img = trans(img)
        
        with t.no_grad():
            aug_pred = model(aug_img)
            all_preds.append(aug_pred)
    
    # 平均预测结果
    final_pred = []
    for i in range(4):  # 四个数字位置
        position_preds = [pred[i] for pred in all_preds]
        final_pred.append(sum(position_preds) / len(position_preds))
    
    return tuple(final_pred)

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
            
            # 使用测试时增强
            pred = test_time_augmentation(res_net, img, n_augment=3)
            
            results += [[name, code] for name, code in zip(img_names, parse2class(pred))]
    
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results

# %%
if __name__ == '__main__':
    print('__name__:', __name__)
    
    # 开始训练
    trainer = Trainer()
    trainer.train()
    acc = trainer.eval()
    print('Local validation accuracy: %.2f%%' % (acc * 100))
    
    # 生成预测结果
    predicts(trainer.best_checkpoint_path, "claude7_result.csv")