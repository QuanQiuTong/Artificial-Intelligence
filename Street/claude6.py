# %%
import pandas as pd
import os
from glob import glob
import json
from PIL import Image
import numpy as np
import torch as t
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, SubsetRandomSampler
import torch.nn.functional as F
import random
import timm
import copy
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold

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

# 增强版配置参数
class Config:
    batch_size = 32  # 减小批量大小以适应更大的模型
    lr = 1e-3  # 略微增大学习率
    momentum = 0.9
    weights_decay = 2e-4  # 增加权重衰减
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 1  # 每个epoch都保存检查点
    print_interval = 50
    checkpoints = './checkpoints_advanced'
    pretrained = None 
    start_epoch = 0
    epoches = 35  # 更多epoch训练
    smooth = 0.15  # 调整标签平滑系数
    erase_prob = 0.5
    model_name = 'efficientnetv2_rw_m'  # 更强的模型
    mixed_precision = True  # 启用混合精度训练
    num_workers = 4  # 减少worker数量避免问题
    use_mixup = True  # 启用Mixup增强
    use_cutmix = True  # 启用CutMix增强
    mixup_alpha = 0.4  # Mixup混合强度
    cutmix_alpha = 0.4  # CutMix混合强度
    mixup_prob = 0.5  # Mixup应用概率
    cutmix_prob = 0.5  # CutMix应用概率
    use_tta = True  # 测试时增强
    tta_num = 5  # TTA次数
    use_ensemble = True  # 模型集成
    ensemble_models = ['efficientnetv2_rw_m', 'convnext_base', 'swin_base_patch4_window7_224']  # 集成模型列表
    use_cv = False  # 是否使用交叉验证

config = Config()

# %%

# MixUp 增强实现
def mixup_data(x, y, alpha=0.2):
    """执行MixUp数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # 随机打乱索引
    index = t.randperm(batch_size).to(x.device)
    
    # 混合图像
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # 返回混合图像和对应的标签
    return mixed_x, y, y[index], lam

# CutMix 增强实现
def cutmix_data(x, y, alpha=0.4):
    """执行CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # 随机打乱索引
    index = t.randperm(batch_size).to(x.device)

    # 生成随机裁剪区域
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 确保裁剪区域在图像内
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 应用裁剪和混合
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 调整混合比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x_mixed, y, y[index], lam

# 增强版数据集类
class DigitsDataset(Dataset):
    """DigitsDataset with advanced augmentation"""
    def __init__(self, mode='train', size=(128, 224), aug=True):
        super(DigitsDataset, self).__init__()
        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        self.batch_count = 0
        
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
                transforms.RandomRotation(12),  # 增加旋转角度
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15),  # 增强的颜色抖动
                transforms.RandomGrayscale(0.1),
                transforms.RandomAffine(12, translate=(0.15, 0.15), scale=(0.75, 1.25), shear=15),  # 增强变形
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.RandomPosterize(bits=4, p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.2),  # 添加随机均衡化
            ])
        
        trans1.extend(trans0)
        
        # 应用变换
        img = transforms.Compose(trans1)(img)
        
        # 随机擦除 - 训练时的额外增强
        if self.aug and self.mode == 'train':
            img = transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3))(img)
        
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

# CBAM注意力模块 (Channel + Spatial)
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        # 空间注意力
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_out = t.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        x = x * channel_out
        
        # 空间注意力
        avg_out = t.mean(x, dim=1, keepdim=True)
        max_out, _ = t.max(x, dim=1, keepdim=True)
        spatial_out = self.conv(t.cat([avg_out, max_out], dim=1))
        
        return x * spatial_out

# 改进的EfficientNet模型架构
class ImprovedModel(nn.Module):
    def __init__(self, class_num=11, model_name='efficientnetv2_rw_m', dropout=0.3):
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
        elif 'convnext' in model_name:
            feature_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        elif 'swin' in model_name:
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # 共享特征提取层后添加注意力
        self.attention = CBAM(feature_dim)  # 升级到CBAM注意力
        
        # 特征融合层
        self.neck = nn.Sequential(
            nn.Linear(feature_dim, 1536),  # 更大的特征维度
            nn.LayerNorm(1536),  # 使用LayerNorm代替BatchNorm
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout)
        )
        
        # 分支层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1536, class_num)
        self.fc2 = nn.Linear(1536, class_num)
        self.fc3 = nn.Linear(1536, class_num)
        self.fc4 = nn.Linear(1536, class_num)

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

# ViT/Swin Transformer模型架构
class TransformerModel(nn.Module):
    def __init__(self, class_num=11, model_name='swin_base_patch4_window7_224', dropout=0.3):
        super(TransformerModel, self).__init__()
        
        # 使用timm库加载预训练Transformer模型
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        # 获取特征维度
        if 'vit' in model_name:
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif 'swin' in model_name:
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported transformer model: {model_name}")
            
        # 特征融合层 - 更大更复杂
        self.neck = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分支层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1536, class_num)
        self.fc2 = nn.Linear(1536, class_num)
        self.fc3 = nn.Linear(1536, class_num)
        self.fc4 = nn.Linear(1536, class_num)

    def forward(self, img):
        # 特征提取
        feat = self.backbone(img)
        
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
    def __init__(self, smooth=0.15, alpha=0.65, gamma=2):
        super(CombinedLoss, self).__init__()
        self.smooth_loss = LabelSmoothEntropy(smooth=smooth)
        self.focal_loss = FocalLoss(gamma=gamma)
        self.alpha = alpha
        
    def forward(self, preds, targets):
        return self.alpha * self.smooth_loss(preds, targets) + (1 - self.alpha) * self.focal_loss(preds, targets)

class Trainer:
    def __init__(self, model_name=config.model_name, val=True, fold=None):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        self.model_name = model_name
        self.fold = fold
        
        # 创建训练和验证数据集
        if not config.use_cv or fold is None:
            self.train_set = DigitsDataset(mode='train')
            self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, 
                                        num_workers=config.num_workers,
                                        pin_memory=True, persistent_workers=True, \
                                        drop_last=True, collate_fn=self.train_set.collect_fn)
            if val:
                self.val_loader = DataLoader(DigitsDataset(mode='val', aug=False), batch_size=config.batch_size, \
                                            num_workers=config.num_workers, pin_memory=True, drop_last=False, 
                                            persistent_workers=True)
            else:
                self.val_loader = None
        else:
            # 使用交叉验证的数据加载方式
            self._setup_cv_loaders(fold)

        # 创建模型
        if 'vit' in model_name or 'swin' in model_name:
            self.model = TransformerModel(config.class_num, model_name=model_name, dropout=0.3).to(self.device)
        else:
            self.model = ImprovedModel(config.class_num, model_name=model_name, dropout=0.3).to(self.device)
        
        # 使用组合损失函数
        self.criterion = CombinedLoss(smooth=config.smooth, alpha=0.65, gamma=2).to(self.device)
        
        # 使用AdamW优化器
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weights_decay)
        
        # 使用OneCycleLR - 更平滑的学习率调度
        steps_per_epoch = len(self.train_loader)
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            epochs=config.epoches,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 学习率预热阶段占总训练的10%
            div_factor=20,  # 初始学习率是最大学习率的1/20
            final_div_factor=1000  # 最终学习率是最大学习率的1/1000
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
                print(f'Load model from {config.pretrained}, Eval Acc: {acc * 100:.2f}')

    def _setup_cv_loaders(self, fold):
        """设置交叉验证数据加载器"""
        dataset = DigitsDataset(mode='train')
        
        # 获取CV分割
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = list(kf.split(range(len(dataset))))[fold]
        
        # 创建采样器
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            dataset, batch_size=config.batch_size,
            sampler=train_sampler, num_workers=config.num_workers,
            pin_memory=True, drop_last=True
        )
        
        self.val_loader = DataLoader(
            dataset, batch_size=config.batch_size,
            sampler=val_sampler, num_workers=config.num_workers,
            pin_memory=True, drop_last=False
        )

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            print(f"Epoch {epoch+1}/{config.epoches}")
            train_acc = self.train_epoch(epoch)
            
            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                if self.val_loader is not None:
                    val_acc = self.eval()
                    print(f'Validation accuracy: {val_acc:.4f}')
                    
                    # 保存模型
                    if val_acc > self.best_acc:
                        os.makedirs(config.checkpoints, exist_ok=True)
                        
                        if self.fold is not None:
                            save_path = os.path.join(
                                config.checkpoints,
                                f'fold{self.fold}-{self.model_name}-epoch-{epoch+1}-acc-{val_acc*100:.2f}.pth'
                            )
                        else:
                            save_path = os.path.join(
                                config.checkpoints,
                                f'{self.model_name}-epoch-{epoch+1}-acc-{val_acc*100:.2f}.pth'
                            )
                        
                        self.save_model(save_path)
                        print(f'{save_path} saved successfully...')
                        self.best_acc = val_acc
                        self.best_checkpoint_path = save_path

    def calculate_loss(self, pred, label):
        """计算损失"""
        return (
            self.criterion(pred[0], label[:, 0]) +
            self.criterion(pred[1], label[:, 1]) +
            self.criterion(pred[2], label[:, 2]) +
            self.criterion(pred[3], label[:, 3])
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        
        for i, batch in enumerate(tbar):
            if i == 0 and epoch == 0:  # 消除 torch 的 warning
                print(f"Initial lr: {self.optimizer.param_groups[0]['lr']}")
                self.optimizer.step()
                self.lr_scheduler.step()
                print(f"After first step lr: {self.optimizer.param_groups[0]['lr']}")
            
            # 处理batch数据
            if isinstance(batch, tuple) and len(batch) == 2:
                img, label = batch
                img = img.to(self.device)
                label = label.to(self.device)
                
                # 随机应用 MixUp 或 CutMix
                use_mixup = config.use_mixup and random.random() < config.mixup_prob
                use_cutmix = config.use_cutmix and random.random() < config.cutmix_prob
                
                # MixUp和CutMix二选一
                if use_mixup and not use_cutmix:
                    img, label_a, label_b, lam = mixup_data(img, label, alpha=config.mixup_alpha)
                elif use_cutmix:
                    img, label_a, label_b, lam = cutmix_data(img, label, alpha=config.cutmix_alpha)
                
                self.optimizer.zero_grad()
                
                # 混合精度训练
                if config.mixed_precision:
                    with t.amp.autocast('cuda'):
                        pred = self.model(img)
                        
                        # 处理MixUp/CutMix的损失
                        if (use_mixup or use_cutmix):
                            # 对两个标签分别计算损失然后线性组合
                            loss1 = self.calculate_loss(pred, label_a)
                            loss2 = self.calculate_loss(pred, label_b)
                            loss = lam * loss1 + (1 - lam) * loss2
                        else:
                            loss = self.calculate_loss(pred, label)
                    
                    self.scaler.scale(loss).backward()
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(img)
                    
                    # 处理MixUp/CutMix的损失
                    if (use_mixup or use_cutmix):
                        loss1 = self.calculate_loss(pred, label_a)
                        loss2 = self.calculate_loss(pred, label_b)
                        loss = lam * loss1 + (1 - lam) * loss2
                    else:
                        loss = self.calculate_loss(pred, label)
                    
                    loss.backward()
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                self.lr_scheduler.step()
                
                total_loss += loss.item()
                
                # 计算准确率 - 仅使用原始标签进行评估
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
                
                # 如果启用测试时增强
                if config.use_tta:
                    pred = self.tta_predict(img)
                else:
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
    
    def tta_predict(self, img):
        """测试时增强预测"""
        # 原始预测
        pred_orig = self.model(img)
        preds = [p.clone() for p in pred_orig]
        
        # 水平翻转
        img_flip = t.flip(img, dims=[3])
        pred_flip = self.model(img_flip)
        for i in range(len(preds)):
            preds[i] += pred_flip[i]
        
        # 亮度对比度变化
        brightness = transforms.ColorJitter(brightness=0.2)
        img_bright = brightness(img)
        pred_bright = self.model(img_bright)
        for i in range(len(preds)):
            preds[i] += pred_bright[i]
        
        # 随机裁剪缩放
        crop_resize = transforms.RandomResizedCrop(size=(128, 224), scale=(0.9, 1.0))
        img_crop = crop_resize(img)
        pred_crop = self.model(img_crop)
        for i in range(len(preds)):
            preds[i] += pred_crop[i]
        
        # 高斯模糊
        if random.random() > 0.5:
            blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            img_blur = blur(img)
            pred_blur = self.model(img_blur)
            for i in range(len(preds)):
                preds[i] += pred_blur[i]
        else:
            # 透视变换代替模糊
            img_persp = transforms.RandomPerspective(distortion_scale=0.1, p=1.0)(img)
            pred_persp = self.model(img_persp)
            for i in range(len(preds)):
                preds[i] += pred_persp[i]
        
        # 平均所有预测结果
        return [p / 5.0 for p in preds]  # 5种增强方式

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

def write2csv(results, csv_path):
    """
    results(list):
    """
    # 定义输出文件
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: x.split('/')[-1])
    save_name = csv_path
    df.to_csv(save_name, sep=',', index=None)
    print('Results saved to %s' % save_name)

# 测试时增强预测
def tta_predict(model, img, n_augments=4):
    """测试时增强预测函数"""
    # 原始预测
    original_pred = model(img)
    preds = [p.detach() for p in original_pred]
    
    # 水平翻转
    img_flip = t.flip(img, dims=[3])
    flip_pred = model(img_flip)
    for i, p in enumerate(flip_pred):
        preds[i] += p.detach()
    
    # 其他增强版本
    tf_list = [
        transforms.RandomResizedCrop(size=(128, 224), scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    ]
    
    for _ in range(n_augments-1):  # 已经做了翻转，所以-1
        aug = random.choice(tf_list)
        img_aug = aug(img)
        aug_pred = model(img_aug)
        for i, p in enumerate(aug_pred):
            preds[i] += p.detach()
    
    # 平均所有预测结果
    return [p/(n_augments+1) for p in preds]

# 单模型预测
def predict_with_model(model_path, model_name, csv_path=None, use_tta=True):
    test_loader = DataLoader(DigitsDataset(mode='test', aug=False), batch_size=config.batch_size, shuffle=False,\
                    num_workers=config.num_workers, pin_memory=True, drop_last=False, persistent_workers=True)
    results = []
    
    # 加载模型
    if 'vit' in model_name or 'swin' in model_name:
        model = TransformerModel(config.class_num, model_name=model_name).cuda()
    else:
        model = ImprovedModel(config.class_num, model_name=model_name).cuda()
    
    model.load_state_dict(t.load(model_path)['model'])
    print(f'Load model from {model_path} successfully')
    
    # 预测
    model.eval()
    tbar = tqdm(test_loader)
    with t.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.cuda()
            
            if use_tta:
                pred = tta_predict(model, img, n_augments=config.tta_num)
            else:
                pred = model(img)
                
            results += [[name, code] for name, code in zip(img_names, parse2class(pred))]
    
    results = sorted(results, key=lambda x: x[0])
    
    if csv_path:
        write2csv(results, csv_path)
        
    return results

# 模型集成预测
def ensemble_predict(model_paths, csv_path):
    """模型集成预测"""
    test_loader = DataLoader(DigitsDataset(mode='test', aug=False), batch_size=config.batch_size, shuffle=False,\
                    num_workers=config.num_workers, pin_memory=True, drop_last=False, persistent_workers=True)
    
    # 存储所有图像名称
    all_img_names = []
    
    # 保存每个位置所有模型的预测logits
    all_logits = [[] for _ in range(4)]
    
    # 逐个模型预测
    for model_path in model_paths:
        # 从路径中提取模型名称
        if '-epoch-' in model_path:
            model_name = model_path.split('-epoch-')[0].split('/')[-1]
        else:
            # 默认使用配置中的模型
            model_name = config.model_name
        
        print(f"Predicting with model: {model_name}")
        
        # 加载模型
        if 'vit' in model_name or 'swin' in model_name:
            model = TransformerModel(config.class_num, model_name=model_name).cuda()
        else:
            model = ImprovedModel(config.class_num, model_name=model_name).cuda()
        
        model.load_state_dict(t.load(model_path)['model'])
        model.eval()
        
        # 当前模型的logits
        current_logits = [[] for _ in range(4)]
        
        # 收集图像名称 (只在第一个模型时收集)
        if len(all_img_names) == 0:
            with t.no_grad():
                for img, img_names in tqdm(test_loader, desc=f"Collecting image names"):
                    all_img_names.extend(img_names)
        
        # 进行预测
        with t.no_grad():
            for i, (img, _) in enumerate(tqdm(test_loader, desc=f"Model {model_path}")):
                img = img.cuda()
                
                # 使用测试时增强
                if config.use_tta:
                    preds = tta_predict(model, img, config.tta_num)
                else:
                    preds = model(img)
                
                # 保存每个位置的logits
                for pos in range(4):
                    current_logits[pos].append(preds[pos].cpu())
        
        # 拼接批次的logits
        for pos in range(4):
            current_logits[pos] = t.cat(current_logits[pos], dim=0)
            
            if len(all_logits[pos]) == 0:
                all_logits[pos] = current_logits[pos]
            else:
                all_logits[pos] += current_logits[pos]
    
    # 对所有模型预测取平均
    avg_logits = [logits / len(model_paths) for logits in all_logits]
    
    # 获取每个位置的预测结果
    ch1, ch2, ch3, ch4 = avg_logits
    char_list = [str(i) for i in range(10)]
    char_list.append('')
    
    ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)
    ch1 = [char_list[i.item()] for i in ch1]
    ch2 = [char_list[i.item()] for i in ch2]
    ch3 = [char_list[i.item()] for i in ch3]
    ch4 = [char_list[i.item()] for i in ch4]
    
    # 组合最终结果
    results = [[name, c1+c2+c3+c4] for name, c1, c2, c3, c4 in zip(all_img_names, ch1, ch2, ch3, ch4)]
    results = sorted(results, key=lambda x: x[0])
    
    if csv_path:
        write2csv(results, csv_path)
    
    return results

# 使用交叉验证训练
def train_with_cross_validation(n_splits=5):
    """交叉验证训练"""
    model_paths = []
    
    for fold in range(n_splits):
        print(f"\n=========== Training Fold {fold+1}/{n_splits} ===========")
        trainer = Trainer(fold=fold)
        trainer.train()
        model_paths.append(trainer.best_checkpoint_path)
        
    return model_paths

# %%
if __name__ == '__main__':
    print('__name__:', __name__)
    
    # 使用交叉验证训练
    if config.use_cv:
        model_paths = train_with_cross_validation(5)
    
    # 或者训练单个模型
    elif not config.use_ensemble:
        trainer = Trainer()
        trainer.train()
        acc = trainer.eval()
        print('Local validation accuracy: %.2f%%' % (acc * 100))
        
        # 使用标准预测进行推理
        predicts = predict_with_model(
            trainer.best_checkpoint_path, 
            config.model_name, 
            "single_model_result.csv", 
            use_tta=config.use_tta
        )
    
    # 或者训练多个模型进行集成
    else:
        model_paths = []
        
        # 训练多个不同的模型
        for model_name in config.ensemble_models:
            print(f"\n=========== Training Model: {model_name} ===========")
            config.model_name = model_name
            trainer = Trainer(model_name=model_name)
            trainer.train()
            model_paths.append(trainer.best_checkpoint_path)
            
        # 使用模型集成进行预测
        ensemble_predict(model_paths, "ensemble_result.csv")