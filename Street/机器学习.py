import os
from glob import glob
import torch as t
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch 
import torch.nn.functional as F
import json
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet18, resnet34
from torchsummary import summary

# 设置随机种子确保可复现性
t.random.manual_seed(0)
t.cuda.manual_seed_all(0)
t.backends.cudnn.benchmark = True
t.backends.cudnn.deterministic = True

# 构建数据集路径索引
dataset_path = "./dataset"
data_dir = {
    'train_data': f'{dataset_path}/mchar_train/',
    'val_data': f'{dataset_path}/mchar_val/',
    'test_data': f'{dataset_path}/mchar_test_a/',
    'train_label': f'{dataset_path}/mchar_train.json',
    'val_label': f'{dataset_path}/mchar_val.json',
    'submit_file': f'{dataset_path}/mchar_sample_submit_A.csv'
}

# 数据分析函数
def data_summary():
    train_list = glob(data_dir['train_data']+'*.png')
    test_list = glob(data_dir['test_data']+'*.png')
    val_list = glob(data_dir['val_data']+'*.png')
    print('train image counts: %d'%len(train_list))
    print('val image counts: %d'%len(val_list))
    print('test image counts: %d'%len(test_list))

def look_train_json():
    with open(data_dir['train_label'], 'r', encoding='utf-8') as f:
        content = f.read()
    content = json.loads(content)
    print(content['000000.png'])

def look_submit():
    df = pd.read_csv(data_dir['submit_file'], sep=',')
    print(df.head(5))

def plot_samples():
    imgs = glob(data_dir['train_data']+'*.png')
    fig, ax = plt.subplots(figsize=(12, 8), ncols=2, nrows=2)
    marks = json.loads(open(data_dir['train_label'], 'r').read())

    for i in range(4):
        img_name = os.path.split(imgs[i])[-1]
        mark = marks[img_name]
        img = Image.open(imgs[i])
        img = np.array(img)

        bboxes = np.array(
            [mark['left'],
            mark['top'],
            mark['width'],
            mark['height']]
        )
        ax[i//2, i%2].imshow(img)
        for j in range(len(mark['label'])):
            rect = patch.Rectangle(bboxes[:, j][:2], bboxes[:, j][2], bboxes[:, j][3], facecolor='none', edgecolor='r')
            ax[i//2, i%2].text(bboxes[:, j][0], bboxes[:, j][1], mark['label'][j])
            ax[i//2, i%2].add_patch(rect)
    plt.show()

def img_size_summary():
    sizes = []
    for img in glob(data_dir['train_data']+'*.png'):
        img = Image.open(img)
        sizes.append(img.size)
    sizes = np.array(sizes)

    plt.figure(figsize=(10, 8))
    plt.scatter(sizes[:, 0], sizes[:, 1])
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('image width-height summary')
    plt.show()

    return np.mean(sizes, axis=0), np.median(sizes, axis=0)

def bbox_summary():
    marks = json.loads(open(data_dir['train_label'], 'r').read())
    bboxes = []
    for img, mark in marks.items():
        for i in range(len(mark['label'])):
            bboxes.append([mark['left'][i], mark['top'][i], mark['width'][i], mark['height'][i]])
    bboxes = np.array(bboxes)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(bboxes[:, 2], bboxes[:, 3])
    ax.set_title('bbox width-height summary')
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    plt.show()
    return np.mean(bboxes), np.median(bboxes)

def label__nums_summary():
    marks = json.load(open(data_dir['train_label'], 'r'))
    dicts = {i: 0 for i in range(10)}
    for img, mark in marks.items():
        for lb in mark['label']:
            dicts[lb] += 1
    
    xticks = list(range(10))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(x=list(dicts.keys()), height=list(dicts.values()))
    ax.set_xticks(xticks)
    plt.show()
    return dicts

def label_summary():
    marks = json.load(open(data_dir['train_label'], 'r'))
    dicts = {}
    for img, mark in marks.items():
        if len(mark['label']) not in dicts:
            dicts[len(mark['label'])] = 0
        dicts[len(mark['label'])] += 1
    dicts = sorted(dicts.items(), key=lambda x: x[0])
    for k, v in dicts:
        print('%d个数字的图片数目: %d'%(k, v))

# 网络配置信息
class Config:
    batch_size = 64
    lr = 1e-2
    momentum = 0.9
    weights_decay = 1e-4
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 1
    print_interval = 50
    checkpoints = './checkpoints/'
    pretrained = None
    start_epoch = 0
    epoches = 100
    smooth = 0.1
    erase_prob = 0.5
    num_workers = 8

config = Config()

# 构建数据集
class DigitsDataset(Dataset):
    def __init__(self, mode='train', size=(128, 256), aug=True):
        super(DigitsDataset, self).__init__()
        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        self.batch_count = 0
        if mode == 'test':
            self.imgs = glob(data_dir['test_data']+'*.png')
            self.labels = None
        else:
            labels = json.load(open(data_dir['%s_label'%mode], 'r'))
            imgs = glob(data_dir['%s_data'%mode]+'*.png')
            self.imgs = [(img, labels[os.path.split(img)[-1]]) for img in imgs \
                   if os.path.split(img)[-1] in labels]

    def __getitem__(self, idx):
        if self.mode != 'test':
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None
        img = Image.open(img)
        trans0 = [                
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        min_size = self.size[0] if (img.size[1] / self.size[0]) < ((img.size[0] / self.size[1])) else self.size[1]
        trans1 = [
            transforms.Resize(128),    
            transforms.CenterCrop((128, self.width))
        ]
        if self.aug:
            trans1.extend([
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomGrayscale(0.1),
                transforms.RandomAffine(15,translate=(0.05, 0.1), shear=5)
            ])
        trans1.extend(trans0)
        if self.mode != 'test':
            return transforms.Compose(trans1)(img), t.tensor(label['label'][:4] + (4 - len(label['label']))*[10]).long()
        else:
            # trans1.append(transforms.RandomErasing(scale=(0.02, 0.1)))
            return transforms.Compose(trans1)(img), self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))
        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels)

# MobileNet模型实现 - baseline中不存在的模型
class DigitsMobilenet(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsMobilenet, self).__init__()
        self.net = mobilenet_v2(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(1280)
        self.fc1 = nn.Linear(1280, class_num)
        self.fc2 = nn.Linear(1280, class_num)
        self.fc3 = nn.Linear(1280, class_num)
        self.fc4 = nn.Linear(1280, class_num)

    def forward(self, img):
        features = self.avgpool(self.net(img)).view(-1, 1280)
        features = self.bn(features)
        fc1 = self.fc1(features)
        fc2 = self.fc2(features)
        fc3 = self.fc3(features)
        fc4 = self.fc4(features)
        return fc1, fc2, fc3, fc4

# ResNet18模型实现 - 比baseline的ResNet50更轻量
class DigitsResnet18(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsResnet18, self).__init__()
        self.net = resnet18(pretrained=True)
        self.net.fc = nn.Identity()
        self.bn = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, class_num)
        self.fc2 = nn.Linear(512, class_num)
        self.fc3 = nn.Linear(512, class_num)
        self.fc4 = nn.Linear(512, class_num)

    def forward(self, img):
        features = self.net(img).squeeze()
        fc1 = self.fc1(features)
        fc2 = self.fc2(features)
        fc3 = self.fc3(features)
        fc4 = self.fc4(features)
        return fc1, fc2, fc3, fc4

# 标签平滑处理
class LabelSmoothEntropy(nn.Module):
    def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
        super(LabelSmoothEntropy, self).__init__()
        self.size_average = size_average
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, preds, targets):
        lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[0] - 1)
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
            raise NotImplementedError

# 训练器类
class Trainer:
    def __init__(self, val=True, model_type='mobilenet'):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        print(f"Using device: {self.device}")
        
        self.train_set = DigitsDataset(mode='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, 
                                     num_workers=config.num_workers, pin_memory=True, 
                                     drop_last=True, collate_fn=self.train_set.collect_fn)
        
        if val:
            self.val_loader = DataLoader(DigitsDataset(mode='val', aug=False), 
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers, 
                                       pin_memory=True, drop_last=False)
        else:
            self.val_loader = None

        # 选择模型类型
        if model_type == 'mobilenet':
            self.model = DigitsMobilenet(config.class_num).to(self.device)
        else:
            self.model = DigitsResnet18(config.class_num).to(self.device)

        self.criterion = LabelSmoothEntropy().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, 10, 2, eta_min=10e-4)
        self.best_acc = 0
        self.best_checkpoint_path = ""

        if config.pretrained is not None:
            self.load_model(config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print(f'Load model from {config.pretrained}, Eval Acc: {acc * 100:.2f}%')

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            print(f'Epoch {epoch+1}/{config.epoches}')
            acc = self.train_epoch(epoch)
            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                if self.val_loader is not None:
                    acc = self.eval() 

                if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(config.checkpoints, f'epoch-{epoch+1}-acc-{acc*100:.2f}.pth')
                    self.save_model(save_path)
                    print(f'{save_path} saved successfully...')
                    self.best_acc = acc
                    self.best_checkpoint_path = save_path

    def train_epoch(self, epoch):
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        self.model.train()
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(img)
            loss = self.criterion(pred[0], label[:, 0]) + \
                  self.criterion(pred[1], label[:, 1]) + \
                  self.criterion(pred[2], label[:, 2]) + \
                  self.criterion(pred[3], label[:, 3])
                  
            total_loss += loss.item()
            loss.backward() 
            self.optimizer.step()
            
            temp = t.stack([
                pred[0].argmax(1) == label[:, 0],
                pred[1].argmax(1) == label[:, 1],
                pred[2].argmax(1) == label[:, 2],
                pred[3].argmax(1) == label[:, 3],
            ], dim=1)

            corrects += t.all(temp, dim=1).sum().item()
            tbar.set_description(f'loss: {loss/(i+1):.3f}, acc: {corrects*100/((i+1)*config.batch_size):.3f}')
            
            if (i + 1) % config.print_interval == 0:
                self.lr_scheduler.step()
                
        return corrects*100/((i+1)*config.batch_size)

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
                tbar.set_description(f'Val Acc: {corrects*100/((i+1)*config.batch_size):.2f}')
                
        self.model.train()
        return corrects / (len(self.val_loader) * config.batch_size)

    def save_model(self, save_path, save_opt=False, save_config=False):
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
        else:
            dicts = t.load(load_path)['model']
            keys = list(net.state_dict().keys())
            values = list(dicts.values())
            new_dicts = {k: v for k, v in zip(keys, values)}
            self.model.load_state_dict(new_dicts)

        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])

        if save_config:
            for k, v in dicts['config'].items():
                config.__setattr__(k, v)

# 模型集成评估函数
def stack_eval(models, val_loader=None):
    if val_loader is None:
        val_loader = DataLoader(DigitsDataset(mode='val', aug=False), 
                              batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True, drop_last=False)
    
    for model in models:
        model.eval()
        model.to(t.device('cuda'))
        
    corrects = 0
    with t.no_grad():
        tbar = tqdm(val_loader)
        for i, (img, label) in enumerate(tbar):
            img = img.cuda()
            label = label.cuda()
            
            # 平均所有模型的预测结果
            preds = [0, 0, 0, 0]
            for model in models:
                model_preds = model(img)
                for j in range(4):
                    preds[j] += model_preds[j]
                    
            # 每个位置的预测取平均
            for j in range(4):
                preds[j] = preds[j] / len(models)
                
            temp = t.stack([
                preds[0].argmax(1) == label[:, 0],
                preds[1].argmax(1) == label[:, 1],
                preds[2].argmax(1) == label[:, 2],
                preds[3].argmax(1) == label[:, 3],
            ], dim=1)
            
            corrects += t.all(temp, dim=1).sum().item()
            tbar.set_description(f'Ensemble Val Acc: {corrects*100/((i+1)*config.batch_size):.2f}')
            
    return corrects / (len(val_loader) * config.batch_size)

# 模型集成预测函数
def ensemble_predict(model_paths, csv_path):
    test_loader = DataLoader(DigitsDataset(mode='test', aug=False), 
                          batch_size=config.batch_size, shuffle=False,
                          num_workers=config.num_workers, pin_memory=True, drop_last=False)
    
    models = []
    for path in model_paths:
        if 'mobilenet' in path.lower():
            model = DigitsMobilenet(config.class_num).cuda()
        else:
            model = DigitsResnet18(config.class_num).cuda()
        model.load_state_dict(t.load(path)['model'])
        models.append(model)
        print(f'Loaded model from {path}')
        
    results = []
    for model in models:
        model.eval()
    
    tbar = tqdm(test_loader)
    with t.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.cuda()
            
            # 平均所有模型的预测结果
            preds = [0, 0, 0, 0]
            for model in models:
                model_preds = model(img)
                for j in range(4):
                    preds[j] += model_preds[j]
                    
            # 每个位置的预测取平均
            for j in range(4):
                preds[j] = preds[j] / len(models)
                
            results += [[name, code] for name, code in zip(img_names, parse2class(preds))]
    
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results

# 单模型预测函数
def predicts(model_path, csv_path):
    test_loader = DataLoader(DigitsDataset(mode='test', aug=False), 
                          batch_size=config.batch_size, shuffle=False,
                          num_workers=config.num_workers, pin_memory=True, drop_last=False)
    results = []
    
    # 根据模型路径确定模型类型
    if 'mobilenet' in model_path.lower():
        model = DigitsMobilenet(config.class_num).cuda()
    else:
        model = DigitsResnet18(config.class_num).cuda()
        
    model.load_state_dict(t.load(model_path)['model'])
    print(f'Load model from {model_path} successfully')
    
    tbar = tqdm(test_loader)
    model.eval()
    with t.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.cuda()
            pred = model(img)
            results += [[name, code] for name, code in zip(img_names, parse2class(pred))]
    
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results

# 解析模型预测结果
def parse2class(prediction):
    ch1, ch2, ch3, ch4 = prediction
    char_list = [str(i) for i in range(10)]
    char_list.append('')
    
    ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)
    ch1, ch2, ch3, ch4 = [char_list[i.item()] for i in ch1], [char_list[i.item()] for i in ch2], \
                      [char_list[i.item()] for i in ch3], [char_list[i.item()] for i in ch4]
                      
    res = [c1+c2+c3+c4 for c1, c2, c3, c4 in zip(ch1, ch2, ch3, ch4)]             
    return res

# 结果写入CSV文件
def write2csv(results, csv_path):
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: x.split('/')[-1])
    df.to_csv(csv_path, sep=',', index=None)
    print(f'Results saved to {csv_path}')

# 主程序入口
if __name__ == '__main__' and False:
    # 数据分析（可选）
    # data_summary()
    # mean, median = img_size_summary()
    # print(f'Image size - mean: {mean}, median: {median}')
    
    # 创建训练器并训练模型
    trainer = Trainer(val=True, model_type='mobilenet')  # 也可以选择'resnet18'
    trainer.train()
    
    # 验证最终模型准确率
    acc = trainer.eval()
    print(f'Final validation accuracy: {acc * 100:.2f}%')
    
    # 生成预测结果
    if trainer.best_checkpoint_path:
        predicts(trainer.best_checkpoint_path, "results.csv")
        
    # 模型集成预测（可选）
    # 如果有多个训练好的模型可以尝试集成
    # model_paths = glob('./checkpoints/*.pth')
    # ensemble_predict(model_paths, "ensemble_results.csv")

if __name__ == '__main__':
    best_model_path = './checkpoints/epoch-34-acc-67.67.pth'
    predicts(best_model_path, "results.csv")