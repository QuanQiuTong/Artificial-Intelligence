# %% 导入必要的库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# %% 创建自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        
        if self.transform:
            x = self.transform(x)
            
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x

# %% 数据加载和预处理
def load_data(data_path, test_size=0.2, batch_size=32):
    """
    加载数据并进行预处理
    """
    # 假设从CSV文件加载数据
    # 实际项目中可能需要根据具体数据源调整
    df = pd.read_csv(data_path)
    
    # 分离特征和标签
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader, scaler

# %% 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        return x

# %% 训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    训练模型函数
    """
    # 将模型移动到指定设备
    model.to(device)
    
    # 存储训练历史记录
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算epoch的平均损失和准确率
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # 保存历史记录
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return model, history

# %% 评估函数
def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型函数
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失和准确率
    test_loss = test_loss / total
    test_acc = correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.show()
    
    return test_loss, test_acc, all_preds, all_labels

# %% 可视化函数
def plot_training_history(history):
    """
    绘制训练历史
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'])
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

# %% 主函数
def main():
    # 配置参数
    data_path = "mchar_data_list_0515.csv"  # 请替换为实际数据路径
    input_dim = 10  # 特征维度
    hidden_dim = 128  # 隐藏层维度
    output_dim = 2  # 输出类别数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    
    # 加载数据
    train_loader, test_loader, scaler = load_data(
        data_path=data_path,
        test_size=0.2,
        batch_size=batch_size
    )
    
    # 创建模型
    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print("模型已保存到 'model.pth'")

# %% 如果是主程序则执行
if __name__ == "__main__":
    main()