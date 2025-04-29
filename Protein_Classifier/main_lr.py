import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from fea import feature_extraction
from sklearn.linear_model import LogisticRegression
from Bio.PDB import PDBParser


class LRModel:
    # todo:
    """
        Initialize Logistic Regression (from sklearn) model.

    """

    """
        Train the Logistic Regression model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        predictions = self.model.predict(data)
        accuracy = np.mean(predictions == targets)
        return accuracy


class LRFromScratch:
    # todo:
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=0.01):
        # 初始化参数
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -700, 700)  # 防止溢出
        return 1 / (1 + np.exp(-z))

    def train(self, train_data, train_targets):
        # 标准化特征
        self.mean = np.mean(train_data, axis=0)
        self.std = np.std(train_data, axis=0) + 1e-8  # 避免除以零
        X_normalized = (train_data - self.mean) / self.std
        
        # 初始化权重和偏置
        n_samples, n_features = X_normalized.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for _ in range(self.max_iterations):
            linear_pred = np.dot(X_normalized, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            dw = (1/n_samples) * np.dot(X_normalized.T, (predictions - train_targets)) + (self.regularization * self.weights)
            db = (1/n_samples) * np.sum(predictions - train_targets)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def evaluate(self, data, targets):
        # 对测试数据也应用相同的标准化
        X_normalized = (data - self.mean) / self.std
        # 使用训练好的模型进行预测
        linear_pred = np.dot(X_normalized, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        # 将概率转换为二进制预测
        binary_predictions = (predictions > 0.5).astype(int)
        # 计算准确率
        accuracy = np.mean(binary_predictions == targets)
        return accuracy


def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # 处理55个任务
        task_col = cast.iloc[:, task]
        
        # 跳过没有足够数据的任务
        if task_col.isna().all():
            continue
            
        # 获取非缺失数据的索引
        valid_indices = ~task_col.isna()
        valid_labels = task_col[valid_indices].values
        valid_data = diagrams[valid_indices]
        
        # 根据标签划分训练集和测试集
        train_indices = np.where((valid_labels == 1) | (valid_labels == 2))[0]
        test_indices = np.where((valid_labels == 3) | (valid_labels == 4))[0]
        
        # 确保有训练和测试数据
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
            
        # 划分数据
        train_data = valid_data[train_indices]
        test_data = valid_data[test_indices]
        
        # 获取并二值化标签 (1,3 -> 1正例; 2,4 -> 0负例)
        train_labels = valid_labels[train_indices]
        test_labels = valid_labels[test_indices]
        
        # 将标签转换为二值形式：正例(1,3)为1，负例(2,4)为0
        train_targets = np.where((train_labels == 1) | (train_labels == 3), 1, 0)

        test_targets = np.where((test_labels == 3) | (test_labels == 1), 1, 0)
        
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    
    return data_list, target_list

def main(args):

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []
    

    # model = LRModel()
    model = LRFromScratch()
    # model = LRFromScratch(learning_rate=0.005, max_iterations=2000, regularization=0.001)
    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)


    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR Training and Evaluation")
    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    args = parser.parse_args()
    main(args)

