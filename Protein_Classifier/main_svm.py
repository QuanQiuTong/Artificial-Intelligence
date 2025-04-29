import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from fea import feature_extraction
from sklearn.svm import SVC
from Bio.PDB import PDBParser


class SVMModel:
# Todo
    """
        Initialize Support Vector Machine (SVM from sklearn) model.

        Parameters:
        - C (float): Regularization parameter. Default is 1.0.
        - kernel (str): Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
    """

    """
        Train the Support Vector Machine model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    """
        Evaluate the performance of the Support Vector Machine model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def __init__(self, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel, random_state=42)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        predictions = self.model.predict(data)
        accuracy = np.mean(predictions == targets)
        return accuracy

class SVMFromScratch:
    def __init__(self, lr=0.001, num_iter=200, c=1):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = 0
        self.C = c
        self.mean = None
        self.std = None

    def compute_loss(self, y, predictions):
        """
        SVM Loss function:
        hinge_loss = 1/2 * ||w||^2 + C * sum(max(0, 1 - y * z))
        """
        num_samples = y.shape[0]
        # 1. Hinge Loss: max(0, 1 - y * z)
        hinge_loss = np.maximum(0, 1 - y * predictions)
        hinge_loss_sum = np.sum(hinge_loss)  # sum
        # 2. Regularization term: 1/2 * ||w||^2
        regularization = 0.5 * np.sum(self.weights ** 2)
        total_loss = regularization + self.C * hinge_loss_sum
        loss = total_loss / num_samples
        return loss
    
    def standardize(self, X):
        return (X - self.mean) / self.std

    #  todo:
    def train(self, train_data, train_targets):
        X = np.array(train_data)
        y = np.array(train_targets)

        # Convert tags to 1 and -1
        y = np.where(y == 0, -1, 1)

        # Standardize Data
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-3   
        X = self.standardize(X)  

        num_samples, num_features = X.shape
        # Initialize weights and biases
        self.weights = np.zeros(num_features)
        self.bias = 0   
        # Gradient descent updates parameters
        for iteration in range(self.num_iter):
            for i in range(num_samples):  
                # 计算SVM输出
                svm_output = np.dot(X[i], self.weights) + self.bias
                
                # 计算梯度
                if y[i] * svm_output >= 1:
                    # 正确分类且间隔大于1，只需要优化正则项
                    dw = self.weights
                    db = 0
                else:
                    # 间隔小于1或分类错误，需要同时优化正则项和hinge loss
                    dw = self.weights - self.C * y[i] * X[i]
                    db = -self.C * y[i]

                # ### Update weights and bias
                self.weights = self.weights - self.lr * dw
                self.bias = self.bias - self.lr * db

            
            if iteration % 10 == 0:
                predictions = np.dot(X, self.weights) + self.bias
                loss = self.compute_loss(y, predictions)
                print(f"Iteration {iteration}, Loss: {loss}")
        

    def predict(self, X):
        # sign 
        X = self.standardize(X)  
        svm_model = np.dot(X, self.weights) + self.bias
        predictions = np.sign(svm_model)  
        return predictions

    def evaluate(self, data, targets):
        X = np.array(data)
        y = np.array(targets)
        y = np.where(y == 0, -1, 1)
        predictions = self.predict(X)
        return np.mean(predictions == y)
    

def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    
    for task in range(1, 56):
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
    
    ## Todo:Model Initialization 
    ## You can also consider other different settings
    model = SVMModel(C=args.C,kernel=args.kernel)
    # model = SVMFromScratch()


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
    parser = argparse.ArgumentParser(description="SVM Training and Evaluation")
    parser.add_argument('--C', type=float, default=1.0, help="Regularization parameter")
    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    parser.add_argument('--kernel', type=str, default='rbf', help="Kernel type for SVM")
    args = parser.parse_args()
    main(args)

