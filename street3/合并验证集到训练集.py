import os
import json
import shutil
from glob import glob
from tqdm import tqdm

# 数据目录
dataset_path = "./dataset"
train_data_dir = f'{dataset_path}/mchar_train/'
val_data_dir = f'{dataset_path}/mchar_val/'
train_label_path = f'{dataset_path}/mchar_train.json'
val_label_path = f'{dataset_path}/mchar_val.json'
merged_label_path = f'{dataset_path}/mchar_train_merged.json'

def merge_train_val():
    # 确保目录存在
    os.makedirs(dataset_path, exist_ok=True)
    
    # 读取训练集和验证集标注
    with open(train_label_path, 'r', encoding='utf-8') as f:
        train_labels = json.load(f)
    
    with open(val_label_path, 'r', encoding='utf-8') as f:
        val_labels = json.load(f)
    
    # 获取训练集图片数量
    train_images = glob(train_data_dir + '*.png')
    train_count = len(train_images)
    print(f"训练集图片数量: {train_count}")
    
    # 获取验证集图片
    val_images = glob(val_data_dir + '*.png')
    val_count = len(val_images)
    print(f"验证集图片数量: {val_count}")
    
    # 创建新的标注字典
    merged_labels = train_labels.copy()
    
    # 记录处理的文件数量
    processed_count = 0
    
    print("开始合并数据集...")
    # 遍历验证集图片
    for val_img_path in tqdm(val_images):
        # 获取原始文件名
        val_img_name = os.path.basename(val_img_path)
        
        # 创建新文件名 (030000.png, 030001.png, ...)
        new_img_name = f"{train_count + processed_count:06d}.png"
        
        # 目标路径
        target_path = os.path.join(train_data_dir, new_img_name)
        
        # 复制图片
        shutil.copy2(val_img_path, target_path)
        
        # 添加到合并的标注
        if val_img_name in val_labels:
            merged_labels[new_img_name] = val_labels[val_img_name]
        
        processed_count += 1
    
    # 保存合并后的标注
    with open(merged_label_path, 'w', encoding='utf-8') as f:
        json.dump(merged_labels, f)
    
    # 备份原始训练标注
    shutil.copy2(train_label_path, f"{train_label_path}.bak")
    
    # 将合并后的标注替换为训练集标注
    shutil.move(merged_label_path, train_label_path)
    
    print(f"成功处理 {processed_count} 个验证集图片")
    print(f"合并后的训练集图片数量: {train_count + processed_count}")
    print(f"合并后的标注已保存到 {train_label_path}")
    print(f"原始训练集标注已备份到 {train_label_path}.bak")

if __name__ == "__main__":
    merge_train_val()