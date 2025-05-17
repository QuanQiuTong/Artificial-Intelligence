"""
We have train_with_cot.json.batch{i} files, 
each containing 100 records, and is in the form of

[
  {
    "id": "0",
    "question": "食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？",
    "answer": "当然可以！下面是详细的解题思路和步骤：\n\n**1. 解析已知条件：**  \n- 食堂运来的萝卜重量：105千克  \n- 运来的青菜是萝卜的3倍  \n\n**2. 分析每一步计算过程：**  \n1. 已知萝卜的重量是105千克。  \n2. 青菜的重量是萝卜的3倍，所以可以用“萝卜的重量 × 3”来计算青菜的重量。  \n3. 计算：青菜重量 = 105千克 × 3  \n\n**3. 进行乘法计算：**  \n105 × 3 = (100 + 5) × 3  \n= 100 × 3 + 5 × 3  \n= 300 + 15  \n= 315\n\n**4. 确认答案**：青菜的重量是315千克。\n\n---\n\n**答案：315**",
    "instruction": "请详细分析并解答以下问题，给出计算步骤，最后用'答案：数字'的格式给出答案"
  },
  {
    "id": "31",
    "question": "一个书架有6层，平均每层放25本书，3个书架一共可以放多少本书？",
    "answer": "当然可以！我们来逐步分析和计算这个问题。\n\n1. 解析问题中的已知条件：\n- 书架的层数：6层\n- 每一层放的书的平均数：25本\n- 有3个这样的书架（总共3个书架）\n\n2. 逐步计算过程：\n- 首先，计算一个书架上的总书数：\n  每层书的数量 × 层数 = 25 × 6 = 150（本）\n\n- 接下来，计算3个书架一共可以放的书的总数：\n  单个书架的书数 × 书架的个数 = 150 × 3 = 450（本）\n\n3. 核对最终答案：\n最终答案为：450。本题的计算符合题目设定，验证无误。\n\n答案：450",
    "instruction": "请详细分析并解答以下问题，给出计算步骤，最后用'答案：数字'的格式给出答案"
  },
    ...
]
"""

# 合并所有batch文件到1个json文件中，按照id排序
import json
import os
from tqdm import tqdm

""" def merge_json_files(input_dir, output_file):
    all_data = []
    batch_files = [f for f in os.listdir(input_dir) if f.startswith("train_with_cot.json.batch")]

    # 读取所有批次文件
    for batch_file in tqdm(batch_files, desc="合并文件"):
        with open(os.path.join(input_dir, batch_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)

    # 按照id排序
    all_data.sort(key=lambda x: int(x["id"]))

    # 保存到新的json文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"合并完成，保存到 {output_file}") """

all_data = []
for i in range(1,121):
    with open(f"train_with_cot.json.batch{i}", 'r', encoding='utf-8') as f:
        data = json.load(f)
        data.sort(key=lambda x: int(x["id"]))
        all_data.extend(data)
    
# 检验是否有重复的id
ids = [item["id"] for item in all_data]
unique_ids = set(ids)
print(f"总条目数: {len(all_data)}")
print(f"唯一ID数: {len(unique_ids)}")
print(f"重复ID数: {len(all_data) - len(unique_ids)}")

# 找出重复的id
duplicate_ids = set([item for item in ids if ids.count(item) > 1])
print(f"重复的ID: {duplicate_ids}")

# with open("train_with_cot.json", 'w', encoding='utf-8') as f:
#     json.dump(all_data, f, ensure_ascii=False, indent=2)