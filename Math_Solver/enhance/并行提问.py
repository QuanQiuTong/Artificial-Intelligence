import json
import requests
import time
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# DeerAPI配置
API_KEY = "sk-i7Dn4nffm4vTlHs15gKVAbORXXC3Kn4OiwRDmJug8dkg6QeU"
API_URL = "https://api.deerapi.com/v1/chat/completions"

# 线程池大小 - 控制并发请求数
MAX_WORKERS = 40
# 每个批次的大小
BATCH_SIZE = 100

def get_cot_explanation(item):
    """
    使用GPT-4o-mini生成解题步骤
    """
    question = item["question"]
    answer = item["answer"]
    id = item["id"]
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'User-Agent': 'DeerAPI/1.0.0 (https://api.deerapi.com)',
        'Content-Type': 'application/json'
    }
    
    prompt = f"""请详细分析并解答以下小学数学题，要给出清晰的计算步骤:

问题: {question}

你需要:
1. 解析问题中的已知条件
2. 分析每一步计算过程
3. 确保最终答案为: {answer}
4. 最后使用"答案：{answer}"格式给出答案

请确保计算步骤清晰且正确。"""

    payload = json.dumps({
        "model": "gpt-4.1-nano-2025-04-14", # "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "你是一个擅长小学数学的助手，请提供清晰的解题思路和步骤。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    
    # 添加重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=payload)
            response.raise_for_status()
            result = response.json()
            explanation = result['choices'][0]['message']['content']
            
            # 构建增强数据项
            enhanced_item = {
                "id": id,
                "question": question,
                "answer": explanation,
                "instruction": "请详细分析并解答以下问题，给出计算步骤，最后用'答案：数字'的格式给出答案"
            }
            return enhanced_item
        except Exception as e:
            print(f"ID {id} - 尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"ID {id} - 所有尝试均失败，返回简单答案")
                return {
                    "id": id,
                    "question": question,
                    "answer": f"经过计算，答案：{answer}",
                    "instruction": "请详细分析并解答以下问题，给出计算步骤，最后用'答案：数字'的格式给出答案"
                }

def process_batch(batch, output_path, batch_number):
    """处理一批数据并保存结果"""
    enhanced_batch = []
    
    # 使用线程池并行处理批次中的项目
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(get_cot_explanation, item): item for item in batch}
        
        # 收集结果
        for future in tqdm(concurrent.futures.as_completed(future_to_item), 
                          total=len(future_to_item),
                          desc=f"处理批次 {batch_number}"):
            try:
                enhanced_item = future.result()
                enhanced_batch.append(enhanced_item)
            except Exception as e:
                print(f"处理项目时出错: {str(e)}")
    
    # 保存批次结果
    batch_path = f"{output_path}.batch{batch_number}"
    with open(batch_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_batch, f, ensure_ascii=False, indent=2)
    
    print(f"批次 {batch_number} 处理完成，保存到 {batch_path}")
    return enhanced_batch

def main():
    # 加载原始数据
    train_file_path = "train.json"
    output_file_path = "train_with_cot.json"
    
    # 检查是否有已处理的批次文件
    batch_files = [f for f in os.listdir('.') if f.startswith(f"{output_file_path}.batch")]
    processed_data = []
    
    if batch_files:
        print(f"发现已处理的批次文件: {len(batch_files)} 个")
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    processed_data.extend(batch_data)
                print(f"从 {batch_file} 加载了 {len(batch_data)} 条数据")
            except Exception as e:
                print(f"读取批次文件 {batch_file} 失败: {str(e)}")
                
        # 按ID排序已处理数据
        processed_ids = set(item["id"] for item in processed_data)
        print(f"已处理 {len(processed_ids)} 条数据")
    else:
        processed_ids = set()
        
    print(f"读取原始训练数据: {train_file_path}")
    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 筛选未处理的数据
    unprocessed_data = [item for item in train_data if item["id"] not in processed_ids]
    print(f"剩余 {len(unprocessed_data)} 条数据待处理")
    
    # 分批处理
    batches = [unprocessed_data[i:i+BATCH_SIZE] for i in range(0, len(unprocessed_data), BATCH_SIZE)]
    
    # 处理每个批次
    for i, batch in enumerate(batches):
        batch_number = i + 1000
        batch_result = process_batch(batch, output_file_path, batch_number)
        processed_data.extend(batch_result)
        
        # 每5个批次保存一次完整结果
        if batch_number % 5 == 0 or batch_number == len(batches):
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            print(f"已处理 {batch_number}/{len(batches)} 批次，保存完整结果到 {output_file_path}")
    
    # 保存最终结果
    if processed_data:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据增强完成！结果保存到: {output_file_path}")
        print(f"增强前数据条数: {len(train_data)}, 增强后数据条数: {len(processed_data)}")
    else:
        print("没有处理任何数据")

if __name__ == "__main__":
    main()