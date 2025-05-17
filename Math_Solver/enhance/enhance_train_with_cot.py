import json
import requests
import time
import re
from tqdm import tqdm
import os

# DeerAPI配置
API_KEY = "sk-LuHa4Yaohe2CMVymp1tDPQuklWH978jmDt4TFhWB6ZifeHVu"  # 替换为你的实际密钥
API_URL = "https://api.deerapi.com/v1/chat/completions"

def get_cot_explanation(question, answer):
    """
    使用GPT-4o-mini生成解题步骤
    """
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
        "model": "gpt-4o-mini",
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
            return explanation
        except Exception as e:
            print(f"尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("所有尝试均失败，返回简单答案")
                return f"经过计算，答案：{answer}"

def main():
    # 加载原始数据
    train_file_path = "train.json"
    output_file_path = "train_with_cot.json"
    
    print(f"读取原始训练数据: {train_file_path}")
    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 创建输出目录(如果需要)
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 增强数据
    enhanced_data = []
    total = len(train_data)
    print(f"开始处理 {total} 条数据...")
    
    for i, item in enumerate(tqdm(train_data)):
        question = item["question"]
        answer = item["answer"]
        
        # 生成解题步骤
        cot_explanation = get_cot_explanation(question, answer)
        
        # 构建增强数据项
        enhanced_item = {
            "id": item["id"],
            "question": question,
            "answer": cot_explanation,
            "instruction": "请详细分析并解答以下问题，给出计算步骤，最后用'答案：数字'的格式给出答案"
        }
        enhanced_data.append(enhanced_item)
        
        # 每100个样本保存一次进度(以防程序中断)
        if (i + 1) % 100 == 0:
            temp_path = f"{output_file_path}.temp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            print(f"已处理 {i+1}/{total} 条数据，临时保存到 {temp_path}")
    
    # 保存最终结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据增强完成！结果保存到: {output_file_path}")
    print(f"增强前数据条数: {len(train_data)}, 增强后数据条数: {len(enhanced_data)}")

if __name__ == "__main__":
    main()