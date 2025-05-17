import json
import re

# 加载增强后的数据
with open("train_with_cot.json", 'r', encoding='utf-8') as f:
    enhanced_data = json.load(f)

with open("train.json", 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 修改答案格式和instruction
formatted_data = []
for item in enhanced_data:
    # 提取数字答案
    answer_match = re.search(r'答案[:：]\s*(\d+)', item["answer"])
    if answer_match:
        number_answer = answer_match.group(1)
        # 修改答案格式：保留原有解释，但最后一行只有数字
        new_answer = item["answer"].replace(f"答案：{number_answer}", f"\n{number_answer}")
        
        formatted_item = {
            "id": item["id"],
            "question": item["question"],
            "thoughts": item["answer"],
            "answer": new_answer,
            "instruction": "请解答以下数学题，先给出详细的解题步骤，然后在最后一行只写一个数字作为最终答案，不要包含任何其他文字、标点或单位。"
        }
        formatted_data.append(formatted_item)

# 保存修改后的数据
with open("train_with_cot_prettier.json", 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)