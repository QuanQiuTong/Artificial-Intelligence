import json
import re

# 加载增强后的数据
with open("train_with_cot.json", 'r', encoding='utf-8') as f:
    enhanced_data = json.load(f)

with open("train.json", 'r', encoding='utf-8') as f:
    original_data = json.load(f)

print(f"原始数据条目数: {len(original_data)}")
print(f"增强数据条目数: {len(enhanced_data)}")

# 检查train_with_cot.json中是否有重复的id
ids = [item["id"] for item in enhanced_data]
unique_ids = set(ids)
print(f"增强数据唯一ID数: {len(unique_ids)}")
print(f"增强数据重复ID数: {len(enhanced_data) - len(unique_ids)}")

# 检验增强后的数据对同id问题的回答是否与原始数据一致
def check_enhanced_data(enhanced_data, original_data):
    validated = []

    # 创建一个字典来存储原始数据的答案
    original_answers = {item["id"]: item["answer"] for item in original_data}
    
    errors = 0
    unfound_ids = 0
    numeric_errors = 0
    format_errors = 0
    # 检查增强后的数据
    for item in enhanced_data:
        item_id = item["id"]
        if item_id in original_answers:
            """ # 修改后的正则表达式，支持整数、小数、分数和百分数
            answer_match = re.search(r'答案[:：]\s*([-+]?\d*\.?\d+(?:/\d+)?%?)', item["answer"]) """
            # 在最后一行中提取右侧的数字答案，支持整数、小数、分数和百分数
            lines = item["answer"].strip().split("\n")
            last_line = lines[-1]
            # 找到所有匹配的数字
            all_matches = re.findall(r'[-+]?\d*\.?\d+(?:/\d+)?%?', last_line)
            if all_matches:
                # 提取最右侧的匹配（最后一个）
                number_answer = all_matches[-1]
                # 检查答案是否一致
                if number_answer != original_answers[item_id] and number_answer+"%" != original_answers[item_id]:
                    print(f"ID: {item_id} 的答案不一致！")
                    print(f"原始答案: {original_answers[item_id]}")
                    print(f"增强答案: {number_answer}")
                    numeric_errors += 1
                else:
                    # 把最后一行统一换成原始答案
                    item["answer"] = "\n".join(lines[:-1]) + f"\n{original_answers[item_id]}"
                    """请分析并解答以下问题，给出简洁计算步骤，最后一行给出数字答案"""
                    item["instruction"] = "请解答以下数学题，先给出必要的解题步骤，最后一行只写一个数字作为最终答案，不要包含任何其他文字、标点或单位。"
                    validated.append(item)
            else:
                print(f"ID: {item_id} 的增强答案格式不正确！无法提取数字答案。")
                print(f"原始答案: {original_answers[item_id]}")
                # print(f"增强答案内容: {item['answer'][-100:]}")  # 显示答案末尾部分
                format_errors += 1
        else:
            print(f"ID: {item_id} 在原始数据中未找到！")
            unfound_ids += 1
    
    # print(f"检查完成! 发现 {errors} 个错误，准确率为 {(len(enhanced_data)-errors)/len(enhanced_data)*100:.2f}%")

    print(f"检查完成! 发现 {numeric_errors} 个数值错误，{unfound_ids} 个未找到的ID，{format_errors} 个格式错误")
    print(f"准确率为 {(len(enhanced_data)-numeric_errors-unfound_ids-format_errors)/len(enhanced_data)*100:.2f}%")

    validated.sort(key=lambda x: int(x["id"]))
    json.dump(validated, open("validated.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
# 执行检查
check_enhanced_data(enhanced_data, original_data)