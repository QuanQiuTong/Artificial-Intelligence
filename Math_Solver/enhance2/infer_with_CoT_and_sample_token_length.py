# %%
import re
import json
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,     # 显式传入 attention_mask
        pad_token_id=tokenizer.eos_token_id,            # 可选：指定 pad_token_id
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
     
    return response

# 添加答案提取函数
def extract_final_answer(response):
    """从模型生成的思维链中提取最终数字答案"""
    # 按行分割取最后一行
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # 提取数字（支持分数、小数、百分数等）
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:/\d+)?%?', last_line)
        if numbers:
            return numbers[0]
    
    # 备用提取方法
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:/\d+)?%?', response)
    if numbers:
        return numbers[-1]
    return "X"  # 默认返回

test_json_new_path = "test.json"

with open(test_json_new_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

model_dir = "./Qwen/Qwen2___5-0___5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id="./output/Qwen-CoT-refined/checkpoint-3600/")

# %%
log = open("infer.log", 'w', encoding='utf-8')
with open("submit.csv", 'w', encoding='utf-8') as file:
    for idx, row in tqdm(enumerate(test_data)):
        instruction = row['instruction']
        input_value = row['question']
        id = row['id']
        
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        response = predict(messages, model, tokenizer)
        # print(f"样本ID {id} 生成长度: {len(tokenizer.encode(response))}")
        # print(response)
        log.write(f"=====\n样本ID {id} 生成长度: {len(tokenizer.encode(response))}\n{response}\n")
        log.flush()
        final_answer = extract_final_answer(response)
        file.write(f"{id},{final_answer}\n")
log.close()
# %%
""" import random
sample_ids = random.sample(range(len(test_data)), 20)

for idx in sample_ids:
    instruction = test_data[idx]['instruction']
    input_value = test_data[idx]['question']
    messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
    response = predict(messages, model, tokenizer)
    print(f"样本ID {test_data[idx]['id']} 生成长度: {len(tokenizer.encode(response))}")
    # 检查末尾是否包含答案
    print(f"末尾内容: {response[-50:]}") """
# %%
