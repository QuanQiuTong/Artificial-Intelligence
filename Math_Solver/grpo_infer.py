import json
import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
from tqdm import tqdm

def extract_answer(text):
    """
    从生成的文本中提取数值答案
    """
    # 去除所有空白字符和特殊符号，仅保留数字
    text = text.strip()
    return text

def main():
    # 加载测试数据
    with open("test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # 获取正确的基础模型路径
    model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="./", revision="master")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    # 加载GRPO优化后的模型
    model = PeftModel.from_pretrained(
        base_model,
        "./output/grpo_model",  # GRPO优化后的模型路径
        device_map="auto"
    )
    
    results = []
    
    for item in tqdm(test_data, desc="Generating answers"):
        question_id = item["id"]
        question = item["question"]
        
        # 构建提示
        prompt = f"<|im_start|>system\n你是一个数学解题专家，请解答下面的问题，只需要给出最终答案。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 生成答案
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            temperature=0.2,  # 低温度使生成更确定性
            do_sample=False,  # 不使用采样，获得最可能的答案
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        answer = extract_answer(generated_text)
        
        results.append({"id": question_id, "ret": answer})
    
    # 保存为CSV文件
    df = pd.DataFrame(results)
    df.to_csv("submit.csv", index=False)
    
    print(f"结果已保存至 submit.csv，共处理 {len(results)} 条数据")

if __name__ == "__main__":
    main()