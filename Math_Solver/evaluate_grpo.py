import json
import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
from tqdm import tqdm

def evaluate_model(model, tokenizer, eval_data):
    """评估模型在数据集上的表现"""
    correct = 0
    total = 0
    
    for item in tqdm(eval_data, desc="Evaluating"):
        question = item["question"]
        correct_answer = item["answer"]
        
        # 构建提示
        prompt = f"<|im_start|>system\n你是一个数学解题专家，请解答下面的问题，只需要给出最终答案。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 生成答案
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        predicted_answer = generated_text.strip()
        
        # 检查答案是否正确
        if predicted_answer == correct_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    # 加载验证数据（这里使用部分训练数据作为验证集）
    with open("train.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # 取前200条作为验证集
    eval_data = all_data[:200]
    
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
    
    # 评估LoRA微调后模型
    lora_model = PeftModel.from_pretrained(
        base_model,
        "./output/Qwen",
        device_map="auto"
    )
    
    lora_accuracy = evaluate_model(lora_model, tokenizer, eval_data)
    print(f"LoRA微调模型准确率: {lora_accuracy:.4f}")
    
    # 评估GRPO优化后模型
    grpo_model = PeftModel.from_pretrained(
        base_model,
        "./output/grpo_model",
        device_map="auto"
    )
    
    grpo_accuracy = evaluate_model(grpo_model, tokenizer, eval_data)
    print(f"GRPO优化模型准确率: {grpo_accuracy:.4f}")
    
    # 输出性能提升
    improvement = (grpo_accuracy - lora_accuracy) / lora_accuracy * 100 if lora_accuracy > 0 else 0
    print(f"GRPO优化提升: {improvement:.2f}%")

if __name__ == "__main__":
    main()