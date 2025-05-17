import json
import torch
import random
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

# 设置随机种子以确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 从测试集中生成多个答案
def generate_multiple_answers(model, tokenizer, question, num_samples=5, temperature=0.8):
    prompt = f"<|im_start|>system\n你是一个数学解题专家，请解答下面的问题，只需要给出最终答案。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    # 明确请求返回attention_mask
    model_inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = model_inputs.input_ids.to(model.device)
    attention_mask = model_inputs.attention_mask.to(model.device)  # 获取注意力掩码
    
    answers = []
    for _ in range(num_samples):
        # 使用不同的随机种子生成多样化的回答
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # 添加注意力掩码
            max_new_tokens=256,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        # 提取数值答案（假设答案是一个数字）
        try:
            # 这里简化处理，实际应用中需要更复杂的答案提取逻辑
            answer = generated_text.strip()
            answers.append(answer)
        except:
            answers.append("")
            
    return answers

# 计算答案相似度和奖励
def calculate_rewards(generated_answers, correct_answer):
    rewards = []
    
    for answer in generated_answers:
        # 提取数值（简化处理）
        try:
            # 对于数学题，我们可以直接比较数值是否相等
            if answer == correct_answer:
                reward = 1.0  # 完全正确
            else:
                reward = 0.0  # 完全错误
        except:
            reward = 0.0  # 无法解析的答案
            
        rewards.append(reward)
        
    return rewards

# PPO训练循环
def ppo_train(model, tokenizer, train_data, num_epochs=3, lr=5e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        # 确保每个epoch开始时模型都处于训练模式
        model.train()
        
        total_reward = 0
        total_samples = 0
        
        for data_item in tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}"):
            question = data_item["question"]
            correct_answer = data_item["answer"]
            
            # 为每个问题生成多个答案
            generated_answers = generate_multiple_answers(model, tokenizer, question)
            
            # 计算每个答案的奖励
            rewards = calculate_rewards(generated_answers, correct_answer)
            
            # 记录奖励统计
            total_reward += sum(rewards)
            total_samples += len(rewards)
            
            # 对于每个生成的答案，根据奖励优化模型
            for answer, reward in zip(generated_answers, rewards):
                if reward > 0:  # 只对正确答案进行优化
                    # 准备输入和标签
                    full_text = f"<|im_start|>system\n你是一个数学解题专家，请解答下面的问题，只需要给出最终答案。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
                    
                    # 编码全文
                    encodings = tokenizer(full_text, return_tensors="pt").to(model.device)
                    input_ids = encodings.input_ids
                    attention_mask = encodings.attention_mask  # 获取注意力掩码
                    
                    # 创建标签，将非答案部分的标签设为-100
                    # 找到助手标记开始的位置
                    assistant_token = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                    assistant_pos = None
                    
                    # 在input_ids中查找助手标记的位置
                    for i in range(len(input_ids[0]) - len(assistant_token)):
                        if torch.all(input_ids[0, i:i+len(assistant_token)] == torch.tensor(assistant_token, device=model.device)):
                            assistant_pos = i + len(assistant_token)
                            break
                    
                    # 创建标签，将非答案部分标记为-100
                    labels = input_ids.clone()
                    if assistant_pos is not None:
                        labels[0, :assistant_pos] = -100  # 将系统和用户部分的标签设为-100
                    
                    # 前向传播
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss * reward  # 根据奖励调整损失
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # 打印每个epoch的平均奖励
        avg_reward = total_reward / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch+1}: Average Reward = {avg_reward:.4f}")
    
    # 保存优化后的模型
    model.save_pretrained("./output/grpo_model-with_attention_mask")
    
def main():
    set_seed(42)
    
    # 加载训练数据
    with open("train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
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
    
    # 特别添加这一行，启用输入需要梯度
    base_model.enable_input_require_grads()
    
    # 加载训练后的LoRA权重
    model = PeftModel.from_pretrained(
        base_model,
        "./output/Qwen/checkpoint-3750",
        device_map="auto",
        # 重要：设置为训练模式，而不是推断模式
        inference_mode=False
    )
    
    # 明确设置模型为训练模式
    model.train()
    
    # 确保所有参数可训练
    for param in model.parameters():
        param.requires_grad = True
    
    # 进行GRPO训练
    ppo_train(model, tokenizer, train_data)
    
    print("GRPO训练完成!")

if __name__ == "__main__":
    main()