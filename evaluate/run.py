# %%
import requests
import json
import os
from typing import List, Dict, Any
from threading import Thread


def load(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def talk(model, sys_prompt, user_prompt):
    url = "https://api.deerapi.com/v1/chat/completions"  # 这里不要用 openai base url，需要改成DEERAPI的中转 https://api.deerapi.com ，下面是已经改好的。

    payload = json.dumps(
        {
            "model": model,  # 这里是你需要访问的模型，改成上面你需要测试的模型名称就可以了。
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    )
    headers = {
        "Accept": "application/json",
        "Authorization": "sk-************************************************",  # 这里放你的 DEERapi key
        "User-Agent": "DeerAPI/1.0.0 (https://api.deerapi.com)",  # 这里也改成 DeerAPI 的中转URL https://www.deerapi.com，已经改好
        "Content-Type": "application/json",
    }

    # return "周树人和鲁迅不是兄弟，但实际上是同一个人。鲁迅是周树人的笔名。周树人是鲁迅的本名，他是中国著名的文学家和思想家。 鲁迅是他的创作名，代表了他在文学上的成就。"

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        content = response.json()["choices"][0]["message"]["content"]
        return content
    except:
        print(response.text)


def ask(model, questions):
    os.makedirs("result", exist_ok=True)
    f = open(f"result/{model}_results.json", "w", encoding="utf-8")
    log = open(f"result/{model}_log.txt", "w", encoding="utf-8")

    for q in questions:
        question_id = q.get("question_id")
        category = q.get("category")
        question_type = q.get("question_type")
        turns = q.get("turns")

        # 这里可以根据问题类型和类别选择不同的模型和提示
        model = "gpt-4o-mini"
        sys_prompt = "You are a helpful assistant."
        user_prompt = turns[0]

        response = talk(model, sys_prompt, user_prompt)
        result = {
            "question_id": question_id,
            "category": category,
            "question_type": question_type,
            "question": user_prompt,
            "response": response,
        }
        json.dump(result, f, ensure_ascii=False, indent=None)
        f.write("\n")

        print(f"问题ID: {question_id}, 回答: \n{response}")
        log.write(f"问题ID: {question_id}, 回答: \n{response}\n")
        log.flush()

    f.close()
    log.close()


# %%
questions = load("computer_science_questions.json")
# {"question_id": 101, "category": "computer_history", "question_type": "factual", "turns": ["冯·诺依曼架构的主要特点是什么？它与现代计算机架构有何关联？"]}

models = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "deepseek-chat",
    "claude-3-5-haiku-20241022",
    "gemini-2.5-flash-preview-04-17",
]

# %%
# 询问模型
if __name__ == "__main__":

    threads = [Thread(target=ask, args=(model, questions)) for model in models]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("所有模型的询问已完成。")

# %%
# 单次提问，用于偶发的无回复
q = {"question_id": 502, "category": "computer_systems", "question_type": "multiple_choice", "turns": ["以下哪种页面置换算法不会出现Belady异常？\nA. FIFO\nB. LRU\nC. Random\nD. Optimal"], "reference": ["B. LRU"]}
resp = talk("gpt-3.5-turbo", "You are a helpful assistant.", q["turns"][0])
print(resp)
# %%
