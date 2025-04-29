
import requests
import json




url = "https://api.deerapi.com/v1/chat/completions"   # 这里不要用 openai base url，需要改成DEERAPI的中转 https://api.deerapi.com ，下面是已经改好的。

payload = json.dumps({
   "model": "gpt-3.5-turbo",  # 这里是你需要访问的模型，改成上面你需要测试的模型名称就可以了。
   "messages": [
      {
         "role": "system",
         "content": "You are a helpful assistant."
      },
      {
         "role": "user",
         "content": "周树人和鲁迅是兄弟吗？"
      }
   ]
})
headers = {
   'Accept': 'application/json',
   'Authorization': 'sk-************************************************', # 这里放你的 DEERapi key
   'User-Agent': 'DeerAPI/1.0.0 (https://api.deerapi.com)',  # 这里也改成 DeerAPI 的中转URL https://www.deerapi.com，已经改好
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)

# 将response转为json格式
response_json = json.loads(response.text)
print(response_json['choices'][0]['message']['content'])  # 这里是你需要的返回结果
