# Please install OpenAI SDK first: `pip3 install openai`
import json

from openai import OpenAI
from pyexpat.errors import messages

client = OpenAI(api_key="sk-8fff2a66fba0431386b0ac42e3a1eb99", base_url="https://api.deepseek.com")
messages=[{"role": "user", "content": "你是一名无人机控制和强化学习领域的论文审稿专家，请给我一些修改意见"}]
response = client.chat.completions.create(model="deepseek-reasoner",messages=messages,stream=False)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
messages.append({'role': 'assistant', 'content': content})
content2="我用一种分层强化学习方法进行无人机编队控制,编队控制方法使用领导者跟随者方法,分层强化学习包括无人机控制层,和目标跟踪制导层,控制层实验与pid方法进行对比,制导层实验中,使用单层强化学习方法,分层方法,和使用pid控制层的分层方法,审稿人认为我的制导层实验对比不够好怎么修改"
try:
    data = json.loads(content2)
    print(data)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
messages.append({"role": "user", "content": content2})
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)
print(response.choices[0].message.content)