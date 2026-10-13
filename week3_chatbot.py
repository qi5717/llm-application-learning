from openai import OpenAI

client = OpenAI(
    api_key="你的API Key",
    base_url= "https://api.deepseek.com"
)

#初始化对话历史
messages = [
    {"role": "system","content":"你是一个友好的AI助手，回答简洁明了。"}
]

print("聊天机器人已启动，输入 '退出'结束对话.\n")

while True:
    user_input = input("你：")
    
    if user_input == "退出":
        break
    
    messages.append({"role": "user", "content": user_input})
    
    stream = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=True
    )
    
    print("AI: ", end="")
    full_reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="",flush=True)
            full_reply += content
    print("\n")
    
    messages.append({"role": "assistant", "content": full_reply})