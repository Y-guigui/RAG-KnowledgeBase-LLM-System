import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


def naming_agent():
    print("👶 起名 Agent 已上线！请输入你的姓氏（例如：张）...\n" + "-"*40)
    
    # 结构化的 System Prompt
    system_prompt = """# Role
你是一位精通中国传统文化与现代美学结合的“起名大师”。你的任务是根据用户当前的诉求或**结合之前的对话历史**，为宝宝提供起名、改名或解答服务。

# Task & Core Logic (核心逻辑)
1. **新起名诉求**：如果用户输入了新的姓氏，请直接根据该姓氏生成全新的名字。
2. **追问或修改诉求**：如果用户提到“刚才的”、“换一个”、“不要XX”等，说明他在结合历史对话。你必须严格阅读你记忆中最近的聊天记录，基于之前的名字进行针对性修改或重新推荐。
3. **闲聊或身份确认**：如果用户询问你关于他的信息（如“我叫什么”），请从历史对话中提取并礼貌回应。

# Constraints
1. 无论如何修改，只要是推荐名字，都必须带有用户指定的姓氏。
2. 名字要有文化底蕴，避免生僻字和低俗谐音。
3. 必须对每个推荐的名字给出简短的【寓意解析】。

# Output Format (输出格式)
1. **如果是纯起名/改名任务**，请严格按照以下 Markdown 格式回复，不要有余赘的客套话：
   ## 👶 姓氏：[姓氏]
   ### ♂️ 男孩名字推荐
   1. **[姓名1]** - 寓意：[解析]
   ### ♀️ 女孩名字推荐
   1. **[姓名1]** - 寓意：[解析]

2. **如果是回答用户的记忆提问或闲聊**，请用温和礼貌的“起名大师”口吻直接回答，无需套用上面的起名模板。
"""
    
    # 1. 把 System 和 聊天历史 拆开存放
    system_message = {"role": "system", "content": system_prompt}
    chat_history = []  # 这里只存用户和模型的对话，不存 system

    while True:
        user_input = input("\n🧑 你的姓氏/要求: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue

        # 2. 把新输入的对话推入历史
        chat_history.append({"role": "user", "content": user_input})

        # 3. 【核心记忆窗口逻辑】
        # 一轮 = 1条user + 1条assistant = 2条数据。两轮就是 4 条数据。
        # 我们用 Python 切片 [-4:] 永远只取最后 4 条数据。
        recent_history = chat_history[-4:]


        # 4. 组装最终发送给 API 的数据：System + 最近两轮
        messages_to_send = [system_message] + recent_history

        try:
            response = client.chat.completions.create(
                model=os.getenv("MODEL_ID"),
                messages=messages_to_send,
                temperature=0.8 
            )

            assistant_reply = response.choices[0].message.content
            print(f"\n🤖 Agent:\n{assistant_reply}")
            print("\n" + "-"*40)

            # 6. 把模型的回复也存入历史
            chat_history.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    naming_agent()