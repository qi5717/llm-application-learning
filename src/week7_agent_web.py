import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from ddgs import DDGS
import datetime

# ================= 页面设置 =================
st.set_page_config(page_title="全能实时 AI 助手", page_icon="🤖")
st.title("🤖 全能实时 AI 助手")

# ================= 侧边栏：清空记忆 =================
with st.sidebar:
    st.header("设置")
    if st.button("🗑️ 清空对话记忆"):
        st.session_state.messages = []
        st.rerun() # 刷新页面

# ================= 初始化 API =================
api_key = st.secrets["DS_KEY"]
llm = ChatOpenAI(model="deepseek-flash", api_key=api_key, base_url="https://api.deepseek.com")

# ================= 定义工具 =================
@tool
def web_search(query: str) -> str:
    """当用户询问当前新闻、实时天气、最新事件时，使用此工具。"""
    try:
        results = DDGS().text(query, max_results=3, backend="duckduckgo")
        if not results:
            return "搜索未找到相关结果。"
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"搜索出现网络异常: {e}"

@tool
def get_current_time(query: str) -> str:
    """获取当前系统时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [web_search, get_current_time]

# ================= 创建 Agent =================
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一个严谨的全能AI助手。"
        "如果用户问题涉及实时天气、新闻等，你必须使用联网搜索工具；"
        "如果搜索工具报错，请如实告诉用户网络异常，绝对不要编造任何虚假信息。"
        "如果问题不需要工具（如常识、数学、写代码），请直接回答。"
    )
)

# ================= 网页交互与记忆 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 显示用户问题
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. 调用 Agent
    with st.spinner("AI 正在思考并调用工具..."):
        langchain_messages = [(m["role"], m["content"]) for m in st.session_state.messages]
        inputs = {"messages": langchain_messages}
        
        try:
            # 直接同步等待最终结果
            final_response = agent.invoke(inputs)
            
            # 获取最后一条 AI 消息
            last_message = final_response["messages"][-1]
            final_answer = last_message.content if last_message.content else "（AI 思考超时，未返回内容）"
            
        except Exception as e:
            final_answer = f"发生错误：{e}"
        
    # 3. 显示 AI 回答并存入记忆
    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
