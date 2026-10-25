import streamlit as st
import requests


st.set_page_config(page_title="全栈特工(前后端分离版)", page_icon="🕵️")
st.title("🕵️ 全栈特工 (前后端分离版)")
st.write("当前前端界面不包含任何 AI 逻辑，所有计算均由 FastAPI 后端完成。")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "skill_used" in msg:
            st.caption(f"*(底层日志：由后端挂载技能 `{msg['skill_used']}`)*")
        st.markdown(msg["content"])
        
if prompt := st.chat_input("请输入您的指令..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("前端正在向本地后端 8000 端口发送 HTTP 请求..."):
        try:
            # 🌟 核心变化：通过 HTTP POST 请求调用后端 API
            response = requests.post(
                "http://127.0.0.1:8000/agent_chat",
                json= {"query":prompt},
                timeout=30
            )
            data = response.json()
            
            reply_text = data.get("reply", "解析失败")
            skill_used  = data.get("skill_used", "unknown")
            
        except Exception as e:
            reply_text = f"无法连接到后端，请确认 uvicorn 是否启动。错误信息：{e}"
            skill_used = "connection_failed"
            
    with st.chat_message("assistant"):
        st.caption(f"*(底层日志：由后端挂载技能 `{skill_used}`)*")
        st.markdown(reply_text)
        
        
    st.session_state.messages.append({
        "role":"assistant",
        "content": reply_text,
        "skill_used": skill_used
    })