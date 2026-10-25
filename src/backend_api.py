from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from functools import lru_cache
from ddgs import DDGS
import json, networkx as nx



# --- 引入 LangGraph ---
from typing import TypedDict
from langgraph.graph import StateGraph, END

app = FastAPI(title="全能特工后端 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = ChatOpenAI(model="deepseek-flash", api_key="API-Key", base_url="https://api.deepseek.com", temperature=0)

# ================= 初始化图谱与知识库 =================
G = nx.DiGraph()
G.add_edge("Charlie", "Bob"); G.add_edge("Bob", "Alice"); G.add_edge("Alice", "David")
HR_DOC = "员工考勤与请假制度：1. 病假需提供三甲医院证明。2. 年假：满1年有5天，满3年有10天。"

SKILLS = {
    "hr_rag": {"desc": "查询请假、报销等制度。"},
    "graph_query": {"desc": "查询人员汇报线。"},
    "web_search": {"desc": "查询实时新闻天气。"},
    "copywriting": {"desc": "需要撰写海报文案、营销口号、标语时使用。"}
}

class ChatRequest(BaseModel):
    query: str

# ================= 1. 核心路由 =================
def route_intent(query):
    prompt = f"判断意图。可用：{json.dumps(SKILLS)}。涉及写文案/标语->copywriting；制度->hr_rag；人员->graph_query；实时->web_search；其他->chat。只输出 JSON: {{\"skill\": \"技能名\"}}\n用户：{query}"
    resp = llm.invoke(prompt)
    return json.loads(resp.content.replace("```json", "").replace("```", "").strip()).get("skill", "chat")

# ================= 2. LangGraph 多智体系统 =================
class AgentState(TypedDict):
    task: str
    draft: str
    review: str
    iterations: int

def writer_node(state: AgentState):
    prompt = f"请写一段关于 '{state['task']}' 的文案（20-30字）。"
    if state.get("review"): prompt += f"\n注意，之前被打回了，意见：{state['review']}\n请重写。"
    draft = llm.invoke(prompt).content
    return {"draft": draft, "iterations": state.get("iterations", 0) + 1}

def reviewer_node(state: AgentState):
    review = llm.invoke(f"严格审核以下文案。标准：1.有排比 2.有诗意 3.字数20-30。通过开头加【审核通过】，否则加【需要修改】。\n文案：{state['draft']}").content
    return {"review": review}

def route_review(state: AgentState):
    if "【审核通过】" in state["review"] or state["iterations"] >= 3:
        return END
    return "writer"

workflow = StateGraph(AgentState)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.set_entry_point("writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges("reviewer", route_review, {"writer": "writer", END: END})
agent_app = workflow.compile()


# 写一个独立的带缓存的函数
@lru_cache(maxsize=100)
def run_multi_agent_with_cache(user_query: str):
    """如果用户问了重复的问题，直接返回上次的结果，不再消耗 Token"""
    print(f"--- [缓存未命中] 正在为 '{user_query}' 启动多智体引擎 ---")
    result = agent_app.invoke({"task": user_query, "iterations": 0})
    return result




# ================= 3. API 接口 =================
@app.post("/agent_chat")
def agent_chat_endpoint(request: ChatRequest):
    user_query = request.query
    skill = route_intent(user_query)
    
    try:
        if skill == "copywriting":
            # 这里调用带缓存的函数
            result = run_multi_agent_with_cache(user_query)
            reply = f"【多智体文案生成】\n最终文案：{result['draft']}\n(经历了 {result['iterations']} 次修改)\n审核意见：{result['review']}"
        elif skill == "hr_rag":
            reply = llm.invoke(f"作为HR助手，根据制度回答：\n{HR_DOC}\n问：{user_query}").content
        elif skill == "graph_query":
            extract = llm.invoke(f"提取人名，只输出名字：{user_query}").content.strip()
            if extract in G:
                paths = list(G.successors(extract))
                reply = f"【图谱查询】{extract} 的上级是：{paths[0] if paths else '无'}。"
            else:
                reply = f"【图谱查询】未找到 '{extract}'。"
        elif skill == "web_search":
            results = DDGS().text(user_query, max_results=2, backend="duckduckgo")
            search_text = "\n".join([f"- {r['body']}" for r in results]) if results else "无搜索结果"
            reply = llm.invoke(f"根据搜索结果回答：\n{search_text}\n问：{user_query}").content
        else:
            reply = llm.invoke(user_query).content
    except Exception as e:
        reply = f"后端异常：{str(e)}"
        skill = "error"

    return {"status": "success", "skill_used": skill, "reply": reply}