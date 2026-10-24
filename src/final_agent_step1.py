from openai import OpenAI
from ddgs import DDGS
import streamlit as st
import networkx as nx
import json
import os


HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# 构建一个微型的企业组织架构图
G = nx.DiGraph()
# 添加节点和关系（A 是 B 的主管）
G.add_edge("Charlie", "Bob", relation="汇报给")
G.add_edge("Bob", "Alice", relation="汇报给")
G.add_edge("Alice", "David", relation="汇报给")
G.add_edge("David", "CEO", relation="汇报给")


# ========== 1. 初始化 ==========
client = OpenAI(
    api_key = st.secrets["DS_KEY"], 
    base_url="https://api.deepseek.com"
)

# 模拟一个内存级的技能注册表    
SKILLS = {
    "hr_rag": {
        "description": "查询公司的规章制度、请假流程、薪酬福利、报销等内部文档。",
        "trigger_keywords": ["请假", "报销", "薪酬", "规章制度"]
    },
    "graph_query": {
        "description": "查询公司内部的人员组织架构、汇报线、部门归属。",
        "trigger_keywords": ["老板", "上级", "部门", "汇报"]
    },
    "web_search": {
        "description": "当用户询问实时新闻、今天天气、外部公开信息时，使用此技能联网搜索。",
        "trigger_keywords": ["新闻", "天气", "最新", "搜索"]
    }
}

# 模拟企业内部文档
HR_DOCUMENTS = """
员工考勤与请假制度：
1. 员工需每天打卡。迟到超过15分钟扣半天工资。
2. 病假需提供三甲医院证明。
3. 年假：入职满1年有5天，满3年有10天。
4. 报销流程：每月25日统一提交发票，需部门主管和财务双审批。
"""


# ========== 2. 核心路由函数（渐进式披露的体现） ==========
def route_intent(user_query):
    """轻量级调度器：只判断意图，不加载具体长指令"""
    skills_desc = "\n".join([f" - {k}: {v['description']}" for k, v in SKILLS.items()])
    
    prompt = f"""
    你是一个智能调度中心。请根据用户的输入，判断应该调用哪个技能。
    可用技能列表：
    {skills_desc}
    
    用户输入：{user_query}
    
    要求：
    1. 如果涉及HR制度，输出 JSON: {{"skill": "hr_rag"}}
    2. 如果涉及人员架构，输出 JSON: {{"skill": "graph_query"}}
    3. 如果涉及实时信息、新闻、天气等外部数据，输出 JSON: {{"skill": "web_search"}}
    4. 如果都不涉及，输出 JSON: {{"skill": "chat"}}
    """
    
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("skill","chat")
    
# ========== 3. 技能执行器 ==========
def execute_skill(skill_name, user_query):
    print(f"--> 命中技能: {skill_name}，加载对应执行器...")
    
    if skill_name == "hr_rag":
        # 真实业务这里需要走 Embedding 检索，这里简化为直接把全量文档塞进去（生产不可取，但先跑通逻辑）
        system_prompt = f"你是一个专业的HR助手。请仅根据以下公司制度回答员工问题。\n文档内容：\n{HR_DOCUMENTS}"
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
        )
        
        return  response.choices[0].message.content
    
    elif skill_name == "graph_query":
        # 让大模型把自然语言转成起点和终点，方便图谱查询
        extract_prompt = f"""
        用户问：{user_query}
        请提取出需要查询关系的人物A（起点）和人物B（终点）。
        只输出 JSON，格式为：{{"start": "人名", "end": "人名"}}
        如果只问了一个人，end 留空。
        """
        resp = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": extract_prompt}],
            response_format={"type": "json_object"}
        )
        info = json.loads(resp.choices[0].message.content)
        start = info.get("start")
        end = info.get("end")
        
        # 图谱查询逻辑
        try:
            if start and not end:
                # 查某人的直接上级
                paths = list(G.successors(start))
                if paths:
                    return f"【图谱查询】{start} 的直接上级是：{paths[0]}。"
                else:
                    return f"【图谱查询】未找到 {start} 的上级。"
                
            elif start and end:
                # 查两人之间的路径
                path = nx.shortest_path(G, source=start, target=end)
                return f"【图谱查询】{start} 到 {end} 的汇报路径为：{' -> '.join(path)}"
        except nx.NetworkXNoPath:
            return f"【图谱查询】在系统中未找到 {start} 和 {end} 之间的关联。"
        except nx.NodeNotFound:
            return f"【图谱查询】图谱中不存在这个人。"
        return "图谱查询失败，请检查人物名称。"
    
    elif skill_name == "web_search":
        try:
            # 1. 让大模型把用户的长句，提炼成搜索引擎最喜欢的关键词
            extract_prompt = f"""
            用户的问题是：{user_query}
            请提取出一个最核心的、适合直接在搜索引擎搜索的关键词短语。只输出关键词本身，不要其他任何废话。
            """
            keyword_resp = client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role": "user", "content": extract_prompt}]
            )
            search_keyword = keyword_resp.choices[0].message.content.strip()
            print(f"--> 提炼后的搜索词: {search_keyword}") # 方便调试")
            
            # 2. 用提炼后的关键词去搜索
            results = DDGS().text(search_keyword, max_results=3, backend="duckduckgo")
            
            # 兜底：如果搜不到，直接返回
            if not results:
                return f"【联网搜索】尝试搜索了 '{search_keyword}'，但网络未返回任何结果。"
            
            search_text = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            
            # 3. 把搜索结果交给大模型总结
            system_prompt = f"你是一个实时信息助手。请根据以下联网搜索结果回答用户。\n搜索结果：\n{search_text}"
            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
            )
            return f"【联网搜索】\n{response.choices[0].message.content}"
        except Exception as e:
            return f"搜索网络异常: {e}"
            
    else:
        # 纯闲聊
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": user_query}]
        )
        return response.choices[0].message.content

# ========== 4. Streamlit 前端 ==========
# 侧边栏展示当前系统的架构
with st.sidebar:
    st.header("🧠 智能体系统状态")
    st.markdown("✅ **渐进式路由调度器**: 已激活")
    st.markdown("✅ **内部HR知识库(RAG)**: 已接入")
    st.markdown("✅ **组织架构图谱(Graph)**: 已接入")
    st.markdown("✅ **联网实时搜索(Web)**: 已接入")
    st.markdown("---")
    st.caption("Full-Stack Agent v1.0")
    
st.title("🕵️ 企业级全栈特工 Agent")
st.write("集成 HR 知识库、组织架构图谱与联网搜索的渐进式智能体。")

if "messages" not in st.session_state:
    st.session_state.messages = load_history()
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
if prompt := st.chat_input("请问有什么可以帮您？"):
    # 1. 显示用户提问
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content": prompt})
    save_history(st.session_state.messages)
    
    with st.spinner("AI 正在思考调度中..."):
        # 2. 路由判定
        skill = route_intent(prompt)
        st.caption(f"*(底层日志：成功挂载技能 `{skill}`)*")
    
        # 3. 执行技能
        answer = execute_skill(skill, prompt)
        
    # 4. 显示回答
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
