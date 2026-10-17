import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import numpy as np


# ========== 1. 页面配置 ==========
st.set_page_config(page_title="PDF问答助手", page_icon="📄", layout="wide")
st.title("📄 PDF文档问答助手")
st.markdown("上传PDF文件，基于文档内容进行智能回答。")


# ========== 2. 初始化客户端 ==========
client_ali = OpenAI(
    api_key="你的API key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

client_ds = OpenAI(
    api_key="你的API key",
    base_url="https://api.deepseek.com"
)

# ========== 3. 工具函数 ==========
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def get_embedding(texts, batch_size=10):
    all_embeddings = []
    for i in range(0, len(texts),batch_size):
        batch = texts[i:i+batch_size]
        response = client_ali.embeddings.create(
            model="text-embedding-v4",
            input=batch
        )   
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ========== 4. 文件上传与处理 ==========
uploaded_file = st.file_uploader("上传PDF文件", type=["pdf"])

# 核心机制：用 session_state 缓存处理好的数据，避免每次提问都重新处理PDF
if uploaded_file and "chunks" not in st.session_state:
    with st.spinner("正在读取和向量化PDF，请稍候..."):
        text = read_pdf(uploaded_file)
        chunks = split_text(text)
        chunk_vectors = get_embedding(chunks)
        bm25 = BM25Okapi([list(c) for c in chunks])
        
        # 将数据存入 session_state
        st.session_state.chunks = chunks
        st.session_state.chunk_vectors = chunk_vectors
        st.session_state.bm25 = bm25
        
    st.success(f"✅ PDF处理完成！共提取 {len(chunks)} 个片段。")
    
# ========== 5. 问答区 ==========
if  uploaded_file and "chunks" in st.session_state:
    st.divider()
    query = st.text_input("请输入你的问题：")   
    
    if query:
        chunks = st.session_state.chunks
        chunk_vectors = st.session_state.chunk_vectors
        bm25 = st.session_state.bm25
        
        with st.spinner("正在检索并生成回答..."):
            # 1. 向量检索
            query_vec = get_embedding([query])[0]
            vec_scores = [cosine_similarity(query_vec, vec) for vec in chunk_vectors]
            vec_indices = list(np.argsort(vec_scores)[::-1][:3])
            
            # 2. BM25检索
            bm_scores = bm25.get_scores(list(query))
            bm_indices = list(np.argsort(bm_scores)[::-1][:3])
             
            # 3. 合并去重
            merged = list(set(vec_indices + bm_indices))
            contexts = [chunks[i] for i in merged]
            
            # 4. 拼接上下文与提示词
            contexts_text = "\n\n".join([f"[片段{i+1}]{c}" for i,c in enumerate(contexts)])
            prompt =  f"""你是一个文档问答助手。只基于以下文档片段回答问题。
文档片段：
{contexts_text}

问题：{query}

要求：回答时在末尾标注你参考了哪个片段，格式如：[参考：片段1, 片段3]"""  
      
            # 5. 调用大模型
            response = client_ds.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": "你是一个文档问答助手，只基于提供的片段回答。"},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            
        # 6. 展示结果
        st.markdown("### 🤖 AI回答")
        st.info(answer) # 使用 info 框展示回答，更美观
        
        # 7. 折叠面板展示检索片段
        with st.expander("🔍 查看检索到的参考片段"):
            for i, c in enumerate(contexts):
                st.text_area(f"片段 {i+1}", c, height=150)