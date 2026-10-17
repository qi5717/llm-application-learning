from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import numpy as np



# 两个客户端
client_ali = OpenAI(
    api_key="你的阿里 API key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
client_ds = OpenAI(
    api_key="你的deepseek API key",
    base_url="https://api.deepseek.com"
)


# ========== 1. 读取PDF ==========
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        # 把每一页的文字提取出来，拼接到text
        # 提示：page.extract_text() 返回这一页的文字
        text += page.extract_text()
    return text

# ========== 2. 切分文本 ==========
def split_text(text, chunk_size=500,overlap=100):
    # 按每200个字切一段（简单粗暴版）
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 50: # 过滤太短的碎片
           chunks.append(chunk)
    return chunks

# ========== 3. 向量化 ==========
def get_embedding(texts, batch_size=10):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client_ali.embeddings.create(
            model="text-embedding-v4",
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings

# ========== 4. 余弦相似度 ==========
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_bm25(chunks):
    # 中文逐字切分（简单版）
    tokenized = [list(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25

def bm25_search(query, chunks, bm25, top_k=3):
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return list(top_indices)

# ========== 5. RAG主流程 ==========
def main():
    # 读取PDF
    pdf_text = read_pdf("document.pdf")
    print(f"PDF总长度：{len(pdf_text)}字")
    
    #切分
    chunks = split_text(pdf_text, chunk_size=200)
    print(f"切分成{len(chunks)}个片段")
    
    #向量化所有片段
    print("正在向量化...")
    chunk_vectors = get_embedding(chunks)
    
    # 建BM25索引
    bm25 = build_bm25(chunks)
    print("BM25索引建立完成")
    
    # 交互式问答
    print("\nPDF问答助手已启动，输入'退出'结束。\n")
    while True:
        query = input("你问：")
        if query == "退出":
            break
        
        # 向量检索
        query_vec = get_embedding([query])[0]
        vec_scores = [cosine_similarity(query_vec, vec) for vec in chunk_vectors]
        vec_indices = list(np.argsort(vec_scores)[::-1][:3])

        # BM25检索
        bm_indices = bm25_search(query, chunks, bm25, top_k=3)

        # 合并去重
        merged_indices = list(set(vec_indices + bm_indices))
        print(f"[向量检索] {vec_indices}")
        print(f"[BM25检索] {bm_indices}")
        print(f"[合并后] {merged_indices}")

        # 取合并后的片段
        contexts = [chunks[i] for i in merged_indices]
        
        # 加这一行：打印检索到的片段（前100字）
        for i, c in enumerate(contexts):
            print(f"\n[片段{i+1}] {c[:100]}...")
         
        #拼接上下文 + 构造prompt
        context_text = "\n\n".join([f"[片段{i+1}] {c}" for i, c in enumerate(contexts)])
        prompt = f"""你是一个文档问答助手。只基于以下文档片段回答问题。

文档片段：
{context_text}

问题：{query}

要求：回答时在末尾标注你参考了哪个片段，格式如：[参考：片段1, 片段3]
"""
        #调用大模型
        response = client_ds.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "你是一个文档问答助手，只基于提供的片段回答。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        print(f"\nAI:{response.choices[0].message.content}\n")
        
def rewrite_query(query):
    response = client_ds.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": "把用户的问题改写成3-5个关键词，用空格分隔，只输出关键词。"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content
        
if __name__ == "__main__":
    main()