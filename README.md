# 🤖 大模型应用开发学习与实践 (LLM Application Learning)

这个仓库记录了我从零开始学习大模型应用开发的过程，包含我独立完成的三个实战项目。

## 📂 项目目录

- `src/week3_chatbot.py`: 命令行 AI 聊天机器人（支持多轮对话与流式输出）
- `src/week4_pdf_rag.py`: 命令行版 PDF 文档问答助手（RAG）
- `src/week5_rag_app.py`: **带 Web 界面的 PDF 文档问答助手（Streamlit）**

## 🚀 核心项目展示：PDF文档问答助手（RAG Web版）

这是一个基于 RAG（检索增强生成）架构的 Web 应用。用户可以上传 PDF 文档，并用自然语言提问，系统会基于文档内容精准回答，并标注引用来源。

### ✨ 功能亮点
- **Web UI 界面**：使用 Streamlit 搭建，支持拖拽上传 PDF。
- **混合检索策略**：向量检索（阿里云 Embedding） + 关键词检索（BM25），大幅提高检索召回率。
- **引用溯源**：大模型回答时标明参考的文档片段。
- **状态缓存**：使用 `session_state` 缓存处理结果，避免重复消耗 API 费用。

### 📸 效果截图
*(这里粘贴你刚才上传图片生成的 Markdown 代码)*
![RAG应用截图](./images/rag_app_01.png)
![RAG应用截图](./images/rag_app_02.png)
### 🛠️ 如何本地运行
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行 RAG Web 应用
streamlit run src/week5_rag_app.py
