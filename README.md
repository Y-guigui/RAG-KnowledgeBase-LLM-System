# KnowledgeBase-RAG-LLM-System

 ## 基于 **Streamlit** 的本地知识库上传与**RAG**问答学习项目
适合作为本地知识库问答与 RAG 检索增强的入门实践
- 在网页端上传 `txt` 文件，自动切分后写入 Chroma 向量库
- 在网页端以聊天形式提问，基于知识库内容进行检索增强回答（RAG） 
- 支持 会话历史查看，流式思维链输出
- 技术栈：Python / Streamlit / LangChain / Chroma / Embeddings / Qwen ChatModel

---

## ✨ 功能一览

### 1) 知识库更新服务（Upload）
- Streamlit 页面上传文件，显示基本格式
- 自动读取文本内容
- 根据配置进行分段（RecursiveCharacterTextSplitter）
- 写入 Chroma 向量库（本地持久化）
- 使用 **MD5 去重**：相同内容不重复入库

### 2) 智能客服（RAG Chat）
- Streamlit Chat UI
- 显示历史消息（session_state）
- LangChain 链式调用：`Retrieval -> Prompt -> LLM -> Output`
- 支持 **流式输出**
- 支持 **消息历史文件存储**
