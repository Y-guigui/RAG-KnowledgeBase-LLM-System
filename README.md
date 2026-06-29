# KnowledgeBase-RAG-LLM-System

基于 LangChain + Chroma + 阿里百炼的本地知识库 RAG 问答系统。支持文档上传、向量检索、多轮对话记忆和流式输出。

---

## 功能特性

- **文档入库**：支持 TXT 文件上传，自动切分、向量化并存入 Chroma 向量数据库
- **MD5 去重**：相同内容的文档自动跳过，避免重复入库
- **RAG 检索增强**：用户提问时自动检索相关知识片段，注入 LLM 上下文后生成回答
- **多轮对话记忆**：基于文件的会话历史管理，支持多用户同时使用
- **流式输出**：打字机效果的流式响应，提升交互体验
- **Web 界面**：基于 Streamlit 的聊天和上传双页面

---

## 项目结构

```
KnowledgeBase-RAG-LLM-System/
├── app_chat.py              # Streamlit 聊天界面
├── app_upload.py            # Streamlit 文件上传界面
├── knowledge_base.py        # 知识库入库服务（切分 / 向量化 / MD5 去重）
├── vector_stores.py         # Chroma 向量数据库封装
├── rag.py                   # RAG 核心链（检索 → 拼上下文 → LLM 回答）
├── file_history_store.py    # 文件级会话历史存储
├── .env                     # 环境变量配置（需手动创建）
├── chat_history/            # 会话历史文件（自动生成）
└── data/chroma_db/          # Chroma 向量库持久化目录（自动生成）
```

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| LLM | 阿里百炼（DashScope，兼容 OpenAI API） |
| Embedding | DashScope text-embedding-v4 |
| 向量数据库 | Chroma（本地持久化） |
| 框架 | LangChain（LCEL）+ Streamlit |
| 文本切分 | RecursiveCharacterTextSplitter |
| 会话记忆 | 自定义 FileChatMessageHistory |

---

## 架构流程

```
┌──────────────┐     ┌──────────────┐
│  app_upload  │────▶│knowledge_base│
│  上传 TXT    │     │ 切分+向量化   │
└──────────────┘     └──────┬───────┘
                            │ 存入
                     ┌──────▼───────┐
                     │    Chroma    │
                     │  向量数据库   │
                     └──────┬───────┘
                            │ 检索
┌──────────────┐     ┌──────▼───────┐     ┌──────────────┐
│  app_chat    │────▶│   rag.py     │────▶│    LLM       │
│  用户提问    │     │  拼装上下文   │     │  流式输出    │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │  file_history│
                     │   会话记忆    │
                     └──────────────┘
```

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- 阿里百炼 API Key（[免费申请](https://bailian.console.aliyun.com/)）

### 2. 安装依赖

```bash
pip install streamlit langchain langchain-chroma langchain-openai langchain-community langchain-text-splitters dashscope python-dotenv
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 阿里百炼 API
API_KEY=sk-xxxxxxxxxxxxx
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型配置
CHAT_MODEL_NAME=qwen3.7-max
EMBEDDING_MODEL_NAME=text-embedding-v4

# Chroma 向量库
COLLECTION_NAME=rag
PERSIST_DIRECTORY=./data/chroma_db

# 文本切分配置
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_SPLITER_CHAR_NUMBER=500
SIMILARITY_THRESHOLD=4

# MD5 去重记录文件
MD5_PATH=./data/md5_database.txt
```

### 4. 启动服务

**上传知识库**：

```bash
streamlit run app_upload.py
```

**开始对话**：

```bash
streamlit run app_chat.py
```

### 5. 使用流程

1. 打开上传页面，上传 TXT 文档（如产品说明、FAQ）
2. 等待向量化完成，看到 `[Success]` 提示
3. 打开聊天页面，输入问题测试检索效果

---

## 关键模块说明

### knowledge_base.py — 知识入库

```python
service = KnowledgeBaseService()
result = service.upload_by_str("文档内容...", "filename.txt")
```

- 对 500 字符以上的长文本自动切分为 chunks
- 每个 chunk 附带元数据（来源文件、入库时间）
- 通过 MD5 哈希防止重复入库

### rag.py — RAG 核心链

```python
rag = RagService()
result = rag.chain.invoke(
    {"input": "用户问题"},
    {"configurable": {"session_id": "user_001"}}
)
```

- LCEL 管道：`用户输入 → 检索 → 拼上下文 → Prompt → LLM → 流式输出`
- 通过 `RunnableWithMessageHistory` 注入多轮对话历史
- 支持 `chain.stream()` 实现打字机效果

### file_history_store.py — 会话记忆

```python
history = FileChatMessageHistory("session_abc", "./chat_history")
history.add_messages([HumanMessage("你好")])
```

- 每个会话对应一个 JSON 文件
- 兼容 `BaseChatMessageHistory` 接口，可无缝接入 LangChain 的记忆体系
- 自动处理文件不存在和 JSON 损坏的异常

---

## 后续可扩展方向

| 方向 | 说明 |
|------|------|
| Agent 升级 | 加入 Tool Use + LangGraph 编排，让 LLM 自主决策是否检索 |
| 多格式支持 | 扩展 PDF、Word、Markdown 等文档格式 |
| 混合检索 | BM25 关键词检索 + 向量语义检索，取并集后 Rerank |
| 检索评估 | 引入 RAGAS 评估 faithfulness / answer relevancy |
| Docker 部署 | 容器化 + FastAPI 替换 Streamlit，支持生产部署 |
| 多模态 | 支持表格、图片的检索与理解 |

---