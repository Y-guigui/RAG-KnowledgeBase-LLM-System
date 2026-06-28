from operator import itemgetter
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from file_history_store import get_history

from dotenv import load_dotenv
load_dotenv()

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
chat_model_name = os.getenv("CHAT_MODEL_NAME", "qwen3.7-max")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagService(object):

    def __init__(self):
        # 向量服务
        self.vector_service = VectorStoreService(embedding=DashScopeEmbeddings(
                model=embedding_model_name, 
                dashscope_api_key=api_key
            )
        )
        # 提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("human", "请回答用户提问：{input}")
            ]
        )
        # 聊天模型
        self.chat_model = ChatOpenAI(
            model=chat_model_name,
            api_key=api_key,       # 传入你的阿里百炼 API Key
            base_url=base_url      # 关键：通过修改 base_url，把请求从 OpenAI 官方服务器路由到阿里云
        )

        # 执行链
        self.chain = self.__get_chain()


    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]) -> str:
            if not docs:
                return "无参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段: {doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        chain = (
            RunnablePassthrough.assign(
                context=itemgetter("input") | retriever | format_document)
            )| self.prompt_template | print_prompt | self.chat_model | StrOutputParser()

        # 检索器 | 提示词 | model | 字符串输出解析器
        # retriver需要的输入是字符串，而拼接的conversation_chain是字典，dict -> str
       

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

if __name__ == '__main__':
    # session id 配置
    session_config ={
        "configurable":{
            "session_id":"user_001",
        }
    }
    result = RagService().chain.invoke({"input":"我身高是175，推荐穿什么尺码的衣服。"},session_config)
    print(result)
    # res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
    # print(res)