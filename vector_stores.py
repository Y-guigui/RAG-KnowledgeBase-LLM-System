"""
向量检索
"""
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
import os
load_dotenv()


collection_name = os.getenv("COLLECTION_NAME", "rag")
persist_directory = os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")
api_key = os.getenv("API_KEY")
similarity_threshold=int(os.getenv("SIMILARITY_THRESHOLD"))
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")


class VectorStoreService(object):
    def __init__(self,embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding= embedding

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k": similarity_threshold})

if __name__ == '__main__':
    dashscope_embedding = DashScopeEmbeddings(
        model=embedding_model_name, 
        dashscope_api_key=api_key
    )
    retriever = VectorStoreService(dashscope_embedding).get_retriever()

    res = retriever.invoke("我的身高180，尺码推荐")
    for i, doc in enumerate(res, 1):
        print(f"--- 结果 {i} ---")
        print(doc.page_content.strip())
        if doc.metadata:
            print(f"来源: {doc.metadata.get('source', '未知')}")