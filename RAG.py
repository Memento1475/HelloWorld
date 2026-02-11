import os
import time
import ollama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# 配置区域
PDF_FILE_NAME = "数据结构.pdf"  # 换材料只需更改这一行
DB_PATH = "./chroma_db_storage"
CHAT_MODEL = "deepseek-r1"
EMBED_MODEL = "nomic-embed-text"

# --------------------------------分割线-----------------------------------


def main():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")

    # 以下为自动化逻辑：检查是否需要重新建库
    # 这里通过检查文件夹是否存在来判断
    # 注意：如果更换了文件，请手动删除 "db_storage"文件夹
    if not os.path.exists(DB_PATH):
        print(f"首次检测到新文件 {PDF_FILE_NAME}，正在初始化数据库")

        # 1.加载并切分
        loader = PyPDFLoader(PDF_FILE_NAME)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        splits = text_splitter.split_documents(docs)

        # 2.建立数据库
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

        # 分批写入防止502错误
        batch_size = 5
        for i in range(0, len(splits), batch_size):
            vectorstore.add_documents(documents=splits[i: i + batch_size])
            print(f"进度: {i + len(splits[i:i + batch_size])}/{len(splits)}", end="\r")
            time.sleep(0.3)
        print("\n 数据库构建完成！")
    else:
        # 直接加载现有数据库
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        print(f" 已加载 {PDF_FILE_NAME} 的现有知识库")

    # ---------- 对话环节 ----------
    print("\n +++++ 进入对话模式 (输入'q'退出) +++++ ")
    while True:
        question = input("\n 提问: ")
        if question.lower() in ['q', 'exit']: break

        # 检索并调用deepseek
        relevant_docs = vectorstore.similarity_search(question, k=10)
        context = "\n\n".join([d.page_content for d in relevant_docs])

        prompt = f"""
                你是一位专家。请根据提供的【参考资料】进行符合规则的回答。

                【规则】：
                1. 必须优先基于【参考资料】中的内容回答。
                2. 请尽量保留原始的专业术语。
                3. 如果资料内容不足以回答问题，可以结合专业知识进行补充，但必须明确标注“资料中未提及，以下为补充知识”。
                4. 回答逻辑要清晰，必要时可以使用Markdown的列表或表格。

                【参考资料】：
                {context}

                【待回答的问题】：
                {question}
                """

        response = ollama.chat(model=CHAT_MODEL, messages=[{'role': 'user', 'content': prompt}], stream=True)
        for chunk in response:
            print(chunk['message']['content'], end='', flush=True)
        print("\n" + "-" * 20)

if __name__ == "__main__":
    main()