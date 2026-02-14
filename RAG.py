import os
import sys
import time
import glob  # 新增：用于查找文件
import ollama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 配置区域
DB_PATH = "./chroma_db_storage"
CHAT_MODEL = "deepseek-r1"
EMBED_MODEL = "nomic-embed-text"


# --------------------------------分割线-----------------------------------

def get_pdf_file():
    """
    自动查找当前目录下的PDF文件
    """
    # 获取当前目录下所有 .pdf 结尾的文件
    pdf_files = glob.glob("*.pdf")

    if not pdf_files:
        print("\n[错误] 当前文件夹下没有找到任何 PDF 文件！")
        print("请将 .exe 文件和 .pdf 文件放在同一个文件夹内。")
        return None

    if len(pdf_files) == 1:
        # 如果只有一个，直接使用
        print(f"\n[系统] 检测到文件: {pdf_files[0]}")
        return pdf_files[0]
    else:
        # 如果有多个，让用户选择
        print("\n[系统] 检测到多个 PDF 文件，请选择一个:")
        for idx, f in enumerate(pdf_files):
            print(f"  {idx + 1}. {f}")

        while True:
            selection = input("\n请输入序号选择文件 (例如 1): ")
            if selection.isdigit() and 1 <= int(selection) <= len(pdf_files):
                return pdf_files[int(selection) - 1]
            print("输入无效，请重新输入。")


def main():
    # 0. 确定运行环境（确保exe双击能找到文件）
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
        os.chdir(application_path)

    # 1. 动态获取 PDF 文件名
    current_pdf_name = get_pdf_file()
    if not current_pdf_name:
        input("\n按回车键退出...")
        return

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    # 2. 检查数据库状态
    # 注意：Chroma一旦创建，会一直保留之前的数据。
    # 为了防止新旧PDF数据混淆，这里加一个简单的判断逻辑：
    # 如果更换了PDF，强烈建议手动删除 chroma_db_storage 文件夹
    if os.path.exists(DB_PATH):
        print("\n[注意] 检测到已有知识库缓存。")
        print(f"如果 {current_pdf_name} 是新换的资料，请关闭本窗口，删除 'chroma_db_storage' 文件夹后重试。")
        print("如果是同一份资料，请按回车继续...")
        # input() # 如果想强制暂停可以取消注释，否则直接继续

    if not os.path.exists(DB_PATH):
        print(f"\n[初始化] 正在为 {current_pdf_name} 建立知识库...")

        # 加载并切分
        loader = PyPDFLoader(current_pdf_name)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        splits = text_splitter.split_documents(docs)

        # 建立数据库
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

        # 分批写入
        batch_size = 5
        for i in range(0, len(splits), batch_size):
            vectorstore.add_documents(documents=splits[i: i + batch_size])
            print(f"进度: {i + len(splits[i:i + batch_size])}/{len(splits)}", end="\r")
            time.sleep(0.3)  # 稍微降速，避免UI卡死
        print("\n[成功] 数据库构建完成！")
    else:
        # 直接加载现有数据库
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        print(f" 已加载现有知识库")

    # ---------- 对话环节 ----------
    print(f"\n +++++ 正在与 {current_pdf_name} 对话 (输入'q'退出) +++++ ")
    while True:
        try:
            question = input("\n 提问: ")
            if not question: continue
            if question.lower() in ['q', 'exit', '退出']: break

            retrieve_start = time.time()
            relevant_docs = vectorstore.similarity_search(question, k=10)
            retrieve_end = time.time()
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

            llm_start = time.time()
            response = ollama.chat(model=CHAT_MODEL, messages=[{'role': 'user', 'content': prompt}], stream=True)

            print("\n> AI回复:")
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)
            print("\n" + "-" * 20)
            llm_end = time.time()

            print(f"\n[检索耗时] {retrieve_end - retrieve_start:.3f} 秒")
            print(f"\n[LLM生成耗时] {llm_end - llm_start:.3f} 秒")

        except Exception as e:
            print(f"\n[出错] {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        input("\n程序发生严重错误，按回车键退出")