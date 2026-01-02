#!/usr/bin/env python3
"""
RAG 职业课程生成系统 - 检索测试脚本

功能:
1. 加载向量数据库
2. 执行相似度搜索
3. 显示检索结果
"""

import os
# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
from typing import List

import colorama
from colorama import Fore, Style

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# 初始化 colorama
colorama.init()

# 配置
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def print_header(text: str) -> None:
    """打印标题"""
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text:^70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")


def print_info(text: str) -> None:
    """打印信息"""
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {text}")


def print_success(text: str) -> None:
    """打印成功信息"""
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {text}")


def print_error(text: str) -> None:
    """打印错误"""
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {text}")


def print_query(text: str) -> None:
    """打印查询"""
    print(f"{Fore.YELLOW}查询: {Style.RESET_ALL}{text}")


def print_result(index: int, doc, score: float) -> None:
    """打印检索结果"""
    print(f"\n{Fore.GREEN}[结果 {index + 1}]{Style.RESET_ALL} "
          f"{Fore.CYAN}相似度: {score:.4f}{Style.RESET_ALL}")

    # 打印元数据
    metadata = doc.metadata
    print(f"  {Fore.CYAN}文件:{Style.RESET_ALL} {metadata.get('filename', 'N/A')}")
    if 'page' in metadata:
        print(f"  {Fore.CYAN}页码:{Style.RESET_ALL} {metadata['page']}")

    # 打印内容
    content = doc.page_content
    if len(content) > 200:
        content = content[:200] + "..."
    print(f"  {Fore.CYAN}内容:{Style.RESET_ALL} {content}")


class LocalEmbeddings(Embeddings):
    """本地 Embedding 包装类，避免网络请求"""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.client = None

    def _load_model(self):
        """延迟加载模型"""
        if self.client is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # 尝试从本地缓存加载
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            local_model_dir = cache_dir / f"models--{self.model_name.replace('/', '--')}"

            if local_model_dir.exists():
                # 找到 snapshot 目录
                snapshot_dir = None
                for d in local_model_dir.iterdir():
                    if d.is_dir() and "snapshots" in str(d):
                        # 获取实际的 snapshot 目录
                        snapshots = list(d.glob("*/"))
                        if snapshots:
                            snapshot_dir = snapshots[0]
                            break

                if snapshot_dir:
                    print_info(f"从本地缓存加载: {snapshot_dir}")
                    self.client = SentenceTransformer(str(snapshot_dir), device=self.device)
                    print_success("模型加载完成")
                    return

            # 回退到在线加载（会使用缓存）
            print_info(f"加载模型: {self.model_name}")
            self.client = SentenceTransformer(self.model_name, device=self.device)
            print_success("模型加载完成")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        self._load_model()
        return self.client.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        self._load_model()
        return self.client.encode(text, normalize_embeddings=True).tolist()


def load_vector_store(persist_dir: Path) -> Chroma:
    """加载向量数据库"""
    print_info("加载向量数据库...")

    if not persist_dir.exists():
        print_error(f"向量数据库不存在: {persist_dir}")
        print_info("请先运行 ingest.py 进行数据入库")
        return None

    # 检测设备
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    # 使用本地 Embeddings（避免网络请求）
    embeddings = LocalEmbeddings(EMBEDDING_MODEL_NAME, device=device)

    # 加载数据库
    vector_store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    print_success(f"向量数据库加载成功")
    print(f"  存储路径: {persist_dir}")

    return vector_store


def search(vector_store: Chroma, query: str, k: int = 3) -> List:
    """执行相似度搜索"""
    print(f"\n{Fore.BLUE}正在搜索...{Style.RESET_ALL}")

    # 使用相似度搜索（带分数）
    results = vector_store.similarity_search_with_score(query, k=k)

    return results


def interactive_search(vector_store: Chroma) -> None:
    """交互式搜索"""
    print_header("交互式检索测试")

    print_info("输入查询问题，输入 'quit' 或 'exit' 退出\n")

    while True:
        try:
            query = input(f"{Fore.YELLOW}>>>{Style.RESET_ALL} ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print_info("\n退出检索测试")
                break

            # 执行搜索
            results = search(vector_store, query, k=3)

            # 显示结果
            print(f"\n{Fore.CYAN}找到 {len(results)} 个相关结果:{Style.RESET_ALL}")
            for i, (doc, score) in enumerate(results):
                print_result(i, doc, score)

            print()  # 空行

        except KeyboardInterrupt:
            print_info("\n\n退出检索测试")
            break
        except Exception as e:
            print_error(f"搜索出错: {e}")


def batch_search(vector_store: Chroma) -> None:
    """批量搜索测试"""
    print_header("批量检索测试")

    test_queries = [
        "网络管理员的工作职责是什么？",
        "信息安全需要掌握哪些技能？",
        "职业等级有哪些划分？",
        "如何防范网络攻击？",
        "数据备份的要求",
    ]

    print_info(f"执行 {len(test_queries)} 个测试查询\n")

    for query in test_queries:
        print_query(query)
        results = search(vector_store, query, k=2)

        for i, (doc, score) in enumerate(results):
            print_result(i, doc, score)

        print()  # 空行分隔
        print("-" * 70)
        print()


def main():
    """主函数"""
    print_header("RAG 职业课程生成系统 - 检索测试")

    # 加载向量数据库
    vector_store = load_vector_store(CHROMA_PERSIST_DIR)
    if vector_store is None:
        return

    # 获取数据库统计
    print_info("\n数据库统计:")
    collection = vector_store._collection
    count = collection.count()
    print(f"  总文档数: {count}")

    # 选择模式
    print(f"\n{Fore.CYAN}请选择测试模式:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}1{Style.RESET_ALL}. 批量测试 (预设查询)")
    print(f"  {Fore.GREEN}2{Style.RESET_ALL}. 交互测试 (手动输入)")
    print(f"  {Fore.GREEN}3{Style.RESET_ALL}. 两种都执行")

    choice = input(f"\n{Fore.YELLOW}选择 (1/2/3, 默认 2):{Style.RESET_ALL} ").strip()

    if choice == '1':
        batch_search(vector_store)
    elif choice == '3':
        batch_search(vector_store)
        print("\n" * 2)
        interactive_search(vector_store)
    else:
        interactive_search(vector_store)


if __name__ == "__main__":
    main()
