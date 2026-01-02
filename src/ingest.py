#!/usr/bin/env python3
"""
RAG 职业课程生成系统 - 数据入库脚本

功能:
1. 读取 docs/ 目录下所有文件
2. 文档切分
3. 向量化存储
4. 元数据管理
"""

import os
# 设置 HuggingFace 镜像（必须在导入 transformers 前设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import colorama
from colorama import Fore, Style
from tqdm import tqdm

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 初始化 colorama
colorama.init()

# 配置
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# 支持的文件类型
SUPPORTED_EXTENSIONS = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
}

# 文本分割配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding 模型配置
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def print_header(text: str) -> None:
    """打印标题"""
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text:^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")


def print_info(text: str) -> None:
    """打印信息"""
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {text}")


def print_success(text: str) -> None:
    """打印成功信息"""
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {text}")


def print_warning(text: str) -> None:
    """打印警告"""
    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {text}")


def print_error(text: str) -> None:
    """打印错误"""
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {text}")


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """获取文件元数据"""
    stat = file_path.stat()
    return {
        "filename": file_path.name,
        "file_path": str(file_path.absolute()),
        "file_extension": file_path.suffix,
        "file_size": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def scan_documents(docs_dir: Path) -> List[Path]:
    """扫描文档目录，返回所有支持的文件"""
    print_info("扫描文档目录...")

    files = []
    for ext in SUPPORTED_EXTENSIONS.keys():
        matched = list(docs_dir.glob(f"**/*{ext}"))
        # 过滤掉 Zone.Identifier 等系统文件
        matched = [f for f in matched if not f.name.startswith("._") and "Zone.Identifier" not in f.name]
        files.extend(matched)

    # 按文件名排序
    files.sort(key=lambda x: x.name)

    print_success(f"找到 {len(files)} 个文档")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    return files


def load_documents(file_paths: List[Path]) -> List[Document]:
    """加载文档"""
    print_info("\n加载文档内容...")

    all_docs = []
    failed_files = []

    for file_path in tqdm(file_paths, desc="加载文档", unit="文件"):
        try:
            ext = file_path.suffix.lower()
            loader_class = SUPPORTED_EXTENSIONS.get(ext)

            if loader_class is None:
                print_warning(f"不支持的文件类型: {file_path.name}")
                continue

            # 使用绝对路径加载
            loader = loader_class(str(file_path.absolute()))
            docs = loader.load()

            # 添加元数据
            metadata = get_file_metadata(file_path)
            for doc in docs:
                doc.metadata.update(metadata)

            all_docs.extend(docs)

        except Exception as e:
            print_error(f"加载文件失败 {file_path.name}: {e}")
            failed_files.append(file_path.name)

    if failed_files:
        print_warning(f"以下文件加载失败: {', '.join(failed_files)}")

    print_success(f"共加载 {len(all_docs)} 个文档片段")
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """切分文档"""
    print_info("\n切分文档...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )

    split_docs = splitter.split_documents(documents)

    print_success(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个片段")
    print(f"  平均片段长度: {sum(len(d.page_content) for d in split_docs) / len(split_docs):.0f} 字符")

    return split_docs


def create_vector_store(documents: List[Document], persist_dir: Path) -> Chroma:
    """创建向量存储"""
    print_info("\n初始化 Embedding 模型...")

    # 使用 GPU 加速（如果可用）
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")
    print(f"  模型: {EMBEDDING_MODEL_NAME}")

    # 使用国内镜像加速下载
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
        show_progress=True,
    )

    print_info("\n创建向量数据库...")

    # 删除已存在的数据库
    if persist_dir.exists():
        print_warning(f"删除已存在的向量数据库: {persist_dir}")
        import shutil
        shutil.rmtree(persist_dir)

    # 持久化目录
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 创建向量存储
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    print_success(f"向量数据库创建成功: {persist_dir}")
    return vector_store


def print_statistics(documents: List[Document], vector_store: Chroma) -> None:
    """打印统计信息"""
    print_header("入库统计")

    # 文档统计
    print(f"{Fore.CYAN}文档信息:{Style.RESET_ALL}")
    print(f"  总片段数: {len(documents)}")

    # 按文件类型统计
    file_types = {}
    for doc in documents:
        ext = doc.metadata.get("file_extension", "unknown")
        file_types[ext] = file_types.get(ext, 0) + 1

    print(f"\n{Fore.CYAN}文件类型分布:{Style.RESET_ALL}")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} 个片段")

    # 元数据示例
    if documents:
        print(f"\n{Fore.CYAN}元数据示例:{Style.RESET_ALL}")
        sample = documents[0].metadata
        for key, value in list(sample.items())[:5]:
            print(f"  {key}: {value}")

    # 向量数据库统计
    print(f"\n{Fore.CYAN}向量数据库:{Style.RESET_ALL}")
    print(f"  存储路径: {CHROMA_PERSIST_DIR.absolute()}")
    print(f"  Embedding 模型: {EMBEDDING_MODEL_NAME}")
    print(f"  向量维度: 384 (paraphrase-multilingual-MiniLM-L12-v2)")


def main():
    """主函数"""
    print_header("RAG 职业课程生成系统 - 数据入库")

    # 检查文档目录
    if not DOCS_DIR.exists():
        print_error(f"文档目录不存在: {DOCS_DIR.absolute()}")
        print_info("请将文档放入 docs/ 目录后重试")
        sys.exit(1)

    # 1. 扫描文档
    file_paths = scan_documents(DOCS_DIR)
    if not file_paths:
        print_warning("未找到任何文档")
        sys.exit(0)

    # 2. 加载文档
    documents = load_documents(file_paths)
    if not documents:
        print_error("文档加载失败")
        sys.exit(1)

    # 3. 切分文档
    split_docs = split_documents(documents)

    # 4. 创建向量存储
    vector_store = create_vector_store(split_docs, CHROMA_PERSIST_DIR)

    # 5. 打印统计信息
    print_statistics(split_docs, vector_store)

    print_success("\n数据入库完成！")
    print(f"\n{Fore.CYAN}后续使用:{Style.RESET_ALL}")
    print(f"  向量数据库已保存至: {CHROMA_PERSIST_DIR.absolute()}")
    print(f"  可使用 query.py 查询数据\n")


if __name__ == "__main__":
    main()
