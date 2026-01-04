#!/usr/bin/env python3
"""
Graph-Enhanced Hybrid Retrieval Testing Script

This script tests the hybrid retrieval system that combines vector similarity
search with knowledge graph traversal for multi-hop reasoning.

Usage:
    # Basic test (interactive mode)
    python src/test_hybrid_retrieve.py

    # Batch test with preset queries
    python src/test_hybrid_retrieve.py --mode batch

    # Custom fusion weights
    python src/test_hybrid_retrieve.py --vector-weight 0.5 --graph-weight 0.5

    # Filter by experience level
    python src/test_hybrid_retrieve.py --experience junior

    # Filter by skill complexity
    python src/test_hybrid_retrieve.py --complexity medium

    # Compare with vector-only retrieval
    python src/test_hybrid_retrieve.py --compare
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# Set HuggingFace mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import colorama
from colorama import Fore, Style

# Add project root to path (for pickle compatibility)
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
SRC_DIR = SCRIPT_DIR
sys.path.insert(0, str(BASE_DIR))

from src.retriever import HybridRetriever, RetrieverConfig, ScoringMethod
from src.graph.graph_builder import VocationalGraphBuilder
from src.test_retrieve import LocalEmbeddings

# Initialize colorama
colorama.init()


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
GRAPH_PATH = DATA_DIR / "output" / "graph.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Test queries for batch mode
TEST_QUERIES = [
    "网络管理员的工作职责是什么？",
    "信息安全需要掌握哪些技能？",
    "会配H3C交换机吗？",  # Multi-hop example
    "Python后端开发需要什么技能？",
    "如何防范网络攻击？",
]


# ============================================================================
# Utility Functions
# ============================================================================
def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text:^70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {text}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {text}")


def print_query(text: str) -> None:
    """Print a query."""
    print(f"{Fore.YELLOW}查询:{Style.RESET_ALL} {text}")


def print_result(
    index: int,
    result,
    show_graph_info: bool = True,
) -> None:
    """Print a retrieval result."""
    # Score display
    score_text = f"{Fore.GREEN}融合分数: {result.fused_score:.4f}{Style.RESET_ALL}"
    vector_text = f"向量: {result.vector_score:.4f}"
    graph_text = f"图谱: {result.graph_score:.4f}"

    print(f"\n{Fore.CYAN}[结果 {index + 1}]{Style.RESET_ALL} {score_text}")
    print(f"  {Fore.CYAN}分数详情:{Style.RESET_ALL} {vector_text} | {graph_text}")

    # Metadata
    metadata = result.document.metadata
    print(f"  {Fore.CYAN}文件:{Style.RESET_ALL} {metadata.get('filename', 'N/A')}")
    if 'page' in metadata:
        print(f"  {Fore.CYAN}页码:{Style.RESET_ALL} {metadata['page']}")

    # Content preview
    content = result.document.page_content
    if len(content) > 200:
        content = content[:200] + "..."
    print(f"  {Fore.CYAN}内容:{Style.RESET_ALL} {content}")

    # Graph info
    if show_graph_info and result.graph_paths:
        print(f"  {Fore.CYAN}图谱路径:{Style.RESET_ALL} {len(result.graph_paths)} 条")
        for i, path in enumerate(result.graph_paths[:3]):  # Show max 3 paths
            path_str = " → ".join([
                Path(p).name if len(p) > 20 else p
                for p in path[:3]  # Truncate long paths
            ])
            print(f"    {i + 1}. {path_str}...")


# ============================================================================
# Loading Functions
# ============================================================================
def load_components():
    """Load vector store and graph."""
    import torch
    from langchain_community.vectorstores import Chroma

    # Load vector store
    print_info("加载向量数据库...")
    if not CHROMA_PERSIST_DIR.exists():
        print_error(f"向量数据库不存在: {CHROMA_PERSIST_DIR}")
        print_info("请先运行 python src/ingest.py 进行数据入库")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    embeddings = LocalEmbeddings(EMBEDDING_MODEL_NAME, device=device)
    vector_store = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )
    print_success("向量数据库加载成功")

    # Load graph
    print_info("加载知识图谱...")
    if not GRAPH_PATH.exists():
        print_error(f"图谱文件不存在: {GRAPH_PATH}")
        print_info("请先运行 python src/graph/build_batch.py 构建图谱")
        return None, None

    graph = VocationalGraphBuilder()
    graph.load(str(GRAPH_PATH))
    stats = graph.get_statistics()
    print_success(f"知识图谱加载成功")
    print(f"  节点数: {stats['total_nodes']}")
    print(f"  边数: {stats['total_edges']}")

    return vector_store, graph


# ============================================================================
# Test Modes
# ============================================================================
def interactive_search(retriever: HybridRetriever, args):
    """Interactive search mode."""
    print_header("交互式图谱增强检索测试")

    print_info("输入查询问题，输入 'quit' 或 'exit' 退出\n")
    print_info(f"配置: 向量权重={args.vector_weight}, 图谱权重={args.graph_weight}")
    if args.experience:
        print_info(f"经验过滤: {args.experience}")
    if args.complexity:
        print_info(f"复杂度过滤: {args.complexity}")
    print()

    while True:
        try:
            query = input(f"{Fore.YELLOW}>>>{Style.RESET_ALL} ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print_info("\n退出检索测试")
                break

            # Execute search
            if args.experience:
                results = retriever.retrieve_by_experience(
                    query, args.experience, top_k=args.top_k
                )
            elif args.complexity:
                results = retriever.retrieve_by_complexity(
                    query, args.complexity, top_k=args.top_k
                )
            else:
                results = retriever.retrieve(query, top_k=args.top_k)

            # Display results
            print(f"\n{Fore.CYAN}找到 {len(results)} 个相关结果:{Style.RESET_ALL}")
            for i, result in enumerate(results):
                print_result(i, result)

            print()

        except KeyboardInterrupt:
            print_info("\n\n退出检索测试")
            break
        except Exception as e:
            print_error(f"搜索出错: {e}")
            import traceback
            traceback.print_exc()


def batch_search(retriever: HybridRetriever, args):
    """Batch search mode with preset queries."""
    print_header("批量图谱增强检索测试")

    print_info(f"配置: 向量权重={args.vector_weight}, 图谱权重={args.graph_weight}")
    print_info(f"执行 {len(TEST_QUERIES)} 个测试查询\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"{Fore.MAGENTA}[查询 {i}/{len(TEST_QUERIES)}]{Style.RESET_ALL}", end=" ")
        print_query(query)

        try:
            results = retriever.retrieve(query, top_k=args.top_k)

            print(f"{Fore.CYAN}找到 {len(results)} 个相关结果:{Style.RESET_ALL}")
            for j, result in enumerate(results):
                print_result(j, result)

        except Exception as e:
            print_error(f"查询出错: {e}")

        print()  # Separator
        print("-" * 70)
        print()


def compare_mode(retriever: HybridRetriever, args):
    """Compare hybrid vs vector-only retrieval."""
    print_header("混合检索 vs 向量检索 对比")

    print_info("将对比以下查询的混合检索和纯向量检索结果:")
    for query in TEST_QUERIES[:3]:  # Show first 3
        print(f"  - {query}")
    print()

    for query in TEST_QUERIES[:3]:
        print(f"\n{Fore.MAGENTA}[查询]{Style.RESET_ALL} {query}")
        print("=" * 70)

        # Vector-only results
        print(f"\n{Fore.BLUE}向量检索结果:{Style.RESET_ALL}")
        vector_results = retriever._vector_search(query)
        for i, (doc, score) in enumerate(vector_results[:3]):
            print(f"  [{i + 1}] 分数: {score:.4f}")
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"      内容: {content}")

        # Hybrid results
        print(f"\n{Fore.GREEN}混合检索结果:{Style.RESET_ALL}")
        hybrid_results = retriever.retrieve(query, top_k=3)
        for i, result in enumerate(hybrid_results):
            print(f"  [{i + 1}] 融合: {result.fused_score:.4f} (向量: {result.vector_score:.4f}, 图谱: {result.graph_score:.4f})")
            content = result.document.page_content[:100] + "..." if len(result.document.page_content) > 100 else result.document.page_content
            print(f"      内容: {content}")

        print()
        print("-" * 70)


# ============================================================================
# Main
# ============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Graph-Enhanced Hybrid Retrieval Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/test_hybrid_retrieve.py
  python src/test_hybrid_retrieve.py --mode batch
  python src/test_hybrid_retrieve.py --vector-weight 0.5 --graph-weight 0.5
  python src/test_hybrid_retrieve.py --experience junior
  python src/test_hybrid_retrieve.py --compare
        """
    )

    # Configuration
    parser.add_argument(
        "--vector-weight", type=float, default=0.6,
        help="Weight for vector similarity (default: 0.6)"
    )
    parser.add_argument(
        "--graph-weight", type=float, default=0.4,
        help="Weight for graph relevance (default: 0.4)"
    )

    # Filters
    parser.add_argument(
        "--experience",
        choices=["entry", "junior", "mid_level", "senior", "expert"],
        help="Filter by experience level"
    )
    parser.add_argument(
        "--complexity",
        choices=["low", "medium", "high"],
        help="Filter by skill complexity"
    )

    # Output
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)"
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "compare"],
        default="interactive",
        help="Test mode (default: interactive)"
    )

    args = parser.parse_args()

    print_header("图谱增强混合检索系统")

    # Load components
    vector_store, graph = load_components()
    if vector_store is None or graph is None:
        return 1

    # Initialize config
    config = RetrieverConfig(
        vector_weight=args.vector_weight,
        graph_weight=args.graph_weight,
    )

    # Initialize retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        graph_builder=graph,
        config=config,
    )

    print_success("混合检索器初始化完成")

    # Run selected mode
    if args.mode == "batch":
        batch_search(retriever, args)
    elif args.mode == "compare":
        compare_mode(retriever, args)
    else:
        interactive_search(retriever, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
