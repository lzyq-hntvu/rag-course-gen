"""
RAG 职业课程生成系统 - 配置文件
"""

from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 目录配置
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
SRC_DIR = BASE_DIR / "src"

# 向量数据库配置
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# 文档处理配置
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PyPDFLoader',
    '.txt': 'TextLoader',
    '.md': 'TextLoader',
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding 模型配置
# 中文推荐: paraphrase-multilingual-MiniLM-L12-v2 (384维, 多语言)
# 英文推荐: all-MiniLM-L6-v2 (384维, 英文)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 日志配置
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "rag_system.log"
