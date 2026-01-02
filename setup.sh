#!/bin/bash
# RAG 职业课程生成系统 - 一键安装脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VENV_DIR=".venv"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RAG 职业课程生成系统 - 环境安装${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 Python 版本
echo -e "${YELLOW}[1/6] 检查 Python 版本...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "  ${RED}错误: 未找到 python3${NC}"
    exit 1
fi
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "  当前 Python 版本: ${GREEN}${python_version}${NC}"
echo ""

# 创建虚拟环境
echo -e "${YELLOW}[2/6] 创建虚拟环境...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "  ${YELLOW}虚拟环境已存在，跳过创建${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "  ${GREEN}虚拟环境创建完成: ${VENV_DIR}/${NC}"
fi
echo ""

# 激活虚拟环境
echo -e "${YELLOW}[3/6] 激活虚拟环境...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "  ${GREEN}虚拟环境已激活${NC}"
echo ""

# 检查 CUDA
echo -e "${YELLOW}[4/6] 检查 CUDA 环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk -F': ' '{print $2}' | awk '{print $1}')
    echo -e "  ${GREEN}检测到 GPU${NC}"
    echo -e "  CUDA 版本: ${GREEN}${cuda_version}${NC}"
    HAS_GPU=true
else
    echo -e "  ${YELLOW}未检测到 GPU，将安装 CPU 版本${NC}"
    HAS_GPU=false
fi
echo ""

# 升级 pip
echo -e "${YELLOW}升级 pip...${NC}"
python -m pip install --upgrade pip -q
echo -e "  ${GREEN}pip 已升级${NC}"
echo ""

# 安装 PyTorch
echo -e "${YELLOW}安装 PyTorch...${NC}"
if [ "$HAS_GPU" = true ]; then
    echo "  安装 CUDA 版本 PyTorch (使用清华镜像)..."
    pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "  安装 CPU 版本 PyTorch..."
    pip install torch
fi
echo -e "  ${GREEN}PyTorch 安装完成${NC}"
echo ""

# 安装项目依赖
echo -e "${YELLOW}[5/6] 安装项目依赖...${NC}"
pip install -r requirements.txt
echo -e "  ${GREEN}依赖安装完成${NC}"
echo ""

# 验证安装
echo -e "${YELLOW}[6/6] 验证安装...${NC}"
echo ""
python -c "
import torch
import langchain
import chromadb
import sentence_transformers
import pydantic

print('✓ PyTorch 版本:', torch.__version__)
print('  GPU 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU 设备:', torch.cuda.get_device_name(0))
print('')
print('✓ LangChain 版本:', langchain.__version__)
print('✓ ChromaDB 版本:', chromadb.__version__)
print('✓ SentenceTransformers 版本:', sentence_transformers.__version__)
print('✓ Pydantic 版本:', pydantic.__version__)
"
echo ""

# 保存激活命令到文件
cat > activate.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "虚拟环境已激活"
EOF
chmod +x activate.sh

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "后续使用请先激活虚拟环境:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo -e "  或运行: ${YELLOW}./activate.sh${NC}"
echo ""
