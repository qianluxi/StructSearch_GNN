#!/bin/bash
# build_index.sh
# 脚本说明

set -e

echo "开始执行 build_index.sh"

# 激活Python环境（如果需要）
# source venv/bin/activate

# 主命令
python main.py "$@"

echo "build_index.sh 执行完成"
