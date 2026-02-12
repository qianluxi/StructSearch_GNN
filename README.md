# StructSearch_v2

## 项目简介
工程结构检索系统 - 版本2

## 功能特性
1. 支持DXF文件解析
2. 图结构表示学习
3. 自监督对比学习
4. 相似性检索

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 预训练
python main.py --config configs/pretrain.yaml

# 微调
python main.py --config configs/finetune.yaml

# 检索
python main.py --config configs/retrieval.yaml
```

## 项目结构
```
StructSearch_v2/
├── configs/          # 配置文件
├── data/            # 数据目录
├── src/             # 源代码
├── experiments/     # 实验输出
├── scripts/         # 脚本文件
├── tests/           # 测试代码
└── notebooks/       # 分析笔记本
```

## 许可证
MIT License
