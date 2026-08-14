# StructSearch_v2

## 项目简介
工程结构检索系统 - 版本2

## 功能特性
1. 支持DXF文件解析
2. 图结构表示学习
3. 图神经网络监督分类训练（交叉熵）
4. 基于图嵌入的相似性检索

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 构建检索索引（首次必跑；图纸库更新后需重建）
python -m src.retrieval.cli --build

# 对单张图纸做结构体系分类 + Top-K 相似检索
python -m src.retrieval.cli --search data\raw\湘湖里_框架_H9.dxf

# 运行检索质量基准评估（Top-1/3/5、聚类指标、混淆矩阵、失败案例）
python -m src.retrieval.cli --benchmark
```

提示：命令需在项目根目录运行；cmd 中若中文显示乱码，先执行 `chcp 65001`。

## 项目结构
```
StructSearch_v2/
├── configs/          # 配置文件（retrieval.yaml 为主）
├── data/             # 数据目录（raw/ 原始 DXF、processed/ 图缓存）
├── src/              # 源代码
├── experiments/      # 实验输出（索引、运行日志与图）
├── tests/            # 测试代码
└── checkpoints/      # 模型产物
```

## 许可证
MIT License
