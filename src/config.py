import torch
import os

# 路径配置
DATA_ROOT = 'data'      # 数据根目录 (raw_dxf 在这里面)
CHECKPOINT_DIR = "experiments/checkpoints"
MODEL_NAME = "best_model.pth"
EMBEDDING_DIR = "experiments/embeddings"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME)


# 训练超参数
BATCH_SIZE = 32         # 显存够的话可以大一点，比如 64
LEARNING_RATE = 0.001   # 学习率
NUM_EPOCHS = 100        # 训练轮数
HIDDEN_CHANNELS = 64    # 隐藏层维度
DROPOUT = 0.5           # 防止过拟合

# 数据集相关
# 根据你的实际数据类别修改这里
# 必须与 dataset.py 中的逻辑一致
CLASS_MAP = {
    '剪力墙': 0,
    '框剪': 1,
    '框架': 2,
    # 可以根据你的文件名增加更多映射
    'SHEAR': 0,
    'FRAME_SHEAR': 1,
    'FRAME': 2
}
NUM_CLASSES = 3         # 总类别数

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')