import os
import shutil
from src.dataset import StructureDataset

def main():
    data_root = 'data'
    processed_dir = os.path.join(data_root, 'processed')
    raw_dir_wrong = os.path.join(data_root, 'raw') # 那个自动生成的空文件夹

    # 1. 清理旧数据
    if os.path.exists(processed_dir):
        print(f"正在删除旧数据: {processed_dir}")
        shutil.rmtree(processed_dir) # 强制删除文件夹及内容
    
    if os.path.exists(raw_dir_wrong):
        print(f"正在删除多余文件夹: {raw_dir_wrong}")
        shutil.rmtree(raw_dir_wrong)

    print("-" * 30)
    print("开始重新生成数据集 (读取 data/raw_dxf)...")
    
    # 2. 初始化 Dataset，这将自动触发 process()
    # 只要 processed 文件夹不存在，它就会重新扫描 raw_dxf
    dataset = StructureDataset(root=data_root)

    print("-" * 30)
    print(f"生成完毕！数据集统计:")
    print(f"样本总数: {len(dataset)}")
    print(f"特征维度: {dataset.num_features}")
    print(f"类别总数: {dataset.num_classes}")
    
    # 打印前3个样本的文件名，确认读取正确
    print("\n前3个样本示例:")
    for i in range(min(3, len(dataset))):
        data = dataset[i]
        print(f"  [{i}] 文件: {getattr(data, 'filename', 'Unknown')} | 类别: {data.y.item()} | 节点数: {data.num_nodes}")

if __name__ == "__main__":
    main()