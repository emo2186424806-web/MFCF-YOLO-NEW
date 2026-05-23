# 生成柱状图的分析代码
import matplotlib

matplotlib.use('Agg')  #
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
# 你的模型路径 (建议用 last.pt 或 best.pt)
MODEL_PATH = r"runs/new-pruning/widerperson+step2/weights/last.pt"
# 剪枝阈值 (图中红线的位置，通常设为 0.01 或 0.05)
THRESHOLD = 0.05


# ===========================================

def check_bn_weights():
    print(f" Loading model from: {MODEL_PATH}")

    # 1. 尝试加载模型
    try:
        model = YOLO(MODEL_PATH)
        net = model.model
    except:
        print(" YOLO wrapper failed, trying torch.load...")
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        net = ckpt['model']

    # 2. 收集所有 BN 层的权重绝对值
    bn_weights = []
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # 我们只关心权重的大小 (绝对值)
            weights = m.weight.data.abs().clone().cpu().numpy()
            bn_weights.extend(weights)

    bn_weights = np.array(bn_weights)
    total_params = len(bn_weights)

    # 3. 计算稀疏度
    # 小于阈值的权重被认为是“可剪掉的”
    prunable_count = (bn_weights < THRESHOLD).sum()
    sparsity_ratio = prunable_count / total_params

    print(f"\n --- Analysis Report ---")
    print(f"Total BN Parameters: {total_params}")
    print(f"Prunable Params (<{THRESHOLD}): {prunable_count}")
    print(f"Sparsity Ratio: {sparsity_ratio:.2%} (Expected > 40% for good pruning)")
    print(f"Min: {bn_weights.min():.4f}, Max: {bn_weights.max():.4f}, Mean: {bn_weights.mean():.4f}")

    # 4. 绘制直方图 (还原你截图的样式)
    plt.figure(figsize=(10, 6))

    # 绘制柱状图
    plt.hist(bn_weights, bins=100, range=(0, 2), color='blue', alpha=0.75, edgecolor='black', label='BN Weights')

    # 绘制红线 (Prune Zone)
    plt.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Prune Zone (<{THRESHOLD})')

    # 设置标题和标签
    plt.title(f"BN Weights Distribution (Sparsity: {sparsity_ratio:.1%})", fontsize=14)
    plt.xlabel("Weight Value (abs)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 保存
    save_name = "bn_sparsity_analysis4.png"
    plt.savefig(save_name, dpi=300)
    print(f"\n Plot saved to: {os.path.abspath(save_name)}")
    plt.show()


if __name__ == "__main__":
    check_bn_weights()