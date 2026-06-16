import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from ultralytics import YOLO

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文
plt.rcParams["axes.unicode_minus"] = False


def extract_tensor_chw(x):
    """从输出中提取第一个4维特征图的 C,H,W."""
    if isinstance(x, torch.Tensor):
        if x.ndim == 4:
            _, c, h, w = x.shape
            return c, h, w
        return None

    if isinstance(x, (list, tuple)):
        for item in x:
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                _, c, h, w = item.shape
                return c, h, w
    return None


def get_model_structure(model_path, imgsz=640, device="cpu"):
    """提取每层输出结构信息."""
    yolo = YOLO(model_path)
    model = yolo.model.to(device)
    model.eval()

    layers = model.model
    records = []
    hooks = []

    def make_hook(idx, module):
        def hook(module, inputs, outputs):
            chw = extract_tensor_chw(outputs)
            if chw is None:
                c, h, w = None, None, None
            else:
                c, h, w = chw

            records.append(
                {"idx": idx, "from": getattr(module, "f", "-"), "module": type(module).__name__, "C": c, "H": h, "W": w}
            )

        return hook

    for i, m in enumerate(layers):
        hooks.append(m.register_forward_hook(make_hook(i, m)))

    x = torch.randn(1, 3, imgsz, imgsz).to(device)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    df = pd.DataFrame(records)
    return df


def build_compare_df(before_df, after_df):
    """按层号对齐剪枝前后结构."""
    df = before_df.merge(after_df, on="idx", how="outer", suffixes=("_before", "_after"))

    # 模块名优先取 before
    df["module"] = df["module_before"].fillna(df["module_after"])
    df["from"] = df["from_before"].fillna(df["from_after"])

    df["C_change"] = df["C_after"] - df["C_before"]
    df["C_prune_ratio"] = 1 - (df["C_after"] / df["C_before"])

    return df[
        [
            "idx",
            "from",
            "module",
            "C_before",
            "H_before",
            "W_before",
            "C_after",
            "H_after",
            "W_after",
            "C_change",
            "C_prune_ratio",
        ]
    ]


def plot_channel_compare(compare_df, save_path):
    """绘制剪枝前后各层通道数对比图."""
    plot_df = compare_df.dropna(subset=["C_before", "C_after"]).copy()

    x = plot_df["idx"].tolist()
    c_before = plot_df["C_before"].tolist()
    c_after = plot_df["C_after"].tolist()

    plt.figure(figsize=(12, 6))
    plt.plot(x, c_before, marker="o", linewidth=2, label="剪枝前通道数")
    plt.plot(x, c_after, marker="s", linewidth=2, label="剪枝后通道数")

    for xi, yi in zip(x, c_after):
        plt.text(xi, yi, str(int(yi)), fontsize=8, ha="center", va="bottom")

    plt.xlabel("层号")
    plt.ylabel("输出通道数 C")
    plt.title("剪枝前后网络各层输出通道数对比")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prune_ratio(compare_df, save_path):
    """绘制各层通道剪枝比例柱状图."""
    plot_df = compare_df.dropna(subset=["C_before", "C_after"]).copy()
    plot_df = plot_df[plot_df["C_before"] > 0]

    x = plot_df["idx"].tolist()
    ratio = (1 - plot_df["C_after"] / plot_df["C_before"]) * 100

    plt.figure(figsize=(12, 6))
    plt.bar(x, ratio, color="#4C72B0")
    for xi, yi in zip(x, ratio):
        plt.text(xi, yi, f"{yi:.1f}%", fontsize=8, ha="center", va="bottom")

    plt.xlabel("层号")
    plt.ylabel("通道剪枝比例 (%)")
    plt.title("剪枝后各层通道压缩比例")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    before_model = r"runs/new-pruning/citypersons+step1/weights/best.pt"  # 剪枝前模型
    after_model = r"runs/new-pruning/widerperson+step5/student/weights/best.pt"  # 剪枝后模型

    save_dir = r"structure_compare1"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imgsz = 640

    before_df = get_model_structure(before_model, imgsz=imgsz, device=device)
    after_df = get_model_structure(after_model, imgsz=imgsz, device=device)

    compare_df = build_compare_df(before_df, after_df)

    csv_path = os.path.join(save_dir, "network_structure_compare.csv")
    compare_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_channel_compare(compare_df, os.path.join(save_dir, "channels_before_after.png"))

    plot_prune_ratio(compare_df, os.path.join(save_dir, "channel_prune_ratio.png"))

    print("结构对比表已保存：", csv_path)
    print("通道对比图已保存：", os.path.join(save_dir, "channels_before_after.png"))
    print("剪枝比例图已保存：", os.path.join(save_dir, "channel_prune_ratio.png"))
    print(compare_df)
