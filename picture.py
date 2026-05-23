import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 全局绘图风格
# =========================
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

# =========================
# 训练结果文件
# =========================
results_files = [
    r"runs/detect/yolon 无预训练widerperson/results.csv",
    r"runs/detect/widerperson+adown/results.csv",
    r"runs/detect/widerperson+adown+wtconv/results.csv",
    r"runs/detect/widerperson+adown+wtconv+ccfm/results.csv",
]

custom_labels = [
    "YOLO11n",
    "YOLO11n + ADown",
    "YOLO11n + ADown + WTConv",
    "YOLO11n + ADown + WTConv + CCFM",
]


params_m = [2.62, 2.24, 1.93, 1.43]   # 单位: M
gflops = [6.6, 5.6, 4.7, 4.6]         # 单位: GFLOPs
fps = [84.81, 85.94, 93.92, 112.13]        # 单位: FPS

# 输出文件夹
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

# 统一颜色
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load_results_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


def plot_comparison(metrics, labels, results_files, custom_labels, save_name, layout=(2, 2)):
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(15, 10))
    axes = np.array(axes).flatten()

    for i, (metric_key, metric_label) in enumerate(zip(metrics, labels)):
        ax = axes[i]

        for j, (file_path, custom_label) in enumerate(zip(results_files, custom_labels)):
            df = load_results_csv(file_path)

            if "epoch" not in df.columns:
                print(f"'epoch' column not found in {file_path}. Available columns: {df.columns.tolist()}")
                continue

            if metric_key not in df.columns:
                print(f"'{metric_key}' column not found in {file_path}. Available columns: {df.columns.tolist()}")
                continue

            ax.plot(
                df["epoch"],
                df[metric_key],
                label=custom_label,
                color=colors[j % len(colors)],
                linewidth=2.2,
            )

        ax.set_title(metric_label, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.legend(frameon=True, loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)

    # 隐藏多余空子图
    for k in range(len(metrics), len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"图像已保存: {save_path}")
    plt.close()


def plot_bar_comparison(labels, values, ylabel, title, save_name, color):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=color, edgecolor="black", linewidth=1.0)

    plt.title(title, fontweight="bold")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # 在柱子上标数值
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"图像已保存: {save_path}")
    plt.close()

def plot_model_complexity(labels, params_m, gflops, fps):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        (params_m, "Parameters (M)", "Model Parameters Comparison", "#4c72b0"),
        (gflops, "GFLOPs", "Model GFLOPs Comparison", "#dd8452"),
        (fps, "FPS", "Model FPS Comparison", "#55a868"),
    ]

    for ax, (values, ylabel, title, color) in zip(axes, metrics):
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            values,
            width=0.45,   # 柱子变细
            color=color,
            edgecolor="black",
            linewidth=1.0
        )

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # 柱顶标数值
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    plt.tight_layout()
    save_path = os.path.join(save_dir, "model_complexity_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"图像已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    # =========================
    # 精度曲线
    # =========================
    metrics = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)"
    ]
    labels = [
        "Precision",
        "Recall",
        "mAP@50",
        "mAP@50-95"
    ]

    plot_comparison(
        metrics=metrics,
        labels=labels,
        results_files=results_files,
        custom_labels=custom_labels,
        save_name="precision_comparison.png",
        layout=(2, 2)
    )

    # =========================
    # 损失曲线
    # =========================
    loss_metrics = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss"
    ]
    loss_labels = [
        "Train Box Loss",
        "Train Class Loss",
        "Train DFL Loss",
        "Val Box Loss",
        "Val Class Loss",
        "Val DFL Loss"
    ]

    plot_comparison(
        metrics=loss_metrics,
        labels=loss_labels,
        results_files=results_files,
        custom_labels=custom_labels,
        save_name="loss_comparison.png",
        layout=(2, 3)
    )

    plot_comparison(
        metrics=["metrics/precision(B)"],
        labels=["Precision"],
        results_files=results_files,
        custom_labels=custom_labels,
        save_name="precision_only.png",
        layout=(1, 1)
    )
    # =========================
    # 参数量 / GFLOPs / FPS 柱状图
    # =========================
    plot_model_complexity(
        labels=custom_labels,
        params_m=params_m,
        gflops=gflops,
        fps=fps
    )
