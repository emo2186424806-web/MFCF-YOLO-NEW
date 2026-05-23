import matplotlib.pyplot as plt
import os

# =========================
# 数据
# =========================
params_dict = {
    1.0: 1.439939,
    0.8: 1.434621,
    0.5: 1.418239,
    0.3: 1.055745,
    0.2: 0.909768,
    0.1: 0.877648,
}

gflops_dict = {
    1.0: 4.6,
    0.8: 4.5,
    0.5: 4.5,
    0.3: 4.1,
    0.2: 3.9,
    0.1: 3.6,
}

# 输出目录
save_dir = "pruning_factor_plots"
os.makedirs(save_dir, exist_ok=True)

# 横坐标按从大到小排序
x = sorted(params_dict.keys(), reverse=True)
params = [params_dict[i] for i in x]
gflops = [gflops_dict[i] for i in x]


def plot_curve(x, y, ylabel, title, color, marker, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color=color, marker=marker, linewidth=2.5, markersize=7)

    plt.xlabel("Pruning Factor")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.2f}", fontsize=8, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图像已保存: {save_path}")


if __name__ == "__main__":
    # 1. 参数量曲线
    plot_curve(
        x=x,
        y=params,
        ylabel="Parameters (M)",
        title="Effect of Pruning Factor on Parameters",
        color="#1f77b4",
        marker="o",
        save_path=os.path.join(save_dir, "curve_params.png")
    )

    # 2. GFLOPs 曲线
    plot_curve(
        x=x,
        y=gflops,
        ylabel="GFLOPs",
        title="Effect of Pruning Factor on GFLOPs",
        color="#ff7f0e",
        marker="s",
        save_path=os.path.join(save_dir, "curve_gflops.png")
    )
