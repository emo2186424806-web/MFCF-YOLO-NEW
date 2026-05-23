import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


def get_layer_channels(model_path, target_layers):
    model = YOLO(model_path).model
    model.eval()

    channels = {}
    for idx in target_layers:
        m = model.model[idx]

        # C3k2 / C3k2_WTConv / 类似结构
        if hasattr(m, "cv2") and hasattr(m.cv2, "conv"):
            channels[idx] = m.cv2.conv.out_channels

        # 普通 Conv
        elif hasattr(m, "conv"):
            channels[idx] = m.conv.out_channels

        # ADown
        elif hasattr(m, "cv1") and hasattr(m, "cv2"):
            if hasattr(m.cv1, "conv") and hasattr(m.cv2, "conv"):
                c1 = m.cv1.conv.out_channels
                c2 = m.cv2.conv.out_channels
                channels[idx] = c1 + c2
            else:
                channels[idx] = None

        # SPPF / C2PSA 等
        elif hasattr(m, "cv1") and hasattr(m.cv1, "conv"):
            channels[idx] = m.cv1.conv.out_channels

        else:
            channels[idx] = None

    return channels


def plot_channel_comparison_multi(
    before_channels,
    prune03_channels,
    prune05_channels,
    prune08_channels,
    save_path="channel_pruning_multi_comparison.png",
):
    layers = list(before_channels.keys())

    before_vals = [before_channels[i] if before_channels[i] is not None else 0 for i in layers]
    prune03_vals = [prune03_channels[i] if prune03_channels[i] is not None else 0 for i in layers]
    prune05_vals = [prune05_channels[i] if prune05_channels[i] is not None else 0 for i in layers]
    prune08_vals = [prune08_channels[i] if prune08_channels[i] is not None else 0 for i in layers]

    x = np.arange(len(layers))
    width = 0.18

    plt.figure(figsize=(14, 7))

    bars1 = plt.bar(x - 1.5 * width, before_vals, width, label="Before Pruning", color="#4C72B0", edgecolor="black")
    bars2 = plt.bar(
        x - 0.5 * width, prune03_vals, width, label="Pruning Rate = 0.3", color="#55A868", edgecolor="black"
    )
    bars3 = plt.bar(
        x + 0.5 * width, prune05_vals, width, label="Pruning Rate = 0.5", color="#DD8452", edgecolor="black"
    )
    bars4 = plt.bar(
        x + 1.5 * width, prune08_vals, width, label="Pruning Rate = 0.8", color="#C44E52", edgecolor="black"
    )

    plt.xticks(x, [f"Layer {i}" for i in layers], rotation=20)
    plt.ylabel("Output Channels")
    plt.title("Channel Comparison Under Different Pruning Rates")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图片已保存: {save_path}")


if __name__ == "__main__":
    # 改成你的实际模型路径
    before_model_path = r"runs/pruning/step2/weights/best.pt"
    prune03_model_path = r"runs/pruning/step2/weights/prune0.3.pt"
    prune05_model_path = r"runs/pruning/step2/weights/prune0.5.pt"
    prune08_model_path = r"runs/pruning/step2/weights/prune0.8.pt"

    # 你要对比的关键层
    target_layers = [6, 8, 13, 16, 19, 22]

    before_channels = get_layer_channels(before_model_path, target_layers)
    prune03_channels = get_layer_channels(prune03_model_path, target_layers)
    prune05_channels = get_layer_channels(prune05_model_path, target_layers)
    prune08_channels = get_layer_channels(prune08_model_path, target_layers)

    print("剪枝前通道数:", before_channels)
    print("0.3剪枝率通道数:", prune03_channels)
    print("0.5剪枝率通道数:", prune05_channels)
    print("0.8剪枝率通道数:", prune08_channels)

    plot_channel_comparison_multi(
        before_channels,
        prune03_channels,
        prune05_channels,
        prune08_channels,
        save_path="channel_pruning_multi_comparison.png",
    )
