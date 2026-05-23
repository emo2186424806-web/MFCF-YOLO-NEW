import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False


def draw_wtconv_diagram():
    _fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("WTConv2d 小波卷积内部结构图", fontsize=18, pad=20)

    # 颜色
    color_input = "#a8d8ff"
    color_wt = "#ffd3b6"
    color_conv = "#c7f380"
    color_res = "#ffb3ba"
    color_output = "#d4bfff"

    def box(x1, y1, x2, y2, label, color):
        ax.add_patch(
            FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle="round,pad=0.1", facecolor=color, edgecolor="black")
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, ha="center", va="center", fontsize=11)

    # 输入
    box(1, 8.5, 2.5, 9.5, "输入特征", color_input)

    # 分支1：基础卷积
    box(1, 7, 2.5, 8, "基础卷积\n+ Scale", color_conv)
    ax.arrow(1.75, 8.4, 0, -0.4, head_width=0.1, head_length=0.1)

    # 分支2：小波卷积主干
    box(4, 8.5, 5.5, 9.5, "小波分解\n(LL + LH+HL+HH)", color_wt)
    ax.arrow(2.6, 9, 0.6, 0, head_width=0.1, head_length=0.1)

    box(4, 7, 5.5, 8, "高频特征卷积\n+ Scale", color_conv)
    ax.arrow(4.75, 8.4, 0, -0.4, head_width=0.1, head_length=0.1)

    box(4, 5.5, 5.5, 6.5, "小波逆变换\n重构", color_wt)
    ax.arrow(4.75, 6.9, 0, -0.4, head_width=0.1, head_length=0.1)

    # 残差相加
    box(2.5, 4, 5.5, 5.5, "残差相加\n基础特征 + 小波重构特征", color_res)
    ax.arrow(1.75, 6.8, 0, -1.3, head_width=0.1, head_length=0.1)
    ax.arrow(4.75, 5.4, 0, -0.4, head_width=0.1, head_length=0.1)

    # 输出
    box(3, 2.5, 5, 3.5, "输出特征", color_output)
    ax.arrow(4, 3.9, 0, -0.4, head_width=0.1, head_length=0.1)

    plt.tight_layout()
    plt.show()


def draw_c3k2_wtconv_diagram():
    _fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("C3k2_WTConv 整体模块结构图（YOLO 风格）", fontsize=18, pad=20)

    color_in = "#e0f7fa"
    color_cv = "#ffccbc"
    color_bottleneck = "#c8e6c9"
    color_cat = "#f8bbd0"
    color_out = "#d1c4e9"

    def block(x1, y1, x2, y2, txt, c):
        ax.add_patch(
            FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle="round,pad=0.1", facecolor=c, edgecolor="black")
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, txt, ha="center", va="center", fontsize=10)

    block(1, 7, 2.5, 8, "输入", color_in)
    block(1, 5.5, 2.5, 6.5, "Conv 1x1", color_cv)
    ax.arrow(1.75, 6.9, 0, -0.3, head_width=0.08)

    block(0.5, 4, 2, 5, "Split", color_cv)
    block(3, 4, 4.5, 5, "Bottleneck_WTConv\n(带小波卷积)", color_bottleneck)
    ax.arrow(2.6, 4.7, 0.3, 0, head_width=0.08)
    ax.arrow(2.2, 4.7, 0.3, 0, head_width=0.08)

    block(1, 2.5, 4.5, 3.5, "Concat 拼接", color_cat)
    ax.arrow(1.2, 3.9, 0, -0.3, head_width=0.08)
    ax.arrow(3.7, 3.9, 0, -0.3, head_width=0.08)

    block(1.5, 1, 4, 2, "Conv 1x1 输出", color_cv)
    ax.arrow(2.75, 2.4, 0, -0.3, head_width=0.08)

    block(1.5, 0, 4, 0.8, "输出特征", color_out)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_wtconv_diagram()  # 画 WTConv2d 内部结构
    draw_c3k2_wtconv_diagram()  # 画 C3k2_WTConv 整体结构
