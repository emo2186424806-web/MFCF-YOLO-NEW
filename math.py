import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def silu(x):
    return x * sigmoid(x)


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)

    plt.figure(figsize=(10, 6))

    plt.plot(x, sigmoid(x), color="blue", linewidth=2, label="Sigmoid")
    plt.plot(x, tanh(x), color="green", linewidth=2, label="Tanh")
    plt.plot(x, relu(x), color="red", linewidth=2, label="ReLU")
    plt.plot(x, silu(x), color="purple", linewidth=2, label="SiLU")

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")

    plt.title("Activation Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("activation_functions.png", dpi=300)
    plt.close()

    print("图片已保存：activation_functions.png")
