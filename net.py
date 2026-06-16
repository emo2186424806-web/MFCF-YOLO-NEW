import torch

from ultralytics import YOLO


def shape_to_str(x):
    """把张量 / 列表 / 元组的 shape 转成可读字符串."""
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    elif isinstance(x, (list, tuple)):
        parts = []
        for item in x:
            if isinstance(item, torch.Tensor):
                parts.append(str(tuple(item.shape)))
            else:
                parts.append(type(item).__name__)
        return "[" + ", ".join(parts) + "]"
    else:
        return type(x).__name__


def get_chw(x):
    """提取输出中的 C, H, W."""
    if isinstance(x, torch.Tensor):
        if x.ndim == 4:
            _, c, h, w = x.shape
            return f"C={c}, H={h}, W={w}"
        elif x.ndim == 3:
            c, h, w = x.shape
            return f"C={c}, H={h}, W={w}"
        else:
            return "-"
    elif isinstance(x, (list, tuple)):
        infos = []
        for i, item in enumerate(x):
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                _, c, h, w = item.shape
                infos.append(f"[{i}] C={c}, H={h}, W={w}")
        return "; ".join(infos) if infos else "-"
    else:
        return "-"


def inspect_model_shapes(model_path, imgsz=640, device="cpu"):
    model = YOLO(model_path)
    net = model.model.to(device)
    net.eval()

    layers = net.model
    records = []
    hooks = []

    def make_hook(idx, module):
        def hook(module, inputs, outputs):
            in_shape = shape_to_str(inputs[0]) if len(inputs) > 0 else "-"
            out_shape = shape_to_str(outputs)
            chw_info = get_chw(outputs)

            from_idx = getattr(module, "f", "-")
            module_name = type(module).__name__

            records.append(
                {
                    "idx": idx,
                    "from": from_idx,
                    "module": module_name,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "chw": chw_info,
                }
            )

        return hook

    for i, m in enumerate(layers):
        hooks.append(m.register_forward_hook(make_hook(i, m)))

    x = torch.randn(1, 3, imgsz, imgsz).to(device)

    with torch.no_grad():
        _ = net(x)

    for h in hooks:
        h.remove()

    print("=" * 130)
    print(f"{'Idx':<5} {'From':<10} {'Module':<25} {'Input Shape':<25} {'Output Shape':<35} {'C,H,W'}")
    print("=" * 130)
    for r in records:
        print(
            f"{r['idx']:<5} {r['from']!s:<10} {r['module']:<25} {r['input_shape']:<25} {r['output_shape']:<35} {r['chw']}"
        )
    print("=" * 130)


if __name__ == "__main__":
    model_path = r"ultralytics/cfg/models/11/ccfm.yaml"  # 改成你的模型路径，比如 best.pt / prune.pt / yaml
    inspect_model_shapes(model_path, imgsz=640, device="cuda:0")
