from ultralytics import YOLO
import torch
import time

# 这里只改成你当前要测试的 yaml 文件
model_yaml = r"runs/new-pruning/widerperson+step5/student/weights/best.pt"

# 加载模型
model = YOLO(model_yaml).model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# 输入尺寸
imgsz = 640
x = torch.randn(1, 3, imgsz, imgsz).to(device)

print("\n" + "=" * 60)
print(f"测试模型: {model_yaml}")
print("=" * 60)

# 参数量和 GFLOPs
model.info(imgsz=imgsz)

# 推理速度 FPS
warmup = 30
runs = 100

with torch.no_grad():
    for _ in range(warmup):
        _ = model(x)

if "cuda" in device:
    torch.cuda.synchronize()

start = time.time()
with torch.no_grad():
    for _ in range(runs):
        _ = model(x)

if "cuda" in device:
    torch.cuda.synchronize()

end = time.time()

avg_time = (end - start) / runs
fps = 1 / avg_time

print(f"\n平均单张推理时间: {avg_time * 1000:.2f} ms")
print(f"FPS: {fps:.2f}")
print("=" * 60)
