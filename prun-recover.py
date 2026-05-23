# 经典的错误，标准的零分
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # model = RTDETR(r'ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model = YOLO(r"runs/pruning/train/Constrained Training YOLO11n/weights/last_prune.pt")
    model.train(
        data=r"CityPersons.yaml",
        cache=False,
        imgsz=640,
        epochs=30,
        single_cls=True,  # 是否是单类别检测
        batch=8,
        close_mosaic=10,
        workers=4,
        optimizer="SGD",
        amp=True,
        project="runs/pruning/re-train",
        name="train",
    )
