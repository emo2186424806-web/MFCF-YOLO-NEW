import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.models import RTDETR

if __name__ == '__main__':
    # model = RTDETR(r'ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model = YOLO(r"ultralytics/cfg/models/11/yolo11n.yaml")
    model.train(data=r'CityPersons.yaml',
                cache=False,
                imgsz=640,
                epochs=30,
                single_cls=True,  # 是否是单类别检测
                batch=8,
                close_mosaic=10,
                workers=4,
                optimizer='SGD',
                amp=True,
                project='runs/pruning/train',
                name='Constrained Training YOLO11n',
                )