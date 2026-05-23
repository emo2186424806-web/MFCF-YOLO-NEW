from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"adown+wtconv+ccfm.yaml")

    model.train(
        data=r"WiderPerson.yaml",
        epochs=300,                                 #100轮
        batch=4,
        imgsz=640,
        workers=2,
        cache=False,
    )