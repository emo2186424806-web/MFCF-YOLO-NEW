from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"adown+wtconv+ccfm.yaml")

    model.train(
        data=r"CityPersons.yaml",
        epochs=300,  # 100轮
        batch=8,
        imgsz=640,
        workers=4,
        cache=False,
    )
