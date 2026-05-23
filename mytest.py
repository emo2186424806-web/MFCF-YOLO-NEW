from ultralytics import YOLO

if __name__ == "__main__":
    # 加载模型
    model = YOLO(r"runs/new-pruning/widerperson+step5/student/weights/best.pt")

    # ===================== 打印模型信息 =====================
    print("\n" + "=" * 65)
    print("            模型结构信息")
    print("=" * 65)
    model.info()

    metrics = model.val(data="WiderPerson.yaml", split="test", imgsz=640, batch=4, workers=2, verbose=True)

    # ===================== 输出指标（可直接写论文） =====================
    print("\n" + "=" * 65)
    print("            模型指标汇总（可直接写论文）")
    print("=" * 65)
    print(f"🎯 精确率 (Precision):   {metrics.box.mp:.2%}")
    print(f"🎯 召回率 (Recall):      {metrics.box.mr:.2%}")
    print(f"🎯 mAP@0.5:              {metrics.box.map50:.2%}")
    print(f"🎯 mAP@0.5:0.95:         {metrics.box.map:.2%}")
    print("=" * 65)
