from ultralytics import YOLO
import os
from utils.yolo.attention import add_attention

root = os.getcwd()

# ---------------- 配置数据集 ----------------
yaml_path = os.path.join(r"WiderPerson.yaml")

# ---------------- Step1: 改进模型正常训练得到 teacher-ready 权重 ----------------
pretrained_model_path = os.path.join(
    r"runs/detect/widerperson+adown+wtconv+ccfm/weights/best.pt"
)

# ---------------- Step2: 稀疏训练 ----------------
step1_train_model_path = os.path.join(r"runs/new-pruning/widerperson+step1/weights/best.pt")
step2_constraint_train_model_path = os.path.join(r"runs/new-pruning/widerperson+step2")

# ---------------- Step3: 结构化剪枝 ----------------
pruning_rate = 0.1
step3_prune_before_model_path = os.path.join(step2_constraint_train_model_path, "weights/last.pt")
step3_prune_after_model_path = os.path.join(step2_constraint_train_model_path, "weights/prune0.2.pt")

# ---------------- Step4: 剪枝后微调 ----------------
step4_finetune_project = os.path.join(r"runs/new-pruning")
step4_finetune_name = "widerperson+step4"

# ---------------- Step5: 蒸馏 ----------------
# teacher: 未剪枝改进模型
step5_teacher_model_path = os.path.join(
    r"runs/detect/widerperson+adown+wtconv+ccfm/weights/best.pt"
)

# student: 剪枝微调后的改进模型
step5_student_model_path = os.path.join(
    rf"{step4_finetune_project}\{step4_finetune_name}\weights\best.pt"
)

step5_output_project = os.path.join(r"runs/new-pruning/widerperson+step5")
step5_output_name = "student"

# 适配 adown+wtconv+ccfm.yaml 的关键蒸馏层
distill_layers = [6, 8, 15, 20, 23, 26]


def step1_train():
    model = YOLO(pretrained_model_path)
    model.train(
        data=yaml_path,
        imgsz=640,
        epochs=50,
        batch=2,
        workers=4,
        project="runs/new-pruning",
        name="widerperson+step1",
    )


def step2_constraint_train():
    model = YOLO(step1_train_model_path)
    model.train(
        data=yaml_path,
        imgsz=640,
        epochs=50,
        batch=4,
        amp=False,
        workers=4,
        project="runs/new-pruning",
        name="widerperson+step2",
    )


def step3_pruning():
    from utils.yolo.det_pruning_adown_wtconv_ccfm import do_pruning
    do_pruning(
        modelpath=step3_prune_before_model_path,
        savepath=step3_prune_after_model_path,
        pruning_rate=pruning_rate,
        data_yaml=yaml_path,
    )


def step4_finetune():
    model = YOLO(step3_prune_after_model_path)
    for param in model.model.parameters():
        param.requires_grad = True

    model.train(
        data=yaml_path,
        imgsz=640,
        epochs=200,
        batch=4,
        workers=4,
        project=step4_finetune_project,
        name=step4_finetune_name,
    )


def step5_distillation():
    model_t = YOLO(step5_teacher_model_path)
    model_s = YOLO(step5_student_model_path)

    # 保留你原来的注意力增强方式
    model_s = add_attention(model_s)

    model_s.train(
        data=yaml_path,
        model_t=model_t.model,
        loss_type="cwd",
        layers=distill_layers,
        amp=False,
        imgsz=640,
        epochs=300,
        batch=4,
        workers=4,
        lr0=0.001,
        project=step5_output_project,
        name=step5_output_name,
    )


if __name__ == "__main__":
    # step1_train()
    #  step2_constraint_train()
    # step3_pruning()
    # step4_finetune()
     step5_distillation()
