import torch


def load_model():
    """
    repo_or_dir:指向 YOLOv5 模型所在的本地目录
    model:用户定义的模型配置
    path:模型权重文件的路径
    source:使用'local’表示从本地目录加载模型
    """
    model = torch.hub.load(repo_or_dir=r"/root/2C307工作室/葛伟/yolo/yolov5/骨龄检测/yolov5",
                    model="custom",
                    path="/root/2C307工作室/葛伟/yolo/yolov5/骨龄检测/test/weight/best.pt",
                    source='local')
    model.eval()
    model.conf =0.6
    return model
model = load_model()
result = model("./image/1526.png")
print(result)