import torch
import torch.hub
from PIL import Image, ImageDraw, ImageFont
import common
from screen_bone import Screen_model
from pathlib import Path


yolov5 = Path(r"G:\Desktop\yolov5-骨龄检测\yolov5")  # 这将自动选择正确的路径类型
best = Path(r"G:\Desktop\yolov5-骨龄检测\test\weights\best.pt")

COLOR = ['blue', 'blue', 'green', 'yellow', 'yellow', 'pink', 'pink', 'orange', 'purple', 'purple', 'brown', 'red',
         'white']
results = {
    'MCPFirst': [],
    'MCPThird': [],
    'MCPFifth': [],
    'DIPFirst': [],
    'DIPThird': [],
    'DIPFifth': [],
    'PIPFirst': [],
    'PIPThird': [],
    'PIPFifth': [],
    'MIPThird': [],
    'MIPFifth': [],
    'Radius': [],
    'Ulna': [],
}


class BoneAgeDetect:
    def __init__(self):
        # 加载目标检测的模型
        self.yolo_model = self.load_model()
        # 加载关节检测的模型
        self.screen_model = Screen_model()

    def load_model(self):
        model = torch.hub.load(
            force_reload=True,
            repo_or_dir=yolov5,
            model="custom",
            path=best, 
            source='local')
        model.conf = 0.6
        model.eval()
        return model

    def detect(self, img_path):
        result = self.yolo_model(img_path)
        # 21个关节
        boxes = result.xyxy[0]
        # 存放的是Numpy数组，方便后面截取
        im = result.ims[0]
        return im, boxes

    def choice_boxes(self, boxes):
        if boxes.shape[0] != 21:
            print("检测的关节数量不正确")
        mcp = self.bone_filters_boxes(boxes, 6, [0, 2])
        middlePhalanx = self.bone_filters_boxes(boxes, 5, [0, 2])
        distalPhalanx = self.bone_filters_boxes(boxes, 4, [0, 2, 4])
        proximalPhalanx = self.bone_filters_boxes(boxes, 3, [0, 2, 4])
        mcpFirst = self.bone_filters_boxes(boxes, 2, [0])
        ulna = self.bone_filters_boxes(boxes, 1, [0])
        radius = self.bone_filters_boxes(boxes, 0, [0])
        return torch.cat([
            distalPhalanx,
            middlePhalanx,
            proximalPhalanx,
            mcp,
            mcpFirst,
            ulna,
            radius], dim=0)

    def bone_filters_boxes(self, boxes, cls_idx, flag):
        # 取出同类别的框
        cls_boxes = boxes[boxes[:, 5] == cls_idx]
        # 对同类别的框按照x坐标进行排序
        cls_idx = cls_boxes[:, 0].argsort()
        return cls_boxes[cls_idx][flag]

    def run(self, img_path,sex):
        im, boxes = self.detect(img_path)
        ok_boxes = self.choice_boxes(boxes)
        # 绘制
        img_pil = Image.fromarray(im)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('simsun.ttc', size=30)
        # 传递截取部分并打分
        sum_score = 0
        for idx, box in enumerate(ok_boxes):
            arthrosis_name = common.arthrosis_order[idx]
            # 9模型计算得分
            one_hot_idx = self.screen_model.get_screen_hot_idx(im, arthrosis_name, box[:4])
            score = self.screen_model.get_sex_score_list(sex, arthrosis_name)[one_hot_idx]
            results[arthrosis_name].append(one_hot_idx + 1)
            results[arthrosis_name].append(score)
            # 画框
            x1, y1, x2, y2 = box[:4]
            draw.rectangle((x1, y1, x2, y2), outline=COLOR[idx], width=3)
            draw.text(xy=(x1, y1 - 28), text=arthrosis_name, fill='red', font=font)
            # 累计总分
            sum_score += score
        # 显示
        # img_pil.show()
        # 年龄
        age = common.calcBoneAge(sum_score, sex)
        export = common.export(results, sum_score, age)

        return export,img_pil

if __name__ == '__main__':
    img_path = './image/1526.png'
    sex = input('请输入性别(boy/girl)：')
    path = input('请输入图片路径：')
    bone_analy = BoneAgeDetect()
    bone_analy.run(img_path,sex)

