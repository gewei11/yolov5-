import cv2
import numpy
import torch
from torch import nn
import common
from my_net import ResNet,one_hot_dic_grade
from torchvision import transforms

img_transforms = transforms.Compose([
    # 将 H W C--> C H W
    # [0 255] -->[0, 1]
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(96, 96), antialias=True)
])
class Screen_model:
    def __init__(self):
        self.DIPmodel = self.load_model('DIP')
        self.DIPFirstmodel = self.load_model('DIPFirst')
        self.MCPmodel = self.load_model('MCP')
        self.MCPFirstmodel = self.load_model('MCPFirst')
        self.MIPmodel = self.load_model('MIP')
        self.PIPmodel = self.load_model('PIP')
        self.PIPFirstmodel = self.load_model('PIPFirst')
        self.Radiusmodel = self.load_model('Radius')
        self.Ulnamodel = self.load_model('Ulna')
    def load_model(self,TYPE):
        # 加载网络
        model = ResNet()
        model.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, one_hot_dic_grade[TYPE])
        )
        model.load_state_dict(torch.load(f'weights/ResNet_{TYPE}.pt'))
        # 开启验证
        model.eval()
        return model
    def choose_modle_get_hot(self, img, cls):
        if cls == 'DIPFifth' or cls == 'DIPThird':
            one_hot = self.DIPmodel(img)
        elif cls == 'DIPFirst':
            one_hot = self.DIPFirstmodel(img)
        elif cls == 'MCPFifth' or cls == 'MCPThird':
            one_hot = self.MCPmodel(img)
        elif cls == 'MCPFirst':
            one_hot = self.MCPFirstmodel(img)
        elif cls == 'MIPFifth' or cls == 'MIPThird':
            one_hot = self.MIPmodel(img)
        elif cls == 'PIPFifth' or cls == 'PIPThird':
            one_hot = self.PIPmodel(img)
        elif cls == 'PIPFirst':
            one_hot = self.PIPFirstmodel(img)
        elif cls == 'Radius':
            one_hot = self.Radiusmodel(img)
        elif cls == 'Ulna':
            one_hot = self.Ulnamodel(img)
        else:
            one_hot = 0
        return one_hot

    def get_screen_hot_idx(self,img,cls,box):
        region = img[int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]
        img = img_transforms(region)
        img = img.unsqueeze(dim=0)
        one_hot = self.choose_modle_get_hot(img,cls)
        return one_hot.argmax()

    def get_sex_score_list(self,sex,cls):
        grade_list = common.SCORE[sex][cls]
        return grade_list

if __name__ == '__main__':
    box = (423,1002,562,1233)
    img = cv2.imread('./image/1526.png')

    screen_bone = Screen_model()
    one_hot_idx = screen_bone.get_screen_hot_idx(img, 'MCPFirst', box)
    score = screen_bone.get_sex_score_list('boy', 'MCPFirst')[one_hot_idx]
    print(score)
