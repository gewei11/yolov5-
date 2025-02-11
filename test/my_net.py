import torch
from torch import nn
import torch.nn.functional as F

TYPE = "MCP"  # 切换欲训练模型

one_hot_list = ['DIP','DIPFirst','MCP','MCPFirst','MIP','PIP','PIPFirst','Radius','Ulna']
one_hot_dic_grade = {
  'DIP':11,
  'DIPFirst':11,
  'MCP':10,
  'MCPFirst':11,
  'MIP':12,
  'PIP':12,
  'PIPFirst':12,
  'Radius':14,
  'Ulna':12
}
one_hot_size = one_hot_dic_grade[TYPE]



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1_conv64_and_maxPool = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3,bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3, 2,1),
        )
        self.layer2_conv64 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1,bias=False),
            nn.BatchNorm2d(64),
        )

        self.layer3_conv64 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1,bias=False),
            nn.BatchNorm2d(64),
        )

        self.layer4_conv64_to_conv128 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1,bias=False),
            nn.BatchNorm2d(128),
        )
        self.layer4_res128 = nn.Conv2d(64, 128, 1, 2, 0,bias=False)

        self.layer5_conv128 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1,bias=False),
            nn.BatchNorm2d(128),
        )

        self.layer6_conv128_to_conv256 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1,bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer6_res256 = nn.Conv2d(128, 256, 1, 2, 0,bias=False)

        self.layer7_conv256 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1,bias=False),
            nn.BatchNorm2d(256),
        )

        self.layer8_conv256_to_conv512 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer8_res512 = nn.Conv2d(256, 512, 1, 2, 0, bias=False)

        self.layer9_conv512 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
        )

        self.layer10_axgPool = nn.AvgPool2d(2,1)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, one_hot_size)
        )

    def forward(self, x):
        x = self.layer1_conv64_and_maxPool(x)

        res = x
        x = self.layer2_conv64(x)
        x = self.relu(x+res)

        res = x
        x = self.layer3_conv64(x)
        x = self.relu(x + res)

        res = self.layer4_res128(x)
        x = self.layer4_conv64_to_conv128(x)
        x = self.relu(x + res)

        res = x
        x = self.layer5_conv128(x)
        x = self.relu(x + res)

        res = self.layer6_res256(x)
        x = self.layer6_conv128_to_conv256(x)
        x = self.relu(x + res)

        res = x
        x = self.layer7_conv256(x)
        x = self.relu(x + res)

        res = self.layer8_res512(x)
        x = self.layer8_conv256_to_conv512(x)
        x = self.relu(x + res)

        res = x
        x = self.layer9_conv512(x)
        x = self.relu(x + res)

        x = self.layer10_axgPool(x)

        x = x.reshape(x.shape[0],-1)
        x = self.classifier(x)

        return x

if __name__ == '__main__':
    x = torch.randn(1, 1, 96, 96)
    model = ResNet()
    print(model(x).shape)
