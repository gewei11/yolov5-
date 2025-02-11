import glob
import os.path
from PIL import Image
import torch
import cv2
import json
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from my_net import ResNet,TYPE,one_hot_size

# 定义一个训练的设备device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置参数
epoches = 25
learn_radio = 0.01
train_batch_size = 180
test_batch_size = 90
wrong_img_path = './wrong_data.json'
save_net_dict = f"weight/Resnet_{TYPE}.pt"
workers = 0

train_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),  # 执行水平翻转的概率为0.5
    transforms.RandomVerticalFlip(p=0.5),  # 执行垂直翻转的概率为0.5
    # transforms.RandomRotation((45), expand=True),
    # transforms.Resize((96, 96)),
    # transforms.Normalize(0.5,0.5),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.Resize((96, 96)),
    # transforms.Normalize(0.5,0.5),
])
class MNISTDataset(Dataset):
    def __init__(self,root=r"./arthrosis",isTrain=True, transform=train_transform):
        super().__init__()
        model_type = "train" if isTrain else "test"
        type = TYPE
        img_paths = glob.glob(os.path.join(root,model_type,type,"*","*"))
        self.dataset = []
        for path in img_paths:
            label = path.rsplit('\\',maxsplit=4)[-2]# linux系统：'/'，windows系统：'\\'
            self.dataset.append((type,label,path))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        type,label, img_path = self.dataset[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        one_hot = torch.zeros(one_hot_size)
        one_hot[int(label)-1] = 1

        return one_hot,img_tensor,img_path

class Trainer:
    def __init__(self):
        # 1. 准备数据
        train_dataset = MNISTDataset(isTrain=True, transform=train_transform)
        test_dataset = MNISTDataset(isTrain=False, transform=test_transform)
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size,num_workers = workers, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=test_batch_size,num_workers = workers, shuffle=False)
        # 初始化网络
        net = ResNet().to(device)
        try:
            net.load_state_dict(torch.load("weight/Resnet_MCP.pt"))  # 加载之前的学习成果，权重记录可以进行迁移学习，更快收敛
            print('已加载学习记录:Resnet_MCP.pt')
        except:
            print('没有学习记录')
        net.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, one_hot_size)
        )  # 迁移学习更换最后一层
        self.net = net.to(device)
        # 损失函数
        # self.loss_fn = nn.MSELoss().to(device) #均方差
        self.loss_fn = nn.CrossEntropyLoss().to(device) #交叉熵
        # 优化器 迁移学习时使用SGD
        # self.opt = torch.optim.Adam(self.net.parameters(), lr=learn_radio)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=learn_radio)
        # 指标可视化
        self.writer = SummaryWriter(f"./arthrosis/{TYPE}")

    def train(self,epoch):
        sum_loss = 0
        sum_acc = 0
        self.net.train()
        for target, input, _ in tqdm.tqdm(self.train_loader,total=len(self.train_loader), desc="训练中。。。"):
            target = target.to(device)
            input = input.to(device)
            # 前向传播得到模型的输出值
            pred_out = self.net(input)
            # 计算损失
            loss = self.loss_fn(pred_out, target)
            sum_loss += loss.item()
            # 梯度清零
            self.opt.zero_grad()
            # 反向传播求梯度
            loss.backward()
            # 更新参数
            self.opt.step()

            # 准确率
            pred_cls = torch.argmax(pred_out, dim=1)
            target_cls = torch.argmax(target, dim=1)
            sum_acc += torch.mean((pred_cls == target_cls).to(torch.float32)).item()

        print('\n')
        avg_loss = sum_loss / len(self.train_loader)
        avg_acc = sum_acc / len(self.train_loader)
        print(f"轮次:{epoch} 训练平均损失率:{avg_loss}")
        print(f"轮次:{epoch} 训练平均准确率:{avg_acc}")
        self.writer.add_scalars(f"{TYPE}_loss", {f"{TYPE}_train_avg_loss":avg_loss}, epoch)
        self.writer.add_scalars(f"{TYPE}_acc", {f"{TYPE}_train_avg_acc":avg_acc}, epoch)
        print('\n')


    def test(self,epoch):
        sum_loss = 0
        sum_acc = 0
        self.net.eval()
        # paths = []
        for target, input, _ in tqdm.tqdm(self.test_loader, total=len(self.test_loader), desc="测试中。。。"):
            target = target.to(device)
            input = input.to(device)
            # 前向传播得到模型的输出值
            pred_out = self.net(input)
            # 计算损失
            loss = self.loss_fn(pred_out, target)
            sum_loss += loss.item()

            # 准确率
            pred_cls = torch.argmax(pred_out, dim=1)
            target_cls = torch.argmax(target, dim=1)
            sum_acc += torch.mean((pred_cls == target_cls).to(torch.float32)).item()
        #     # 找出测试不准确的图片路径，并显示
        #     for idx in range(len(pred_cls)):
        #         if pred_cls[idx] != target_cls[idx]:
        #             print('\n测试不准确的图片路径:',self.test_loader.dataset[idx][2])
        #             print(f'预测结果：{pred_cls[idx]}，真实结果：{target_cls[idx]}')
        #             paths.append(self.test_loader.dataset[idx][2])
        #             img_warn = cv2.imread(self.test_loader.dataset[idx][2])
        #             cv2.imshow('img_warning',img_warn)
        #             cv2.waitKey(50)
        # # 存储图片路径
        # with open(wrong_img_path,'w') as file:
        #     if paths is not None:
        #         json.dump(paths,file)

        print('\n')
        avg_loss = sum_loss / len(self.test_loader)
        avg_acc = sum_acc / len(self.test_loader)
        self.writer.add_scalars(f"{TYPE}_loss", {f"{TYPE}_test_avg_loss": avg_loss}, epoch)
        self.writer.add_scalars(f"{TYPE}_acc", {f"{TYPE}_test_avg_acc": avg_acc}, epoch)
        print(f"轮次:{epoch}  测试平均损失率:{avg_loss}")
        print(f"轮次:{epoch}  测试平均准确率: {avg_acc}")
        print('\n')

        return avg_acc


    def run(self):
        global learn_radio
        pro_acc = 0
        for epoch in range(epoches):
            self.train(epoch)
            avg_acc = self.test(epoch)
            # 保存最优模型
            if avg_acc > pro_acc:
                pro_acc = avg_acc
                torch.save(self.net.state_dict(), save_net_dict)
                print(f'已保存{TYPE}模型')
            learn_radio *= 0.99

if __name__ == '__main__':
    tra = Trainer()
    tra.run()

