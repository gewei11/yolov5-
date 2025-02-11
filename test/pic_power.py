import cv2
import os
import glob
from PIL import Image
import warnings
import time
from torchvision import transforms
warnings.filterwarnings('error')
pic_transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),  # 执行水平翻转的概率为0.5
    transforms.RandomVerticalFlip(p=0.5),  # 执行垂直翻转的概率为0.5
    transforms.RandomRotation((30), expand=True),
    # transforms.Resize((96, 96), antialias=True),
    # transforms.Normalize(0.5,0.5),
])
size_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96), antialias=True)
])
base_path = "img/test" #增强train   预处理 train、test
Max_num = 1000 #更改此处
Type = "DIP" #更改此处
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

for num in range(1,one_hot_dic_grade[Type]+1):
    org_img_paths = glob.glob(os.path.join(base_path,Type,str(num),"*"))

    # 均衡控制
    power_num = Max_num - len(org_img_paths)
    if power_num < 0 :
        print(f"{Type}/{num} 超过上限")
    while power_num > 0:
        for path in org_img_paths:
            try:
                # png转jpg
                # image_name = str(int(time.time() * 1000000)) + '.jpg'
                # targe_path = path.rsplit('/', maxsplit=1)[0]
                # png_image = Image.open(path)
                # png_image.save(targe_path + '/' + image_name, format="jpeg")
                # os.remove(path)

                # 数据增强
                png_image = Image.open(path)
                targe_path = path.rsplit('\\', maxsplit=1)[0]
                pic = pic_transform(png_image)
                image_name = Type +'_' + str(int(time.time() * 1000000)) + '.png'
                pic.save(targe_path + '/' + image_name, format="png")
                print(f'增强成功{path}')
                power_num -= 1
                if power_num == 0:
                    print(f"{Type}/{num}  已达增强上限数量！！！！！！！！！！！！！！！！！！！！")
                    break
                    
                # 数据检查
                # img = cv2.imread(path)
                # # 检查是否能灰度
                # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # print(f"已检查文件: {path}")
                # # 重新保存，部分有问题图片[1,192,192]>>[3,96,96]
                # img_resize = cv2.resize(img, (96, 96))
                # cv2.imwrite(path, img_resize, [cv2.IMWRITE_JPEG_QUALITY, 90])

            except Exception as e:
                # 打印异常信息
                print("发生异常：", str(e))
                # 删除异常文件
                os.remove(path)
                print(f"已删除异常文件: {path}！！！！！！！！！！！！！！！！！！！！！！！！！！！")

'''
# 全部增强之后全部预处理
org_img_paths = glob.glob(os.path.join(base_path,"*","*","*"))
for path in org_img_paths:
    png_image = Image.open(path)
    pic = size_transform(png_image)
    pic.save(path, format="png")
    print(f'尺寸修改成功{path}')
'''
