import os
import cv2
from tqdm import tqdm
def opt_img(img_path):
    img = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(tileGridSize=(3, 3))
    # 自适应直方图均衡化
    dst1 = clahe.apply(img)
    cv2.imwrite(img_path, dst1)

pic_path_folder = r'../VOCdevkit/VOC2007/JPEGImages'
if __name__ == '__main__':
    for pic_folder in tqdm(os.listdir(pic_path_folder)):
        data_path = os.path.join(pic_path_folder, pic_folder)
        # 去雾
        opt_img(data_path)
