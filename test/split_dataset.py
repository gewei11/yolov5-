"""
1.将图片和标注数据按比例切分为 训练集和测试集
2.原图片的目录名是:  JPEGImages
3.对应的txt标签是之前转换的labels
4.训练集、测试集、验证集 路径和VOC2007路径保持一致
"""
import shutil
import random
import os

BASE_PATH = r"G:\Desktop\yolov5-骨龄检测\VOCdevkit"
image_suffix = ".png"
# 数据集路径
image_original_path = os.path.join(BASE_PATH, "VOC2007/JPEGImages/")
label_original_path = os.path.join(BASE_PATH, "VOC2007/labels/")

# 训练集路径
train_image_path = os.path.join(BASE_PATH, "train/images/")
train_label_path = os.path.join(BASE_PATH, "train/labels/")

# 验证集路径
val_image_path = os.path.join(BASE_PATH, "val/images/")
val_label_path = os.path.join(BASE_PATH, "val/labels/")
# 测试集路径
test_image_path = os.path.join(BASE_PATH, "test/images/")
test_label_path = os.path.join(BASE_PATH, "test/labels/")

# 数据集划分比例，训练集75%，验证集15%，测试集15%，按需修改
train_percent = 0.7
val_percent = 0.15
test_percent = 0.15


# 检查文件夹是否存在
def mkdir():
	if not os.path.exists(train_image_path) and train_percent > 0:
		os.makedirs(train_image_path)
	if not os.path.exists(train_label_path) and train_percent > 0:
		os.makedirs(train_label_path)

	if not os.path.exists(val_image_path) and val_percent > 0:
		os.makedirs(val_image_path)
	if not os.path.exists(val_label_path) and val_percent > 0:
		os.makedirs(val_label_path)

	if not os.path.exists(test_image_path) and test_percent > 0:
		os.makedirs(test_image_path)
	if not os.path.exists(test_label_path) and test_percent > 0:
		os.makedirs(test_label_path)


def main():
	mkdir()
	total_txt = os.listdir(label_original_path)
	num_txt = len(total_txt)
	list_all_txt = range(num_txt)  # 范围 range(0, num)
	# 0.75 * num_txt
	num_train = int(num_txt * train_percent)
	# 0.15 * num_txt
	# 如果测试集test_percent==0, 直接使用总数量减去训练集的数量
	if test_percent == 0:
		num_val = num_txt - num_train
	else:
		num_val = int(num_txt * val_percent)
	num_test = num_txt - num_train - num_val

	train = random.sample(list_all_txt, num_train)
	# 在全部数据集中取出train
	val_test = [i for i in list_all_txt if not i in train]
	# 再从val_test取出num_val个元素，val_test剩下的元素就是test
	val = random.sample(val_test, num_val)
	print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))
	for i in list_all_txt:
		name = total_txt[i][:-4]

		srcImage = image_original_path + name + '.png'
		srcLabel = label_original_path + name + '.txt'

		if i in train:
			dst_train_Image = train_image_path + name + '.png'
			dst_train_Label = train_label_path + name + '.txt'
			shutil.copyfile(srcImage, dst_train_Image)
			shutil.copyfile(srcLabel, dst_train_Label)
		elif i in val:
			dst_val_Image = val_image_path + name + '.png'
			dst_val_Label = val_label_path + name + '.txt'
			shutil.copyfile(srcImage, dst_val_Image)
			shutil.copyfile(srcLabel, dst_val_Label)
		else:
			dst_test_Image = test_image_path + name + '.png'
			dst_test_Label = test_label_path + name + '.txt'
			shutil.copyfile(srcImage, dst_test_Image)
			shutil.copyfile(srcLabel, dst_test_Label)


if __name__ == '__main__':
	main()

