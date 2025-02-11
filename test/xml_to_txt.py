import xml.etree.ElementTree as ET
import os

from PIL import Image


def convert(size, box):
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return (x, y, w, h)

def convert_format(xml_files_path, save_txt_files_path, classes):
    if not os.path.exists(save_txt_files_path):
        os.makedirs(save_txt_files_path)
    xml_files = os.listdir(xml_files_path)
    # print(xml_files)
    for xml_name in xml_files:
        # print(xml_name)
        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        out_txt_f = open(out_txt_path, 'w', encoding='utf-8')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        if size is None:
            w, h = get_imgwh(xml_file)
        else:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if w == 0 or h == 0:
                w, h = get_imgwh(xml_file)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            # print(w, h, b)
            try:
                bb = convert((w, h), b)
            except:
                print(f"convert转换异常: {xml_file}")
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def get_imgwh(xml_file):
    img_path = xml_file.replace("Annotations", "JPEGImages").replace(".xml", image_suffix)
    img_pil = Image.open(img_path)
    w, h = img_pil.size
    return w, h

if __name__ == "__main__":
    """
    说明:
    BASE_PATH: 数据集标签目录的上一级路径
    注意数据集里面的标签文件: 目录名是 Annotations
    保存为txt的标签目录名是:  labels
    """
    BASE_PATH = r"./VOCdevkit/VOC2007"
    image_suffix = ".png"
    # 需要转换的类别，需要一一对应
    classes = [
        'Radius',
        'Ulna',
        'MCPFirst',
        'ProximalPhalanx',
        'DistalPhalanx',
        'MiddlePhalanx',
        'MCP'
    ]
    # 2、voc格式的xml标签文件路径
    xml_files = os.path.join(BASE_PATH, "Annotations")
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files = os.path.join(BASE_PATH, "labels")
    convert_format(xml_files, save_txt_files, classes)
