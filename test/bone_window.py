from tkinter import *
from PIL import Image, ImageTk
import analyse_bone_age
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import sys
sys.path.insert(0, 'G:\Desktop\yolov5-骨龄检测\yolov5')

class Window_bone():
    def __init__(self):
        self.root = Tk()
        self.img_Label = Label(self.root)
        self.img_outLabel = Label(self.root)
        self.txt = Text(self.root)
        self.detect = analyse_bone_age.BoneAgeDetect()

    def bone_start(self,sex,path):
        self.txt.delete(1.0, END)  # 清除文本
        img = Image.open(path)
        img = img.resize((330, 330), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.img_Label.config(image=photo)
        self.img_Label.image = photo

        export,img_out = self.detect.run(path,sex)

        img_out = img_out.resize((330, 330), Image.LANCZOS)
        photo_out = ImageTk.PhotoImage(img_out)
        self.img_outLabel.config(image=photo_out)
        self.img_outLabel.image = photo_out

        self.txt.insert(END, export)  # 追加显示运算结果export

    def run(self):
        # 窗口
        self.root.title('骨龄检测')
        self.root.geometry('1000x800') # 这里的乘号不是 * ，而是小写英文字母 x
        # 标题
        lb_top = Label(self.root, text='骨龄检测程序',
                   bg='#d3fbfb',
                   fg='red',
                   font=('华文新魏', 32),
                   width=20,
                   height=2,
                   relief=SUNKEN)
        lb_top.pack()
        lb_sex = Label(self.root, text='请输入性别：')
        lb_path = Label(self.root, text='请输入骨骼图片路径：')
        lb_sex.place(relx=0.01, rely=0.25, relwidth=0.09, relheight=0.05)
        lb_path.place(relx=0.29, rely=0.25, relwidth=0.16, relheight=0.05)
        inp_sex = Entry(self.root)
        inp_sex.place(relx=0.1, rely=0.25, relwidth=0.18, relheight=0.05)
        inp_path = Entry(self.root)
        inp_path.place(relx=0.44, rely=0.25, relwidth=0.3, relheight=0.05)

        # 结果文本
        self.txt.place(rely=0.8, relwidth=1, relheight=0.3)

        # 按钮
        btn1 = Button(self.root, text='开始检测', command=lambda: self.bone_start(inp_sex.get(), inp_path.get()))
        btn1.place(relx=0.76, rely=0.2, relwidth=0.2, relheight=0.1)

        # 图像
        self.img_Label.place(relx=0.05, rely=0.3, relwidth=0.45, relheight=0.5)
        self.img_outLabel.place(relx=0.55, rely=0.3, relwidth=0.45, relheight=0.5)

        self.root.mainloop()

if __name__ == '__main__':
    win = Window_bone()
    win.run()
