import tkinter as tk
import cv2
import torch
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torchvision.transforms as transforms
import pandas as pd
from model7 import MobileNetV2  # 导入修改后的MobileNetV2模型类
import numpy as np
import torch.nn.functional as F
import time

# 初始化Tkinter窗口
root = tk.Tk()
root.title("人脸识别")

# 加载模型和数据
model = MobileNetV2(num_classes=38)
model.load_state_dict(torch.load("mobilenetv2_7b_2_model03.pth", map_location=torch.device('cuda')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 打开摄像头
video_source = 0
video_capture = cv2.VideoCapture(video_source)

# 加载人员数据
students_data = pd.read_excel("students.xlsx")

# 创建左侧Canvas用于显示实时画面
video_canvas = tk.Canvas(root, width=480, height=480)
video_canvas.pack(side=tk.LEFT)
tb = tk.Canvas(root, width=480, height=480)
tb.pack(side=tk.RIGHT)
# 创建右侧上半部分Canvas用于显示预测人脸图像
face_canvas = tk.Canvas(tb, width=300, height=240, bg="white")
face_canvas.pack(side=tk.TOP, padx=20, pady=10)

# 创建右侧下半部分Label用于显示预测结果
label_text = tk.StringVar()
result_label = tk.Label(tb, textvariable=label_text, font=('Helvetica', 14), fg='blue')
result_label.pack(side=tk.BOTTOM, pady=40)

# 定义变量用于跟踪上一次更新的时间
last_update_time = time.time()
errortime = time.time()
current_face_image = None
current_label = ""
keydoor = 0
door = 0


# 更新画面
def update():
    global last_update_time, current_face_image, current_label, keydoor, errortime, door
    ret, frame = video_capture.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_image = image.crop((x, y, x + w, y + h))
            face_image_tensor = transform(face_image).unsqueeze(0)
            with torch.no_grad():
                output = model(face_image_tensor)
            _max, predicted = torch.max(output, 1)
            probability = F.softmax(output, dim=1)
            class_index = predicted.item()
            if (class_index < len(students_data)) and (_max.item() > 3.0):
                student_info = students_data.iloc[class_index]
                name = student_info['姓名']
                student_id = student_info['学号']
                current_label = f"姓名：{name}\n学号：{student_id}"
                current_face_image = face_image
                # keydoor = 1
                if (time.time() - last_update_time) >= 5 or current_label == "":
                    last_update_time = time.time()
                    face_photo = ImageTk.PhotoImage(image=current_face_image)
                    face_canvas.create_image(160, 120, image=face_photo, anchor=tk.CENTER)
                    face_canvas.photo = face_photo
                    label_text.set(current_label)
                    door = 1
                    keydoor=1
            else:
                current_label = "查无此人"
                current_face_image = face_image
                # keydoor = 1
                if (time.time() - last_update_time) >= 5:
                    door = 0
                keydoor=1

        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        video_canvas.photo = photo
        print("1", current_label)
        if current_label == "":
            label_text.set("已初始化，请刷脸")
            door = 0

        if ((time.time() - last_update_time) >= 3) and (
                (time.time() - last_update_time) <= 5) and keydoor and current_label != "" and door:
            label_text.set("有人通行，请勿尾随")
            current_label = "禁止通行"
            face_canvas.delete("all")  # 删除之前的人脸图片
        else:
            if (time.time() - last_update_time) >= 5 and keydoor:
                if door == 0:
                    if current_label == "查无此人":
                        face_photo = ImageTk.PhotoImage(image=current_face_image)
                        face_canvas.create_image(160, 120, image=face_photo, anchor=tk.CENTER)
                        face_canvas.photo = face_photo
                        label_text.set(current_label)
                        door = 0
                        if (time.time() - errortime) >= 3:
                            errortime = time.time()

                    print("1111111111111111111111111111111111111111111111")
                if (current_label != "查无此人") or ((time.time() - errortime) >= 2):
                    label_text.set("请刷脸")
                    face_canvas.delete("all")  # 删除之前的人脸图片
                    current_label = "禁止通行"
                    door = 0

        print("2", current_label)
        print(time.time())
        print(last_update_time)
        print(errortime)
    root.after(10, update)  # 持续更新画面


# 人脸检测函数
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


# 开始更新画面
update()

# 运行Tkinter事件循环
root.mainloop()
