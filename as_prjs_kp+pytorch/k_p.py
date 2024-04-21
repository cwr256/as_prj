from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import pandas as pd


# 函数：加载图像并将其调整为指定大小
def load_image(image_path, target_size=(100, 100)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式加载图像
    img_resized = cv2.resize(img, target_size)  # 调整图像大小
    return img_resized


# 函数：加载数据集
def load_dataset(data_dir):
    X = []  # 用于存储图像像素数据
    y = []  # 用于存储标签
    label_map = {}  # 用于将文件夹名映射到整数标签
    label_index = 0  # 标签索引
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name not in label_map:
                label_map[dir_name] = label_index
                label_index += 1
            dir_path = os.path.join(root, dir_name)
            for file_name in os.listdir(dir_path):
                image_path = os.path.join(dir_path, file_name)
                image = load_image(image_path)  # 加载并调整图像大小
                X.append(image.flatten())
                y.append(label_map[dir_name])  # 使用映射后的整数标签
    return np.array(X), np.array(y)


# 函数：用PCA进行特征提取
def apply_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


# 函数：训练K近邻分类器
def train_knn(X_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


# 函数：使用K近邻分类器进行预测
def predict_knn(knn, X_test):
    return knn.predict(X_test)


# 函数：使用Tkinter显示图像
def show_image(image, label):
    root = tk.Tk()
    root.title("Face Recognition")
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    panel = tk.Label(root, image=imgtk)
    panel.image = imgtk
    panel.pack(side="top", padx=10, pady=10)
    label = tk.Label(root, text=label, font=("Helvetica", 16))
    label.pack(side="bottom", padx=10, pady=10)
    root.mainloop()


# 加载数据集
data_dir = "data02"
X, y = load_dataset(data_dir)

# 检查数据集是否为空
if len(X) == 0 or len(y) == 0:
    print("数据集为空，请检查数据路径")
    exit()

# 应用PCA进行特征提取
pca, X_pca = apply_pca(X, n_components=30)  # 设置合适的主成分数量

# 检查主成分数量是否合理
if pca.n_components_ < 2:
    print("数据集维度过低，无法应用PCA")
    exit()

# 训练K近邻分类器
knn = train_knn(X_pca, y)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 创建Tkinter窗口
root = tk.Tk()
root.title("人脸识别")

# 创建左侧Canvas用于显示实时画面
bt = tk.Canvas(root, width=240, height=480)
bt.pack(side=tk.LEFT)
tb = tk.Canvas(root, width=480, height=480)
tb.pack(side=tk.RIGHT)

panel = tk.Label(tb)
panel.pack(side="top", padx=10, pady=10)

key =1
def exit_app():
    global key
    key = 0

exit_button = tk.Button(bt, text="Exit", command=exit_app)
exit_button.pack(pady=20)

# 加载人员数据
students_data = pd.read_excel("students.xlsx")

# 函数：使用K近邻分类器进行预测，并返回置信度
def predict_with_confidence(knn, X_test):
    distances, indices = knn.kneighbors(X_test, n_neighbors=3)  # 选择最近的3个邻居
    confidence = 1 / (1 + distances.sum(axis=1))  # 计算置信度，距离越小，置信度越高
    predictions = knn.predict(X_test)  # 预测标签
    # print(predictions)
    # print(confidence)
    return predictions, confidence


# 实时识别
while (True and key):
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 对每个检测到的人脸进行识别
    for (x, y, w, h) in faces:
        # 裁剪人脸区域并调整尺寸
        face = cv2.resize(gray[y:y + h, x:x + w], (100, 100))

        # 对裁剪后的人脸图像进行特征提取
        face_pca = pca.transform(face.flatten().reshape(1, -1))
        print(face_pca)
        # 使用K近邻分类器进行预测，并返回置信度
        predicted_label, confidence = predict_with_confidence(knn, face_pca)

        # 判断置信度是否高于阈值
        threshold = 0.000023  # 设置阈值
        if confidence[0] > threshold:
            # 在图像上标注识别结果
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            student_info = students_data.iloc[predicted_label[0]]
            name = student_info['姓名']
            student_id = student_info['学号']
            current_label = f"姓名：{name}\n学号：{student_id}"
            cv2.putText(frame, f"Label: {student_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # 识别为陌生人
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # # 在图像上标注识别结果
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # student_info = students_data.iloc[predicted_label]
        # name = student_info['姓名']
        # student_id = student_info['学号']
        # current_label = f"姓名：{name}\n学号：{student_id}"
        # cv2.putText(frame, f"Label: {student_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 在Tkinter窗口中显示图像
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.configure(image=imgtk)
    root.update()

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
