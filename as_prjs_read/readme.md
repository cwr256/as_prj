# 一、KNN+PCA

## 1.原理

1. **K近邻（KNN）**：
   - KNN是一种基于实例的学习算法，它通过比较一个样本与训练集中的所有样本的距离来进行分类。
   - 当给定一个新的样本时，KNN会找到与该样本最近的K个训练样本。
   - 然后，KNN会将新样本分配给这K个训练样本中最常见的类别（对于分类问题）或者计算它们的平均值（对于回归问题）。
   - 在人脸识别中，KNN可以用来根据样本之间的相似度来识别未知的人脸。
2. **主成分分析（PCA）**：
   - PCA是一种无监督学习技术，用于降低数据的维度。
   - 它通过线性变换将原始特征映射到一个新的特征空间，新的特征空间中的特征称为主成分。
   - PCA的目标是找到能够最大化数据方差的方向，将数据投影到这些方向上，从而使得投影后的数据具有最大的差异性。
   - 在人脸识别中，PCA可以用来提取人脸图像的主要特征，从而降低数据的维度并且保留重要的信息。

## 2.关键代码

导入需使用的库

```python
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import pandas as pd
```

图像预处理与数据集加载

```
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
```

PCA特征提取

```
# 函数：用PCA进行特征提取
def apply_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca
```

K近邻分类器

```
# 函数：训练K近邻分类器
def train_knn(X_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn
# 函数：使用K近邻分类器进行预测
def predict_knn(knn, X_test):
    return knn.predict(X_test)
```

显示图像

```
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
```

实际运行

```
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
```

## 3.运行结果

识别结果：
<img src="F:\as_prjs_all\1.png" style="zoom: 45%;" /><img src="F:\as_prjs_all\2.png" style="zoom:45%;" />

# 二、卷积神经网络（使用MobileNetV2）

## 1.原理

### 1.1整体原理

将数据集八二分，80%输入网络训练，20%用以测试，输出权重文件，输入图片调用权重文件配合对应的网络完成推理。推理完成后根据推理出的embedding推出index，根据index索引students.xlsx，然后输出学号、姓名。

### 1.2MobileNetV2网络原理

MobileNetV2是一种轻量级的卷积神经网络结构，适用于移动端和嵌入式设备等资源受限的场景。原官方版本的网络参数较大，为适应我的设备以及数据集的情况，做出了一些调整，将Relu6函数替换为PRelu函数，将最后一层的全局平均池化层改换为全局深度可分离卷积，增加Dropout层以及L1正则化等。

MobileNetV2基于Inverted Residuals with Linear Bottlenecks的结构，包括了深度可分离卷积、残差连接和线性瓶颈。通过使用深度可分离卷积来减少参数数量，从而在保持模型轻量级的同时提高了模型的性能。主要由两部分组成：特征提取部分和分类器部分。特征提取部分采用了一系列的Inverted Residual模块，用于提取输入图像的特征；分类器部分则负责将提取的特征映射到类别空间。

**优势**：

- **轻量级结构**：MobileNetV2采用了一系列轻量级设计，包括深度可分离卷积、线性瓶颈等，从而大大减少了模型的参数数量，使得模型适用于移动设备和嵌入式设备等资源受限的场景。
- **高性能**：尽管是轻量级结构，MobileNetV2在保持模型轻量化的同时，也保持了较高的性能，可以在一定程度上满足对准确率和速度的要求。
- **可扩展性**：MobileNetV2模型可以通过调整宽度因子（width multiplier）和分辨率因子（resolution multiplier）来灵活地平衡模型的性能和计算开销，以适应不同的应用场景和设备要求。比如使用0.5的width multiplier可以减小计算时间，但是在推理时需要注意参数不能缺省。
- **高效的特征学习**：Inverted Residuals结构使得模型在进行特征学习时更加高效，能够更好地利用网络深度和宽度来提高特征的表达能力。

## 2.流程图

图像处理

网络的完整流程图如下

<img src="F:\as_prjs_all\3.png" style="zoom:40%;" />

Bottlenet流程图如下

<img src="F:\as_prjs_all\4.png" style="zoom:45%;" />

## 3.关键代码

网络代码：

```python
import ...
class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNPReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes)
        )
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNPReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNPReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class GDConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=1):
        super(GDConv, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride=1, padding=0, bias=False)
    def forward(self, x):
        out = self.conv(x)
        out = F.avg_pool2d(out, out.size()[2:])
        return out
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [2, 24, 2, 2],
                [2, 32, 2, 2],
                [2, 64, 3, 2],
                [1, 96, 2, 1],
                [1, 160, 2, 2],
                [1, 320, 1, 1],
            ]
        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNPReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNPReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # 添加 Dropout 层
            GDConv(self.last_channel, num_classes),  # 使用 GDConv 替换最后一层的分离卷积
            nn.BatchNorm2d(num_classes),
            nn.PReLU(num_classes),
        )
        # L1 正则化
        self.l1_regularization = nn.L1Loss(size_average=False)
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x, target=None):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        # 计算 L1 正则化项
        if target is not None:
            l1_loss = self.l1_regularization(x, target)
            return x, l1_loss
        else:
            return x
```

训练代码，在训练中我还使用了衰弱学习率的方法来防止过拟合

```python
import ...
# 将模型和数据加载到 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 准备数据集
dataset = ImageFolder('data03', transform=transform)
# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 创建模型实例并将其移动到GPU上
model = MobileNetV2(num_classes=38).to(device)
model.train()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 设置学习率衰减策略
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 模型训练
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=30):
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(num_epochs):
        scheduler.step()  # 更新学习率
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_loss = running_train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        # 在测试集上验证模型性能
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        test_loss = running_test_loss / len(test_loader.dataset)
        test_accuracy = correct_test / total_test
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    # 绘制训练过程中的损失和正确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss_list, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracy_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()
# 训练模型
train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
# 保存模型
torch.save(model.state_dict(), 'mobilenetv2_7b_3_model02.pth')
```

推理（含tkinter的gui界面），设置了5s的通行时间，2s的防尾随报警。

```python
import ...
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
            else:
                current_label = "查无此人"
                current_face_image = face_image
                # keydoor = 1
                if (time.time() - last_update_time) >= 5:
                    door = 0
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
```

## 4.运行效果

### 4.1训练效果

原MobileNetV2网络训练效果如下

<img src="F:\as_prjs_all\v1-50.png" style="zoom:66%;" />

修改后的MobileNetV2网络训练结果如下

<img src="F:\as_prjs_all\5.png" style="zoom:66%;" />

### 4.2运行效果

实际运行结果如下

<img src="F:\as_prjs_all\6.png" style="zoom:35%;" />

<img src="F:\as_prjs_all\7.png" alt="7" style="zoom:35%;" />

<img src="F:\as_prjs_all\11.png" style="zoom:35%;" />

# 三、mindspore框架下训练

## 1.原理

在该部分，我们使用的是Mobilefacenet网络。网络结构的主要组成部分如下：

1. **ConvBlock**：卷积块，包括了卷积、批归一化和PReLU激活函数。
2. **LinearBlock**：线性块，类似ConvBlock，但没有PReLU激活函数。
3. **DepthWise**：深度可分离卷积块，由一组卷积操作组成，其中包括了两个ConvBlock和一个LinearBlock，用于实现深度可分离卷积的效果。
4. **Residual**：残差块，包括了多个DepthWise块，用于构建残差连接。
5. **GDC**：全局深度特征压缩块，用于将特征图转换为全局深度特征。
6. **MobileFaceNet**：MobileFaceNet网络模型，由一系列ConvBlock、Residual和DepthWise块组成，用于提取人脸图像的特征。
7. **PartialFC**：带有部分FC层的ArcFace模型，用于人脸识别的特征匹配。
8. **Network**：网络模型的组合，将MobileFaceNet和PartialFC组合在一起构成完整的人脸识别模型。

该网络的流程可以简述为：输入的人脸图像经过一系列的卷积、深度可分离卷积和残差连接等操作，逐步提取图像的特征。最后通过全局深度特征压缩块（GDC）将特征图转换为全局深度特征向量。得到的特征向量经过部分FC层进行特征匹配，输出最终的识别结果。

## 2.代码

Mobilefacenet网络代码如下

```python
import ...
__all__ = ["get_mbf", "get_mbf_large"]
class Flatten(Cell):
    """
    Flatten.
    """
    def construct(self, x):
        """
        construct.
        """
        return x.view(x.shape[0], -1)
class ConvBlock(Cell):
    """
    ConvBlock.
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0), group=1):
        super().__init__()
        self.layers = nn.SequentialCell(
            Conv2d(in_c, out_c, kernel, group=group, stride=stride, pad_mode='pad',
                   padding=padding, has_bias=False),
            BatchNorm2d(num_features=out_c),
            PReLU(channel=out_c)
        )
    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)
class LinearBlock(Cell):
    """
    LinearBlock.
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0), group=1):
        super().__init__()
        self.layers = nn.SequentialCell(
            Conv2d(in_c, out_c, kernel, group=group, stride=stride, pad_mode='pad',
                   padding=padding, has_bias=False),
            BatchNorm2d(num_features=out_c)
        )
    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)
class DepthWise(Cell):
    """
    DepthWise.
    """
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2),
                 padding=(1, 1, 1, 1), group=1):
        super().__init__()
        self.residual = residual
        self.layers = nn.SequentialCell(
            ConvBlock(in_c, out_c=group, kernel=(1, 1), padding=(0, 0, 0, 0), stride=(1, 1)),
            ConvBlock(group, group, group=group, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(group, out_c, kernel=(1, 1), padding=(0, 0, 0, 0), stride=(1, 1))
        )
    def construct(self, x):
        """
        construct.
        """
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output
class Residual(Cell):
    """
    Residual.
    """
    def __init__(self, c, num_block, group, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1)):
        super().__init__()
        cells = []
        for _ in range(num_block):
            cells.append(DepthWise(c, c, True, kernel, stride, padding, group))
        self.layers = SequentialCell(*cells)
    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)
class GDC(Cell):
    """
    GDC.
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.layers = nn.SequentialCell(
            LinearBlock(512, 512, kernel=(7, 7), stride=(1, 1), padding=(0, 0, 0, 0), group=512),
            Flatten(),
            Dense(512, embedding_size, has_bias=False),
            BatchNorm1d(embedding_size))

    def construct(self, x):
        """
        construct.
        """
        return self.layers(x)
class MobileFaceNet(Cell):
    """
    Build the mobileface model.

    Args:
        num_features (Int): The num of features. Default: 512.
        blocks (Tuple): The architecture of backbone. Default: (1, 4, 6, 2).
        scale (Int): The scale of network blocks. Default: 2.

    Examples:
        >>> net = MobileFaceNet(num_features=512, blocks=(1,4,6,2), scale=2)
    """

    def __init__(self, num_features=512, blocks=(1, 4, 6, 2), scale=2):
        super().__init__()
        self.scale = scale
        self.layers = nn.CellList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2),
                      padding=(1, 1, 1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1),
                          padding=(1, 1, 1, 1), group=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], group=128, kernel=(3, 3),
                         stride=(1, 1), padding=(1, 1, 1, 1)),
            )

        self.layers.extend(
            [
                DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2),
                          padding=(1, 1, 1, 1), group=128),
                Residual(64 * self.scale, num_block=blocks[1], group=128, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1, 1, 1)),
                DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2),
                          padding=(1, 1, 1, 1), group=256),
                Residual(128 * self.scale, num_block=blocks[2], group=256, kernel=(3, 3),
                         stride=(1, 1), padding=(1, 1, 1, 1)),
                DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2),
                          padding=(1, 1, 1, 1), group=512),
                Residual(128 * self.scale, num_block=blocks[3], group=256, kernel=(3, 3),
                         stride=(1, 1), padding=(1, 1, 1, 1)),
            ])

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1),
                                  padding=(0, 0, 0, 0))
        self.features = GDC(num_features)
        self._initialize_weights()
    def _initialize_weights(self):
        """
        initialize_weights
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                                                 cell.weight.data.shape, cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.data.shape,                                                  cell.bias.data.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer('ones', cell.gamma.data.shape))
                cell.beta.set_data(initializer('zeros', cell.beta.data.shape))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                                                 cell.weight.data.shape, cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.data.shape,
                                                   cell.bias.data.dtype))
    def construct(self, x):
        """
        construct.
        """
        for func in self.layers:
            x = func(x)
        x = self.conv_sep(x)
        x = self.features(x)
        return x
def get_mbf(num_features=512, blocks=(1, 4, 6, 2), scale=2):
    """
    Get the mobilefacenet-0.45G.
    Examples:
        >>> net = get_mbf(512)
    """
    return MobileFaceNet(num_features, blocks, scale=scale)
def get_mbf_large(num_features=512, blocks=(2, 8, 12, 4), scale=4):
    """
    Get the large mobilefacenet.

    Examples:
        >>> net = get_mbf_large(512)
    """
    return MobileFaceNet(num_features, blocks, scale=scale)
class PartialFC(nn.Cell):
    """
    Build the arcface model without loss function.

    Args:
        num_classes (Int): The num of classes.
        world_size (Int): Number of processes involved in this work.

    Examples:
        >>> net=PartialFC(num_classes=num_classes, world_size=device_num)
    """
    def __init__(self, num_classes, world_size):
        super().__init__()
        self.l2_norm = ops.L2Normalize(axis=1)
        self.weight = Parameter(initializer(
            "normal", (num_classes, 512)), name="mp_weight")
        self.sub_weight = self.weight
        self.linear = ops.MatMul(transpose_b=True).shard(
            ((1, 1), (world_size, 1)))
    def construct(self, features):
        """
        construct.
        """
        total_features = self.l2_norm(features)
        norm_weight = self.l2_norm(self.sub_weight)
        logits = self.forward(total_features, norm_weight)
        return logits
    def forward(self, total_features, norm_weight):
        """
        forward.
        """
        logits = self.linear(total_features, norm_weight)
        return logits
class Network(nn.Cell):
    """
    WithLossCell.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self._backbone = backbone
        self.fc = head
    def construct(self, data):
        """
        construct.
        """
        out = self._backbone(data)
        logits = self.fc(out)
        return logits
```

使用以下代码进行本地单张图片的推理

```python
net = get_mbf(num_features=512)
head = PartialFC(num_classes=38, world_size=1)
train_net = Network(net, head)
param_dict = load_checkpoint("arcface_mobilefacenet-24_71.ckpt")
load_param_into_net(train_net, param_dict)
img_path = "dataarcface/B21041636/B21041636.jpg"  # 此处可以更换为自己的图像路径
img = Image.open(img_path)
img = img.resize((112, 112), Image.BICUBIC)
img = np.array(img).transpose(2, 0, 1)
img = ms.Tensor(img, ms.float32)
img = ((img / 255) - 0.5) / 0.5
img = ms.Tensor(img, ms.float32)
if len(img.shape) == 4:
    pass
elif len(img.shape) == 3:
    img = img.expand_dims(axis=0)
net_out = train_net(img)
embeddings = net_out.asnumpy()
print(embeddings)
```

在华为云端使用ModelArts进行训练如下所示

<img src="F:\as_prjs_all\8.png" alt="8" style="zoom:50%;" />

保留了最后10组的权重文件，选取最好的一个权重保存即可

推理如下代码如下

```python
import ...
net = get_mbf(num_features=512)
head = PartialFC(num_classes=38, world_size=1)
train_net = Network(net, head)
param_dict = load_checkpoint("arcface_mobilefacenet_8-41_350.ckpt")
load_param_into_net(train_net, param_dict)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# 加载人员数据
students_data = pd.read_excel("students.xlsx")
# 打开摄像头
video_source = 0
video_capture = cv2.VideoCapture(video_source)
# 创建Tkinter窗口
root = tk.Tk()
root.title("人脸识别")
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
    if current_face_image is not None:
        face_photo = ImageTk.PhotoImage(image=current_face_image)
        face_canvas.create_image(160, 120, image=face_photo, anchor=tk.CENTER)
        face_canvas.photo = face_photo
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_image = image.crop((x, y, x + w, y + h))
            imge = face_image.resize((112, 112), Image.BICUBIC)
            imge = np.array(imge).transpose(2, 0, 1)
            imge = ms.Tensor(imge, ms.float32)
            imge = ((imge / 255) - 0.5) / 0.5
            imge = ms.Tensor(imge, ms.float32)
            if len(imge.shape) == 4:
                pass
            elif len(imge.shape) == 3:
                imge = imge.expand_dims(axis=0)
            net_out = train_net(imge)
            embeddings = net_out.asnumpy()
            predicted = np.argmax(embeddings, axis=1)
            class_index = predicted.item()
            _max = np.max(embeddings)
            print(_max)
            if (class_index < len(students_data)) and _max >= 0.63:
                student_info = students_data.iloc[class_index]
                name = student_info['姓名']
                student_id = student_info['学号']
                current_label = f"姓名：{name}\n学号：{student_id}"
                current_face_image = face_image
                if (time.time() - last_update_time) >= 5 or current_label == "":
                    last_update_time = time.time()
                    face_photo = ImageTk.PhotoImage(image=current_face_image)
                    face_canvas.create_image(160, 120, image=face_photo, anchor=tk.CENTER)
                    face_canvas.photo = face_photo
                    label_text.set(current_label)
                    print(embeddings)
                    door = 1
                keydoor = 1
            else:
                current_label = "查无此人"
                current_face_image = face_image
                keydoor = 0
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        video_canvas.photo = photo
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
                if (current_label != "查无此人") or ((time.time() - errortime) >= 2):
                    label_text.set("请刷脸")
                    face_canvas.delete("all")  # 删除之前的人脸图片
                    current_label = "禁止通行"
                    door = 0
        print("2", current_label)
        print('time', time.time())
        print("last_update_time", last_update_time)
        print("eroor", errortime)
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
```

## 3.运行效果

运行效果如下：

<img src="F:\as_prjs_all\9.png" alt="9" style="zoom:45%;" />

<img src="F:\as_prjs_all\10.png" alt="10" style="zoom:45%;" />

# 后记

写在最后，这次项目包含三个项目，历时两周，虽然费了很大心血，但是验收时还是不是非常尽人愿，甚至是用一些小手法才通过，所以还有很多的遗憾，包括最后部署到手机APP做出的识别效果也是非常差（在此严肃批评mindvision项目组，这么有（pai piao）意义的项目为啥就不继续更新了（生气.gif））

ps：代码及原理有想探讨探讨的，可以联系QQ：2562029787，请注明来意捏~（防诈意识.认真脸）