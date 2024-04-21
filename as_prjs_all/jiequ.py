import cv2  # 导入OpenCV库
import os  # 导入os库

# 打开摄像头，0代表系统默认摄像头
cap = cv2.VideoCapture(0)

# 设置停止标志
stop = False

# 加载人脸检测器模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# 设置保存图像的目录
save_dir = 'B21041301'

# 创建保存图像的目录（如果不存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 设置保存图像的计数器
img_counter = 0

# 循环检测人脸
while not stop:
    # 读取摄像头捕获的帧
    success, img = cap.read()

    # 初始化灰度图像变量
    gray = None
    #
    # 检测图像中的人脸
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # # 遍历检测到的人脸，给人脸画上矩形框
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)

    # 显示带有人脸框的图像
    cv2.imshow('img', img)

    # 等待按键事件，每1毫秒检查一次按键
    c = cv2.waitKey(1)

    # 如果按下的是q键，就停止循环
    if c & 0xFF == ord('q'):
        stop = True

    # 如果按下的是s键，就保存人脸图像
    elif c & 0xFF == ord('s'):
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            img_name = os.path.join(save_dir, f"B21041301-{img_counter}.jpg")
            cv2.imwrite(img_name, face_img)
            print(f"Saved {img_name}")
            img_counter += 1

# 关闭摄像头窗口
cap.release()
cv2.destroyAllWindows()
