import cv2
import os


def detect_faces_in_frame(frame):
    # 加载人脸检测器模型
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # 初始化灰度图像变量
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测图像中的人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # 遍历检测到的人脸，返回人脸图像列表
    face_images = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_images.append(face_img)

    return face_images
