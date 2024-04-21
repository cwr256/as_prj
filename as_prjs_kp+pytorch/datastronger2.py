import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# 定义数据增强函数
def augment_image(image):
    augmented_images = []

    # 随机角度
    random_angles = [random.randint(-30, 30) for _ in range(10)]
    # 随机平移
    random_translations = [(random.randint(-20, 20), random.randint(-20, 20)) for _ in range(10)]

    for angle in random_angles:
        # 旋转
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale=1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # 缩放
        scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5)  # 缩放因子为1.5

        augmented_images.extend([rotated_image, scaled_image])

    for translation in random_translations:
        # 平移
        translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

        augmented_images.append(translated_image)

    # 随机大小遮挡
    for _ in range(10):
        mask_height = random.randint(10, 50)  # 遮挡高度
        mask_width = random.randint(10, 50)  # 遮挡宽度
        x = random.randint(0, cols - mask_width)  # 遮挡位置 x 坐标
        y = random.randint(0, rows - mask_height)  # 遮挡位置 y 坐标

        masked_image = image.copy()
        masked_image[y:y + mask_height, x:x + mask_width] = 0  # 将遮挡区域置为黑色

        augmented_images.append(masked_image)

    # 图像亮度增强
    for _ in range(40):
        brightness_factor = random.uniform(0.5, 1.5)  # 随机亮度因子
        brightness_augmented_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        augmented_images.append(brightness_augmented_image)

    # 图像亮度减弱
    for _ in range(40):
        brightness_factor = random.uniform(0.5, 1.5)  # 随机亮度因子
        brightness_augmented_image = cv2.convertScaleAbs(image, alpha=1.0/brightness_factor, beta=0)
        augmented_images.append(brightness_augmented_image)

    # 镜像翻转
    for _ in range(40):
        mirrored_image = cv2.flip(image, 1)  # 1表示水平镜像
        augmented_images.append(mirrored_image)

    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)  # 锐化核
    sharpened_image = cv2.filter2D(image, -1, kernel)
    for _ in range(40):
        augmented_images.append(sharpened_image)

    return augmented_images

# 数据增强后保存的目录
output_dir = 'data04'

# 遍历每个学生的文件夹
for student_folder in tqdm(os.listdir('face2')):
    student_folder_path = os.path.join('face2', student_folder)
    output_student_folder_path = os.path.join(output_dir, student_folder)
    os.makedirs(output_student_folder_path, exist_ok=True)

    # 读取学生文件夹中的所有图片
    student_images = []
    for filename in os.listdir(student_folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(student_folder_path, filename)
            image = cv2.imread(image_path)
            student_images.append(image)

    # 从学生文件夹中随机选择10张图片进行数据增强
    selected_images = random.sample(student_images, 10)
    for image in selected_images:
        # 进行数据增强
        augmented_images = augment_image(image)

        # 保存增强后的图像
        for idx, augmented_image in enumerate(augmented_images):
            output_filename = f"{student_folder}_{idx}.jpg"
            output_path = os.path.join(output_student_folder_path, output_filename)
            cv2.imwrite(output_path, augmented_image)
