import numpy as np
import os
from ArcFace import ArcFace
from preprocess_image import preprocess_photo
from compute_cosine_similarity import compute_cosine_similarity
from mindspore import Tensor
from mindspore import dtype as mstype

# 假设您的38张本地照片保存在一个列表中，每张照片都有一个对应的文件名
local_photos_dir = 'dataarcface2'  # 本地照片所在的文件夹路径
input_dir = 'cs'
# 获取本地照片的文件名列表
local_photos = os.listdir(local_photos_dir)

# 准备好输入照片
input_photos = os.listdir(input_dir)

# 预处理输入照片，假设您已经实现了预处理函数 preprocess_photo


# 加载模型
model = ArcFace(world_size=1)

# 提取输入照片的特征向量
# 对于输入照片，label可以设置为0或者任何值，因为您只需要对比与本地照片的相似度

index = []
similarities = []
for pic in input_photos:
    input_path = os.path.join(input_dir, pic)
    preprocessed_input = preprocess_photo(input_path)
    label = Tensor(3, mstype.int32)  # 使用一个标量作为标签
    input_feature = model(preprocessed_input, label=label)
    print("1")
    for local_photo in local_photos:
        # 构建本地照片的完整路径
        local_photo_path = os.path.join(local_photos_dir, local_photo)

        # 预处理本地照片
        preprocessed_local = preprocess_photo(local_photo_path)

        # 提取本地照片的特征向量
        # 使用本地照片的索引作为标签
        label = Tensor(local_photos.index(local_photo), mstype.int32)
        # print(label)
        local_feature = model(preprocessed_local, label=label)

        # 计算余弦相似度
        similarity = compute_cosine_similarity(input_feature, local_feature)
        similarities.append(similarity)
    # 找到相似度最高的本地照片
    most_similar_index = similarities.index(max(similarities))
    most_similar_photo = local_photos[most_similar_index]
    # print(similarities)
    print("Most similar photo:", most_similar_photo)
    index.append(most_similar_photo)
    similarities=[]
# 计算输入照片与本地照片的相似度
print(index)

