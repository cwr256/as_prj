import numpy as np

def compute_cosine_similarity(vector1, vector2):
    """
    计算两个向量的余弦相似度。

    Args:
        vector1: 第一个向量，可以是一维numpy数组或列表。
        vector2: 第二个向量，可以是一维numpy数组或列表。

    Returns:
        余弦相似度。
    """
    vector1 = np.squeeze(vector1)
    vector2 = np.squeeze(vector2)
    dot_product = np.dot(vector1, vector2)  # 计算点积
    norm_vector1 = np.linalg.norm(vector1)  # 计算向量1的范数
    norm_vector2 = np.linalg.norm(vector2)  # 计算向量2的范数
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)  # 计算余弦相似度

    return cosine_similarity

# # 测试 compute_cosine_similarity 函数
# vector1 = np.array([1, 2, 3])
# vector2 = np.array([4, 5, 6])
# cosine_similarity = compute_cosine_similarity(vector1, vector2)
# print(f"Cosine similarity between vector1 and vector2: {cosine_similarity}")
