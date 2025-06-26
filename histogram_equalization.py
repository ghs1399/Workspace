import numpy as np
import matplotlib.pyplot as plt

# 定义图像数据
image_data = [
    [1, 1, 3, 2, 2, 3, 2, 1],
    [7, 1, 2, 6, 2, 6, 2, 3],
    [6, 3, 1, 6, 0, 4, 6, 1],
    [3, 6, 5, 7, 5, 2, 6, 5],
    [6, 6, 2, 5, 7, 7, 5, 0],
    [2, 6, 0, 5, 7, 2, 5, 0],
    [2, 1, 1, 2, 3, 2, 2, 1],
    [1, 1, 2, 3, 3, 0, 2, 1]
]

image_array = np.array(image_data)
rows, cols = image_array.shape
total_pixels = rows * cols

# 计算原始直方图
hist_original = np.zeros(8)
for i in range(rows):
    for j in range(cols):
        hist_original[image_array[i, j]] += 1

# 计算概率分布
pr_rk = hist_original / total_pixels

# 计算累积分布函数
sk = np.zeros(8)
sk[0] = pr_rk[0]
for k in range(1, 8):
    sk[k] = sk[k-1] + pr_rk[k]

# 映射到新的灰度级
L = 8  # 灰度级数
sk_scaled = np.round((L - 1) * sk)

# 创建均衡化后的图像
equalized_image = np.zeros_like(image_array)
for i in range(rows):
    for j in range(cols):
        equalized_image[i, j] = sk_scaled[image_array[i, j]]

# 计算均衡化后的直方图
hist_equalized = np.zeros(8)
for i in range(rows):
    for j in range(cols):
        hist_equalized[equalized_image[i, j]] += 1

# 打印计算表
print("histogram_equalization_calculation_table:")
print("| r_k | n_k | p_r(r_k) | s_k | s_k' |")
print("|-----|-----|----------|-----|------|")
for k in range(8):
    print(f"| {k} | {int(hist_original[k])} | {pr_rk[k]:.3f} | {sk[k]:.3f} | {int(sk_scaled[k])} |")

# 绘制直方图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(8), hist_original)
plt.title("original_histogram")
plt.xlabel("grayscale_value")
plt.ylabel("number_of_pixels")

plt.subplot(1, 2, 2)
plt.bar(range(8), hist_equalized)
plt.title("balanced_histogram")
plt.xlabel("grayscale_value")
plt.ylabel("number_of_pixels")

plt.tight_layout()
plt.show()