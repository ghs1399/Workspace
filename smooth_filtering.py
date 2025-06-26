import numpy as np

# 原始图像矩阵
original_matrix = np.array([
    [8, 1, 1, 1, 1, 8, 1, 1],
    [1, 7, 1, 1, 1, 1, 1, 7],
    [1, 1, 1, 5, 5, 1, 1, 1],
    [5, 8, 5, 5, 5, 5, 5, 8],
    [1, 1, 1, 5, 5, 1, 1, 1],
    [8, 1, 1, 8, 1, 1, 1, 1],
    [1, 7, 1, 1, 1, 1, 1, 7],
    [1, 1, 1, 1, 7, 1, 1, 1]], dtype=float) # 将数据转换为浮点数

# 均值滤波矩阵
def mean_filter(matrix): # 定义均值滤波矩阵
    filtered = np.zeros_like(matrix) # 将其置0
    rows, cols = matrix.shape # 求行列数
    
    for i in range(rows): # 检索每一行
        for j in range(cols): # 检索每一列
            # 获取3x3邻域
            neighborhood = []
            for x in range(i-1, i+2): # 某行及上下两行
                for y in range(j-1, j+2): # 某列及左右两列
                    if 0 <= x < rows and 0 <= y < cols: # 正常情况
                        neighborhood.append(matrix[x, y]) # 原始矩阵
                    else: # 有超出边界的部分
                        neighborhood.append(0.0)  # 边界外补0
            
            # 计算均值
            filtered[i, j] = np.mean(neighborhood)
    
    return filtered

# 中值滤波矩阵
def median_filter(matrix): # 定义中值滤波矩阵
    filtered = np.zeros_like(matrix) # 将其置0
    rows, cols = matrix.shape # 求行列数
    
    for i in range(rows): # 检索每一行
        for j in range(cols): # 检索每一列
            # 获取3x3邻域
            neighborhood = []
            for x in range(i-1, i+2): # 某行及上下两行
                for y in range(j-1, j+2): # 某列及左右两列
                    if 0 <= x < rows and 0 <= y < cols: # 正常情况
                        neighborhood.append(matrix[x, y]) # 原始矩阵
                    else: # 有超出边界的部分
                        neighborhood.append(0)  # 边界外补0
            
            # 计算中值
            filtered[i, j] = np.median(neighborhood)
    
    return filtered

# 应用滤波
mean_result = mean_filter(original_matrix) # 填充结果
median_result = median_filter(original_matrix) # 填充结果

# 输出结果
print("原始矩阵:")
print(original_matrix.astype(int))  # 原始矩阵显示为整数
print("\n3x3均值滤波结果:")
print(np.round(mean_result, 2))  # 均值滤波结果保留2位小数
print("\n3x3中值滤波结果:")
print(median_result.astype(int))  # 中值滤波结果保留整数