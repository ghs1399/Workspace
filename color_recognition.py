import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 读取图像（替换为实际魔方图片路径）
img_path = "magic_cube1.png"  
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {img_path}，请检查路径是否正确")

# BGR 转 RGB，适配 matplotlib 显示逻辑
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

# RGB 转 HSV 色彩空间，OpenCV 中 H 范围 [0,179]，S、V 范围 [0,255]，后续归一化到 [0,1]
hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hue = hsvImg[:, :, 0] / 180.0  
sat = hsvImg[:, :, 1] / 255.0  
val = hsvImg[:, :, 2] / 255.0  


# 增强对比度函数（模仿 MATLAB imadjust 逻辑）
def imadjust(src, tol=1, vin=[0, 255], vout=[0, 255]):
    tol = max(0, min(100, tol))
    if tol > 0:
        # 计算直方图
        hist = np.zeros(256, dtype=np.int32)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r, c]] += 1
        # 计算累积直方图
        cum = hist.copy()
        for i in range(1, 256):
            cum[i] = cum[i - 1] + hist[i]
        # 确定输入范围边界
        total_pixels = src.shape[0] * src.shape[1]
        low_bound = total_pixels * tol / 100
        upp_bound = total_pixels * (100 - tol) / 100
        vin[0] = 0
        while vin[0] < 255 and cum[vin[0]] < low_bound:
            vin[0] += 1
        vin[1] = 255
        while vin[1] > 0 and cum[vin[1]] > upp_bound:
            vin[1] -= 1
    # 计算拉伸参数并映射
    if vin[1] <= vin[0]:
        scale = 1
    else:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    bias = vout[0] - vin[0] * scale
    dst = src.astype(np.float32) * scale + bias
    return np.clip(dst, vout[0], vout[1]).astype(src.dtype)


# 增强 V（亮度）和 S（饱和度）通道对比度
val = imadjust((val * 255).astype(np.uint8)) / 255.0
sat = imadjust((sat * 255).astype(np.uint8)) / 255.0


# 颜色范围定义（HSV 空间，[色调下限, 色调上限, 饱和度下限, 亮度下限, 亮度上限]）
# 扩大灰色、黄色（含浅黄色）范围，覆盖新色块
color_ranges = [
    # 1. 红色（跨 0 点，涵盖正红、深红等）
    [0.92, 0.08, 0.4, 0.3, 0.9],  
    # 2. 浅绿色
    [0.25, 0.38, 0.3, 0.3, 0.9],  
    # 3. 深绿色
    [0.18, 0.25, 0.3, 0.3, 0.9],  
    # 4. 亮绿色（覆盖新的青色系绿色）
    [0.35, 0.55, 0.3, 0.6, 1.0],  
    # 5. 蓝色
    [0.50, 0.78, 0.3, 0.3, 0.9],  
    # 6. 浅黄色（扩大范围，覆盖更宽的黄色调）
    [0.08, 0.25, 0.05, 0.7, 1.0],  
    # 7. 标准黄色
    [0.14, 0.20, 0.3, 0.4, 0.8],  
    # 8. 橙色（扩大范围）
    [0.02, 0.12, 0.2, 0.5, 1.0],  
    # 9. 黑色
    [0.00, 1.00, 0.0, 0.0, 0.3],  
    # 10. 粉色
    [0.90, 0.05, 0.2, 0.5, 0.9],  
    # 11. 紫色
    [0.70, 0.85, 0.2, 0.3, 0.9],  
    # 12. 灰色（扩大范围：降低饱和度、亮度限制）
    [0.00, 1.00, 0.0, 0.2, 0.8],  
    # 13. 品红
    [0.85, 0.98, 0.3, 0.4, 0.9],  
]

# 对应颜色标签，与上面 color_ranges 顺序一致
color_labels = [
    'Red', 'Light Green', 'Dark Green', 'Bright Green', 
    'Blue', 'Light Yellow', 'Yellow', 'Orange', 
    'Black', 'Pink', 'Purple', 'Gray', 
    'Magenta'
]


# 生成魔方区域掩模
cube_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

for i, (h_min, h_max, s_min, v_min, v_max) in enumerate(color_ranges):
    # 处理红色跨 0 点
    if i == 0:
        mask = ((hue >= h_min) | (hue <= h_max)) & (sat >= s_min) & (val >= v_min) & (val <= v_max)
    else:
        mask = (hue >= h_min) & (hue <= h_max) & (sat >= s_min) & (val >= v_min) & (val <= v_max)
    
    # 形态学去噪
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = mask > 0  # 转回布尔类型
    
    cube_mask |= mask  # 合并掩模


# 优化魔方区域（最大连通域 + 填充 + 膨胀）
contours, _ = cv2.findContours(cube_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError('未检测到魔方区域')

# 保留最大轮廓
contour = max(contours, key=cv2.contourArea)
cube_mask = np.zeros_like(cube_mask, dtype=np.uint8)
cv2.drawContours(cube_mask, [contour], -1, 255, -1)
cube_mask = cv2.fillPoly(cube_mask, [contour], 255) > 0  # 填充孔洞
cube_mask = cv2.dilate(cube_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))) > 0  # 膨胀


# 提取魔方面区域
y, x = np.where(cube_mask)
x_min, x_max = max(0, min(x) - 15), min(img.shape[1] - 1, max(x) + 15)
y_min, y_max = max(0, min(y) - 15), min(img.shape[0] - 1, max(y) + 15)
faceRegion = img[y_min:y_max + 1, x_min:x_max + 1]


# 九宫格识别
height, width = faceRegion.shape[:2]
grid_h, grid_w = round(height / 3), round(width / 3)
gridColors = [[None] * 3 for _ in range(3)]

plt.figure(figsize=(10, 10))
plt.imshow(faceRegion)
plt.axis('on')

for i in range(3):
    for j in range(3):
        # 裁剪格子
        grid = faceRegion[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
        if grid.size == 0:
            continue
        
        # 计算 HSV 中值（抗噪）
        grid_hsv = cv2.cvtColor(grid, cv2.COLOR_RGB2HSV)
        med_h = np.median(grid_hsv[..., 0] / 180.0)
        med_s = np.median(grid_hsv[..., 1] / 255.0)
        med_v = np.median(grid_hsv[..., 2] / 255.0)
        
        # 匹配颜色
        best_score, colorLabel = 0, 'Unknown'
        for k, (h_min, h_max, s_min, v_min, v_max) in enumerate(color_ranges):
            # 红色跨 0 点处理
            if k == 0:
                hue_ok = (med_h >= h_min) or (med_h <= h_max)
            else:
                hue_ok = (med_h >= h_min) and (med_h <= h_max)
            
            sat_ok = (med_s >= s_min)
            val_ok = (med_v >= v_min) and (med_v <= v_max)
            
            # 特殊颜色加强判断
            if k == 3:  # 亮绿色（覆盖青色系）
                score = 1 if (0.35 <= med_h <= 0.55) and (med_s >= 0.3) and (med_v >= 0.6) else 0
            elif k == 5:  # 浅黄色（扩大范围）
                score = 1 if (0.08 <= med_h <= 0.25) and (med_s >= 0.05) and (med_v >= 0.7) else 0
            elif k == 7:  # 橙色（扩大范围）
                score = 1 if (0.02 <= med_h <= 0.12) and (med_s >= 0.2) and (med_v >= 0.5) else 0
            elif k == 11:  # 灰色（扩大范围）
                score = 1 if (med_s < 0.5) and (0.2 <= med_v <= 0.8) else 0  # 简化逻辑：低饱和 + 中亮度
            elif k == 12:  # 品红
                score = 1 if (0.85 <= med_h <= 0.98) and (med_s >= 0.3) and (med_v >= 0.4) else 0
            else:
                score = hue_ok * sat_ok * val_ok
            
            if score > best_score:
                best_score = score
                colorLabel = color_labels[k]
        
        # 绘制标注
        rect = Rectangle((j * grid_w, i * grid_h), grid_w, grid_h,
                         linewidth=2, edgecolor='w', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(j * grid_w + grid_w / 2, i * grid_h + grid_h / 2,
                 colorLabel, color='white', fontsize=12,
                 ha='center', va='center', fontweight='bold')
        
        gridColors[i][j] = colorLabel


plt.tight_layout()
plt.show()

# 输出结果
print('九宫格颜色识别结果：')
for row in gridColors:
    print(row)