import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_traffic_lights(image_path):
    # 1. 图像读取与预处理
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Matplotlib显示格式
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      # 转换到HSV空间
    height, width = img.shape[:2]

    # 2. 定义精准HSV阈值（适配灯的颜色特性）
    ## 红灯：分两段覆盖“内亮”和“外暗”
    lower_red_bright = np.array([0, 80, 100])   # 亮核心：H(0-15), S(>80), V(>100)
    upper_red_bright = np.array([15, 255, 255])
    lower_red_dark = np.array([160, 50, 50])    # 暗边缘：H(160-180), S(>50), V(>50)
    upper_red_dark = np.array([180, 255, 255])
    
    ## 黄灯：亮黄色，覆盖暖黄到橙黄
    lower_yellow = np.array([20, 80, 80])       # H(20-35), S(>80), V(>80)
    upper_yellow = np.array([35, 255, 255])
    
    ## 绿灯（天蓝色）：适配高亮度青色
    lower_cyan = np.array([85, 60, 70])         # H(85-105), S(>60), V(>70)
    upper_cyan = np.array([105, 255, 255])

    # 3. 生成颜色掩码
    mask_red_bright = cv2.inRange(hsv, lower_red_bright, upper_red_bright)
    mask_red_dark = cv2.inRange(hsv, lower_red_dark, upper_red_dark)
    mask_red = cv2.bitwise_or(mask_red_bright, mask_red_dark)  # 合并两段红色
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)

    # 4. 形态学操作（适度去噪，避免破坏灯的轮廓）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆核适配圆形灯
    ## 开运算：去噪（1次迭代）
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_cyan = cv2.morphologyEx(mask_cyan, cv2.MORPH_OPEN, kernel, iterations=1)
    ## 闭运算：连接轮廓（1次迭代）
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_cyan = cv2.morphologyEx(mask_cyan, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. 轮廓筛选（面积 + 圆形度 + 垂直分布约束）
    def filter_contours(contours, min_area=150):
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue  # 过滤小噪声
            
            # 圆形度判断（允许一定变形，适配灯的边缘）
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.3:  # 宽松圆形度，适配非完美圆形
                continue
            
            valid.append(cnt)
        
        # 额外约束：交通灯通常垂直排列，筛选y坐标分层的轮廓
        if len(valid) >= 3:
            # 按y坐标排序（从上到下：红灯→黄灯→绿灯）
            valid.sort(key=lambda c: cv2.boundingRect(c)[1])
            # 保留前3个（假设最多3个灯）
            return valid[:3]
        return valid

    # 检测并筛选轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cyan, _ = cv2.findContours(mask_cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_red = filter_contours(contours_red)
    valid_yellow = filter_contours(contours_yellow)
    valid_cyan = filter_contours(contours_cyan)

    # 6. 绘制轮廓（颜色映射：红→黄，黄→红，青→绿）
    result = img_rgb.copy()
    cv2.drawContours(result, valid_red, -1, (255, 255, 0), 2)   # 红灯→黄色轮廓
    cv2.drawContours(result, valid_yellow, -1, (255, 0, 0), 2)  # 黄灯→红色轮廓
    cv2.drawContours(result, valid_cyan, -1, (0, 255, 0), 2)   # 绿灯→绿色轮廓

    # 7. 可视化结果
    plt.figure(figsize=(12, 6))
    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")
    # 检测结果
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Result")
    plt.axis("off")
    # 图例
    legend_handles = [
        plt.Line2D([0], [0], color=(255/255, 255/255, 0/255), lw=2, label='red'),
        plt.Line2D([0], [0], color=(255/255, 0/255, 0/255), lw=2, label='yellow'),
        plt.Line2D([0], [0], color=(0/255, 255/255, 0/255), lw=2, label='green')
    ]
    plt.legend(handles=legend_handles, loc="lower right")
    
    plt.show()


if __name__ == "__main__":
    image_path = "light1.png"  # 替换为实际图像路径
    detect_traffic_lights(image_path)