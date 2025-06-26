import cv2
import numpy as np

# 圆形结构体，便于处理
class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.area = np.pi * radius * radius

def main():
    # 1. 读取图像
    src = cv2.imread("circles.png")
    if src is None:
        print("Error: Could not load image!")
        return -1

    print(f"图像成功加载，尺寸: {src.shape[1]}x{src.shape[0]}")

    # 2. 创建处理副本
    display = src.copy()

    # 3. 多步骤预处理 - 增强小圆检测能力
    # 3.1 转换为灰度图
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 3.2 自适应直方图均衡化 - 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3.3 高斯模糊 - 减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2)

    # 3.4 Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 3.5 形态学操作 - 连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel)

    # 显示预处理步骤（调试用）
    cv2.imshow("Gray", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edges)

    # 4. 霍夫圆检测 - 使用多组参数检测不同大小的圆
    circles = []

    # 4.1 检测小圆 (半径 5-30 像素)
    small_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1,
                                     blurred.shape[0] / 16,  # 最小圆心间距
                                     param1=100, param2=15, minRadius=5, maxRadius=30)
    if small_circles is not None:
        circles.extend(small_circles[0, :])
        print(f"检测到 {len(small_circles[0])} 个小圆")

    # 4.2 检测中圆 (半径 20-80 像素)
    mid_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1,
                                   blurred.shape[0] / 10,
                                   param1=100, param2=25, minRadius=20, maxRadius=80)
    if mid_circles is not None:
        circles.extend(mid_circles[0, :])
        print(f"检测到 {len(mid_circles[0])} 个中圆")   

    # 5. 转换到自定义结构并过滤重复圆
    detected_circles = []
    overlap_threshold = 0.7  # 重叠面积阈值

    for c in circles:
        x, y, r = c
        new_circle = Circle((int(x), int(y)), int(r))

        is_duplicate = False
        for existing in detected_circles:
            # 计算圆心距离
            distance = np.sqrt((new_circle.center[0] - existing.center[0]) ** 2 +
                               (new_circle.center[1] - existing.center[1]) ** 2)

            # 计算重叠面积比例
            area_ratio = new_circle.area / existing.area
            min_radius = min(new_circle.radius, existing.radius)

            # 如果圆心距离小且面积相似，认为是同一个圆
            if distance < min_radius and (0.7 < area_ratio < 1.3):
                # 保留较大的圆
                if new_circle.radius > existing.radius:
                    existing = new_circle
                is_duplicate = True
                break

        if not is_duplicate:
            detected_circles.append(new_circle)

    print(f"最终检测到 {len(detected_circles)} 个圆")

    # 6. 按圆心坐标排序 (从左到右，从上到下)
    if detected_circles:
        # 计算分组阈值，使用最大半径的2倍
        max_radius = max(circle.radius for circle in detected_circles)
        group_threshold = max_radius * 2

        def sort_key(circle):
            group = circle.center[1] // group_threshold
            return (group, circle.center[0])

        detected_circles.sort(key=sort_key)
    else:
        print("警告: 未检测到任何圆，将显示原图")

    # 7. 在原图上标注所有检测到的圆
    for i, c in enumerate(detected_circles):
        # 绘制圆形边界
        cv2.circle(display, c.center, c.radius, (0, 255, 0), 2)

        # 绘制圆心
        cv2.circle(display, c.center, 3, (0, 0, 255), -1)

        # 标注序号 - 根据圆大小调整文字位置
        text_offset = 15 if c.radius > 30 else 10
        cv2.putText(display, str(i + 1),
                    (c.center[0] - text_offset, c.center[1] + text_offset // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 if c.radius > 30 else 0.5,
                    (255, 0, 0),
                    2 if c.radius > 30 else 1)

        # 显示半径信息
        radius_text = f"r:{c.radius}"
        cv2.putText(display, radius_text,
                    (c.center[0] - c.radius, c.center[1] - c.radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

    # 8. 显示中间处理步骤
    # 检查数组是否为空（通过判断元素数量）
    if gray.size > 0 and blurred.size > 0 and edges.size > 0:
        preprocess_display = np.hstack([gray, blurred, edges])
        preprocess_display = cv2.resize(preprocess_display, None, fx=0.6, fy=0.6)
        cv2.imshow("Preprocessing Steps: Gray | Blurred | Edges", preprocess_display)

    # 9. 显示最终结果
    cv2.imshow("Sorted Circles Detection", display)
    cv2.imwrite("detected_circles_result.png", display)
    print("结果已保存至 detected_circles_result.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    main()