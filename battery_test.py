import cv2
import numpy as np

def detect_battery_cells(image_path, min_area=300, min_circularity=0.7):
    """
    检测纽扣电池：新增圆形度过滤，解决不规则噪声误检
    :param image_path: 图片路径
    :param min_area: 最小轮廓面积（过滤小噪声）
    :param min_circularity: 最小圆形度（过滤非圆形噪声）
    :return: 绘制结果的图像、电池信息列表
    """
    # 1. 图像预处理
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 分类参数
    area_min_small = 100  
    area_max_small = 500  
    area_min_mid = 800   
    area_max_mid = 1500  

    battery_info = []  

    for cnt in contours:
        # 过滤1：最小面积
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  

        # 过滤2：圆形度（新增！）
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue  # 避免除以0
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue  # 非圆形，视为噪声

        # 计算中心坐标
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m01'] != 0 else 0

        # 区分电池大小
        if area_min_small < area < area_max_small:
            size_type = "Small"
        elif area_min_mid < area < area_max_mid:
            size_type = "Medium"
        else:
            size_type = "Large"

        # 判断正反面
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y + h, x:x + w]
        mean, std_dev = cv2.meanStdDev(roi)
        std_dev = std_dev[0][0]

        front_back = "Front" if std_dev > 15 else "Back"

        # 记录信息 + 绘制结果
        battery_info.append((size_type, front_back, (cx, cy)))
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(image, f"{size_type} | {front_back}", (cx - 40, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return image, battery_info


# -------------------------- 运行演示 --------------------------
if __name__ == "__main__":
    image_paths = ["battery1.png"]  # 替换为你的图片路径
    
    # 关键参数调整：
    # min_area：过滤小噪声；min_circularity：过滤非圆形轮廓（建议0.7~0.8）
    result_img, info = detect_battery_cells(image_paths[0], min_area=300, min_circularity=0.7)  
    
    cv2.imshow("Result", result_img)
    print("Detection Results:")
    for idx, (size, fb, pos) in enumerate(info):
        print(f"  Battery {idx+1}: Size[{size}] | Side[{fb}] | Position{pos}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()