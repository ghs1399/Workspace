import cv2
import numpy as np

def preprocess_image(image_path):
    """读取图像并进行预处理（灰度化、高斯模糊）"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return image, blurred

def detect_license_plate_color(image):
    """在 HSV 颜色空间中筛选车牌颜色区域（以蓝底车牌为例，可扩展其他颜色）"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 蓝底车牌的 HSV 大致范围，可根据实际情况微调
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def find_candidate_contours(mask):
    """查找颜色筛选后的轮廓"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours_by_shape(contours, image_width, image_height):
    """根据轮廓的形状、比例等特征筛选车牌轮廓"""
    candidate_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or area > image_width * image_height * 0.5:  # 过滤过小或过大区域
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        # 车牌的宽高比大致范围，可根据实际情况调整
        if 2 < aspect_ratio < 5:  
            candidate_contours.append(contour)
    return candidate_contours

def crop_and_correct_license_plate(contour, image):
    """根据轮廓抠取车牌区域并尝试校正"""
    x, y, w, h = cv2.boundingRect(contour)
    license_plate = image[y:y + h, x:x + w]
    # 这里简单做了灰度化+阈值二值化的校正（可根据实际需求换更复杂的方法，比如透视变换等）
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_plate

def detect_license_plate(image_path):
    """完整的车牌检测流程"""
    image, blurred = preprocess_image(image_path)
    height, width = image.shape[:2]
    color_mask = detect_license_plate_color(image)
    contours = find_candidate_contours(color_mask)
    filtered_contours = filter_contours_by_shape(contours, width, height)
    
    license_plates = []
    for contour in filtered_contours:
        plate = crop_and_correct_license_plate(contour, image)
        license_plates.append(plate)
    return license_plates

if __name__ == "__main__":
    image_paths = [
        "car1.png", "car2.png"]  # 替换成你实际的图片路径列表
    for path in image_paths:
        result_plates = detect_license_plate(path)
        if result_plates:
            for i, plate in enumerate(result_plates):
                cv2.imshow(f"Detected License Plate {i + 1} in {path}", plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"在 {path} 中未检测到车牌")