import cv2
import numpy as np
import os


def detect_color(hsv_value):
    """Enhanced color detection with better yellow range"""
    h, s, v = hsv_value

    # Improved color ranges with better yellow detection
    color_ranges = {
        "yellow": ((15, 50, 150), (35, 255, 255)),  # Wider yellow range
        "light_yellow": ((20, 30, 180), (35, 150, 255)),  # Light yellow
        "green": ((35, 50, 50), (85, 255, 255)),
        "light_green": ((35, 30, 150), (85, 150, 255)),
        "red": ((0, 50, 50), (10, 255, 255)),
        "red2": ((170, 50, 50), (180, 255, 255)),
        "blue": ((90, 50, 50), (130, 255, 255)),
        "purple": ((130, 50, 50), (160, 255, 255)),
        "pink": ((160, 50, 50), (180, 255, 255)),
        "orange": ((5, 50, 50), (20, 255, 255)),
        "cyan": ((80, 50, 50), (100, 255, 255)),
    }

    # Check each color range
    for color, (lower, upper) in color_ranges.items():
        if (
            lower[0] <= h <= upper[0]
            and lower[1] <= s <= upper[1]
            and lower[2] <= v <= upper[2]
        ):
            return color

    # Special handling for red (wraps around 0/180)
    if (0 <= h <= 10 or 170 <= h <= 180) and s >= 50 and v >= 50:
        return "red"

    return "unknown"


def multi_threshold_detection(image):
    """Use multiple thresholding methods to catch all shapes"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 1: Adaptive threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh1 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Method 2: Otsu's threshold
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Method 3: Manual threshold for light colors
    _, thresh3 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Combine all thresholds
    combined = cv2.bitwise_or(thresh1, cv2.bitwise_or(thresh2, thresh3))

    return combined


def detect_shapes_enhanced(image_path):
    """Enhanced shape detection with better small object detection"""
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image!")
        return None

    original = image.copy()

    # Enhance contrast for better detection
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # Use multiple thresholding methods
    thresh = multi_threshold_detection(enhanced)

    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No shapes detected!")
        return image

    # Process each contour with lower area threshold
    detected_shapes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Lower minimum area threshold to catch small shapes
        if area < 50:  # Reduced from 80 to 50
            continue

        # Shape analysis
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )  # More sensitive approximation
        x, y, w, h = cv2.boundingRect(approx)

        # Color detection using the centroid region
        mask = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Get color from multiple points in the shape
        mean_color = cv2.mean(hsv, mask=mask)[:3]
        color = detect_color(mean_color)

        # Shape classification
        vertices = len(approx)
        aspect_ratio = float(w) / h if h > 0 else 1

        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            if 0.9 <= aspect_ratio <= 1.1:
                shape = "square"
            else:
                shape = "rectangle"
        elif vertices > 6:
            # Check circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                shape = "circle"
            else:
                shape = "polygon"
        else:
            shape = f"{vertices}-sided"

        detected_shapes.append(
            {
                "contour": contour,
                "color": color,
                "shape": shape,
                "area": area,
                "position": (x, y, w, h),
            }
        )

    # Sort by area to draw larger shapes first
    detected_shapes.sort(key=lambda x: x["area"], reverse=True)

    # Draw results
    for shape_info in detected_shapes:
        contour = shape_info["contour"]
        color = shape_info["color"]
        shape = shape_info["shape"]
        x, y, w, h = shape_info["position"]

        # Draw contour
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Add label
        label = f"{color} {shape}"

        # Adjust text size based on shape size
        font_scale = max(0.4, min(0.7, w / 100))
        thickness = 1 if w < 50 else 2

        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Position text above the shape
        text_x = max(0, x)
        text_y = max(text_height + 5, y - 5)

        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (text_x, text_y - text_height - 5),
            (text_x + text_width, text_y + 5),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        print(f"Detected: {label} (Area: {shape_info['area']:.0f})")

    return image


def test_with_debug_info(image_path):
    """Test function with debug information"""
    print("Starting enhanced shape detection...")
    result = detect_shapes_enhanced(image_path)

    if result is not None:
        print("Detection completed successfully!")
        return result
    else:
        print("Detection failed!")
        return None


if __name__ == "__main__":
    image_path = "colors.png"
    result = test_with_debug_info(image_path)

    if result is not None:
        cv2.imshow("Enhanced Shape Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("enhanced_result.png", result)

