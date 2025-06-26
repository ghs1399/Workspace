import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def is_point_in_region(x, y, region):
    """判断点是否在区域内"""
    return (region[0] <= x <= region[0] + region[2] and
            region[1] <= y <= region[1] + region[3])

def analyze_answer_sheet(image_path, standard_answers=None):
    """分析答题卡图像并评分"""
    # 读取图像
    I = cv2.imread(image_path)
    if I is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
    # 自适应二值化
    kun1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 3)
    
    # Canny边缘检测
    kun2 = cv2.Canny(gray, 100, 200)
    
    # 移除不需要的区域
    kun2[0:100, :] = 0
    kun2[140:322, :] = 0
    
    # 闭运算连接边缘
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kun3 = cv2.morphologyEx(kun2, cv2.MORPH_CLOSE, se)
    
    # 寻找轮廓并获取边界框
    contours, _ = cv2.findContours(kun3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > 10:  # 过滤掉左侧可能的干扰区域
            stats.append({'BoundingBox': (x, y, w, h)})
    
    # 创建可视化图像
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.title('标记位置')
    plt.axis('off')
    
    # 定义4个主要区域（按region1到region4顺序）
    regions = [
        (16, 100, 42, 200),   # region1
        (70, 100, 42, 200),   # region2
        (120, 100, 42, 200),  # region3
        (175, 100, 42, 200)   # region4
    ]
    
    # 定义A-D选项区域
    option_regions = [
        [(16, 100, 10, 200), (70, 100, 10, 200), (120, 100, 10, 200), (175, 100, 10, 200)],  # A
        [(30, 100, 10, 200), (80, 100, 10, 200), (130, 100, 10, 200), (184, 100, 10, 200)],  # B
        [(40, 100, 10, 200), (90, 100, 10, 200), (140, 100, 10, 200), (192, 100, 10, 200)],  # C
        [(50, 100, 10, 200), (100, 100, 10, 200), (150, 100, 10, 200), (200, 100, 10, 200)]   # D
    ]
    
    option_labels = ['A', 'B', 'C', 'D']  # 选项标签
    answers = []  # 存储答案（区域号, Y坐标, 选项）
    
    # 标记所有检测到的区域
    for stat in stats:
        x, y, w, h = stat['BoundingBox']
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
        center_x = x + w/2
        center_y = y + h/2
        
        # 1. 先判断属于哪个主要区域(region1-4)
        region_num = 0
        for r, region in enumerate(regions):
            if is_point_in_region(center_x, center_y, region):
                region_num = r + 1
                break
        
        # 2. 再判断选项
        option = ''
        for o, region_list in enumerate(option_regions):
            for ro, region_opt in enumerate(region_list):
                if is_point_in_region(center_x, center_y, region_opt):
                    option = option_labels[o]
                    break
            if option:
                break
        
        # 3. 存储有效答案
        if region_num > 0 and option:
            answers.append({'region': region_num, 'y': center_y, 'option': option, 
                           'x_pos': center_x, 'y_pos': center_y})
    
    # 结果排序与对比
    if answers:
        # 排序规则：先按区域号(region1-4)，再按Y坐标(从上到下)
        sorted_answers = sorted(answers, key=lambda x: (x['region'], x['y']))
        
        # 初始化正确题数
        correct_count = 0
        user_answers = ['' for _ in range(20)]  # 存储用户答案（最多20题）
        question_idx = 0  # 当前题号索引
        
        # 处理region4的合并逻辑并对比答案
        prev_y = float('-inf')
        current_options = ''
        current_question = 0
        
        for ans in sorted_answers:
            reg = ans['region']
            y = ans['y']
            opt = ans['option']
            x_pos = ans['x_pos']
            y_pos = ans['y_pos']
            
            if reg == 4:
                # 判断y坐标是否相近（阈值3像素）
                if abs(y - prev_y) < 3:
                    # 相近，合并选项
                    current_options += opt
                else:
                    # 新行或非region4，处理已合并的选项
                    if current_options:
                        # 排序选项以便与标准答案对比
                        sorted_opts = ''.join(sorted(current_options))
                        user_answers[current_question] = sorted_opts
                        print(f'第{current_question+1}题: {sorted_opts}')
                        plt.text(x_pos, y_pos-5, f"{current_question+1}:{sorted_opts}", 
                                color='blue', fontsize=10, fontweight='bold')
                        
                        # 对比标准答案
                        if standard_answers and current_question < len(standard_answers):
                            std_ans = standard_answers[current_question]
                            if sorted_opts == std_ans:
                                correct_count += 1
                                plt.text(x_pos, y_pos+10, '✓', color='green', fontsize=12)
                            else:
                                plt.text(x_pos, y_pos+10, '✗', color='red', fontsize=12)
                    
                    # 重置当前选项，更新题号
                    current_options = opt
                    current_question = question_idx
                    question_idx += 1
                
                prev_y = y
            else:
                # 非region4区域，直接处理单选项
                user_answers[question_idx] = opt
                print(f'第{question_idx+1}题: {opt}')
                plt.text(x_pos, y_pos-5, f"{question_idx+1}:{opt}", 
                        color='blue', fontsize=10, fontweight='bold')
                
                # 对比标准答案
                if standard_answers and question_idx < len(standard_answers):
                    std_ans = standard_answers[question_idx]
                    if opt == std_ans:
                        correct_count += 1
                        plt.text(x_pos, y_pos+10, '✓', color='green', fontsize=12)
                    else:
                        plt.text(x_pos, y_pos+10, '✗', color='red', fontsize=12)
                
                question_idx += 1
        
        # 处理最后一组region4的合并选项
        if current_options and current_question < 20:
            sorted_opts = ''.join(sorted(current_options))
            user_answers[current_question] = sorted_opts
            print(f'第{current_question+1}题: {sorted_opts}')
            plt.text(x_pos, y_pos-5, f"{current_question+1}:{sorted_opts}", 
                    color='blue', fontsize=10, fontweight='bold')
            
            # 对比标准答案
            if standard_answers and current_question < len(standard_answers):
                std_ans = standard_answers[current_question]
                if sorted_opts == std_ans:
                    correct_count += 1
                    plt.text(x_pos, y_pos+10, '✓', color='green', fontsize=12)
                else:
                    plt.text(x_pos, y_pos+10, '✗', color='red', fontsize=12)
        
        # 输出统计结果
        print('\n=== 答题统计 ===')
        total_questions = min(20, question_idx)
        print(f'正确题目数: {correct_count}/{total_questions}')
        print(f'正确率: {correct_count/total_questions*100:.1f}%')
        
        # 输出详细对比
        print('\n=== 答案对比 ===')
        for q in range(20):
            user_ans = user_answers[q] if q < len(user_answers) else '未识别'
            std_ans = standard_answers[q] if standard_answers and q < len(standard_answers) else 'N/A'
            result = '正确' if user_ans == std_ans else '错误'
            print(f'第{q+1}题: 你的答案={user_ans} | 标准答案={std_ans} | {result}')
        
        plt.tight_layout()
        plt.show()
    else:
        print('警告: 未识别到任何答案，请检查区域坐标设置。')

if __name__ == "__main__":
    # 标准答案（20题）
    standard_answers = [
        'A', 'B', 'C', 'D', 'A',
        'A', 'B', 'C', 'D', 'B',
        'A', 'B', 'C', 'D', 'C',
        'AB', 'BC', 'BC', 'CD', 'ABC'
    ]
    
    # 请替换为你的图像路径
    image_path = r"answer_sheet2.png"
    
    # 确保中文能正常显示
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    analyze_answer_sheet(image_path, standard_answers)    