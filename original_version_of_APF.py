import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class ArtificialPotentialField:
    def __init__(self, start, goal, obstacles, step_size=0.1, max_iter=1000):
        
        # 一些有用的参数
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        
        # 势场参数
        self.attractive_gain = 1.0  # 引力增益系数
        self.repulsive_gain = 1.0   # 斥力增益系数
        self.repulsive_range = 3.0  # 障碍物影响范围
        
        # 局部极小值处理参数
        self.min_distance_threshold = 0.2  # 到达终点的阈值
        self.stuck_threshold = 0.1         # 判断陷入局部极小值的移动距离阈值
        self.stuck_iterations = 20         # 判断陷入局部极小值的迭代次数
        self.random_perturbation = 0.1     # 随机扰动幅度
        
        # 记录路径
        self.path = [self.start.copy()]
        
    def attractive_force(self, position):
        """计算引力"""
        direction = self.goal - position # 方向向量
        distance = np.linalg.norm(direction) # 方向向量模长
        if distance < 1e-6:  # 避免除以零
            return np.zeros(2)
        
        # 引力大小与距离成正比（带入书上引力场求完梯度后的公式）
        magnitude = self.attractive_gain * distance # 引力模长
        return magnitude * (direction / distance) # 引力向量
    
    def repulsive_force(self, position):
        """计算所有障碍物的斥力合力"""
        total_force = np.zeros(2)
        
        for obstacle in self.obstacles:
            obs_pos = np.array(obstacle[:2])
            obs_radius = obstacle[2]
            
            # 计算机器人到障碍物的向量
            direction = position - obs_pos # 方向向量
            distance = np.linalg.norm(direction) # 方向向量模长
            
            if distance < 1e-6:  # 避免除以零
                continue
                
            # 如果距离大于影响范围，则斥力为零
            if distance > self.repulsive_range + obs_radius:
                continue
                
            # 计算有效距离 (减去障碍物半径)
            effective_distance = distance - obs_radius
            if effective_distance <= 0:
                effective_distance = 0.01  # 避免除以零，设置一个很小的值
                
            # 斥力大小与距离的平方成反比（带入书上斥力场求完梯度后的公式）
            magnitude = self.repulsive_gain * (1/effective_distance - 1/(self.repulsive_range)) / (effective_distance**2) # 斥力模长
            total_force += magnitude * (direction / distance) # 斥力向量
            
        return total_force
    
    def is_stuck(self, recent_positions):
        """检查是否陷入局部极小值"""
        if len(recent_positions) < self.stuck_iterations:
            return False
            
        # 计算最近几次移动的总距离
        total_movement = 0
        for i in range(1, len(recent_positions)):
            total_movement += np.linalg.norm(recent_positions[i] - recent_positions[i-1])
            
        return total_movement < self.stuck_threshold
    
    def random_perturbation_force(self):
        """生成随机扰动"""
        angle = np.random.uniform(0, 2*np.pi)
        return self.random_perturbation * np.array([np.cos(angle), np.sin(angle)])
    
    def plan_path(self):
        """执行路径规划"""
        current_pos = self.start.copy()
        stuck_counter = 0
        recent_positions = []
        
        for i in range(self.max_iter):
            # 检查是否到达目标
            distance_to_goal = np.linalg.norm(self.goal - current_pos)
            if distance_to_goal < self.min_distance_threshold:
                print("Reached goal!")
                break
                
            # 计算合力
            attractive_f = self.attractive_force(current_pos)
            repulsive_f = self.repulsive_force(current_pos)
            total_force = attractive_f + repulsive_f
            
            # 记录最近位置用于检测局部极小值
            recent_positions.append(current_pos.copy())
            if len(recent_positions) > self.stuck_iterations:
                recent_positions.pop(0)
                
            # 检查是否陷入局部极小值
            if self.is_stuck(recent_positions):
                print("Detected local minimum, applying random perturbation")
                total_force += self.random_perturbation_force()
                stuck_counter += 1
                
                # 如果多次陷入局部极小值，增加随机扰动
                if stuck_counter > 5:
                    self.random_perturbation *= 1.5
            else:
                stuck_counter = 0
                self.random_perturbation = 0.5  # 重置扰动幅度
                
            # 归一化力方向并移动
            force_norm = np.linalg.norm(total_force)
            if force_norm > 1e-6:  # 避免除以零
                direction = total_force / force_norm
            else:
                direction = np.zeros(2)    
            new_pos = current_pos + self.step_size * direction
            
            # 确保新位置不与任何障碍物碰撞
            collision = False
            for obstacle in self.obstacles:
                obs_pos = np.array(obstacle[:2])
                obs_radius = obstacle[2]
                if np.linalg.norm(new_pos - obs_pos) < obs_radius:
                    collision = True
                    break
                    
            if collision:
                # 如果碰撞，尝试减小步长
                new_pos = current_pos + 0.5 * self.step_size * direction
                
            self.path.append(new_pos.copy())
            current_pos = new_pos
            
        else:
            print("Reached maximum iterations without reaching goal")
            
        return np.array(self.path)
    
    def calculate_path_length(self):
        """计算路径长度"""
        if len(self.path) < 2:
            return 0
            
        length = 0
        for i in range(1, len(self.path)):
            length += np.linalg.norm(self.path[i] - self.path[i-1])
            
        return length
    
    def plot_environment(self, path=None):
        """可视化环境和路径"""
        plt.figure(figsize=(20, 20))
        
        # 绘制起点和终点
        plt.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        plt.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            circle = Circle(obstacle[:2], obstacle[2], color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
            
        # 绘制路径
        if path is not None:
            plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
            plt.plot(path[:, 0], path[:, 1], 'b.', markersize=5)
            
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Artificial Potential Field Path Planning')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

# 定义环境和障碍物
start_point = (0, 0)  # 可以根据需要修改起点
goal_point = (20, 20)
obstacles = [
    (3, 3, 1.0), (6, 3, 1.0), (9, 6, 1.0), (13, 6, 1.0),
    (5, 7, 1.0), (7, 10, 1.0), (15, 9, 1.0), (4, 12, 1.0),
    (10, 12, 1.0), (14, 13, 1.0), (19, 13, 1.0), (16, 17, 1.0),
    (19, 17, 1.0)
]

# 创建路径规划器并规划路径
apf = ArtificialPotentialField(start_point, goal_point, obstacles)
path = apf.plan_path()

# 计算路径长度
path_length = apf.calculate_path_length()
print(f"Path length: {path_length:.2f}")

# 可视化结果
apf.plot_environment(path)