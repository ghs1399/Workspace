import numpy as np
import plotly.graph_objects as go


class PotentialFieldPlanner:
    def __init__(self, start, goal, obstacles, att_gain=1.0, rep_gain=100.0): # 设置参数
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.obstacles = obstacles
        self.att_gain = att_gain
        self.rep_gain = rep_gain

    def attractive_potential(self, x, y): # 计算引力场
        gx, gy = self.goal
        return 0.5 * self.att_gain * ((x - gx) ** 2 + (y - gy) ** 2)

    def repulsive_potential(self, x, y): # 计算斥力场
        U_rep = 0.0
        for ox, oy, r in self.obstacles:
            dx = x - ox
            dy = y - oy
            dist = np.hypot(dx, dy) # 计算欧氏距离（L2范数）
            dist = max(dist, 1e-5)  # 避免除以0
            U_rep += self.rep_gain * np.exp(-((dist / r) ** 2)) # 高斯型斥力场
        return U_rep

    def total_potential(self, x, y): # 计算合力场
        return self.attractive_potential(x, y) + self.repulsive_potential(x, y)

    def generate_field_grid(self, x_range, y_range, resolution=100): # 生成平面内势场图
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.total_potential)(X, Y)
        return X, Y, Z

    def compute_path(
        self,
        step_size=0.01,
        tol=0.3,
        max_iter=5000,
        noise_gain=0.02, # 噪声收益（用于逃离局部极小值）
        stuck_threshold=1e-4, # 局部极小值判定阈值
    ): # 梯度下降法参数
        path = [self.start.copy()]
        current_pos = self.start.copy()

        for i in range(max_iter): # 中心差分近似计算梯度
            eps = 1e-4 # 微小增量（对应偏导数中的无穷小）
            grad_x = (
                self.total_potential(current_pos[0] + eps, current_pos[1])
                - self.total_potential(current_pos[0] - eps, current_pos[1])
            ) / (2 * eps) # X方向的偏导数（偏导数定义式的一个必要不充分条件式子）
            grad_y = (
                self.total_potential(current_pos[0], current_pos[1] + eps)
                - self.total_potential(current_pos[0], current_pos[1] - eps)
            ) / (2 * eps) # Y方向的偏导数（偏导数定义式的一个必要不充分条件式子）
            grad = np.array([grad_x, grad_y]) # 合成梯度向量

            if np.linalg.norm(grad) < stuck_threshold: # 梯度过小视为进入局部极小值，添加噪声以逃离
                noise = noise_gain * np.random.uniform(-1, 1, size=2)
            else:
                noise = 0

            next_pos = current_pos - step_size * grad + noise # 沿负梯度方向移动一个步长
            next_pos = np.clip(next_pos, 0, 21) # 限制不能超出边界
            path.append(next_pos.copy()) # 记录路径
            current_pos = next_pos # 更新位置

            if np.linalg.norm(current_pos - self.goal) < tol: # 进入目标点阈值内判定到达目标点
                print(f"Goal reached in {i + 1} steps.")
                break

        return np.array(path)

    @staticmethod # 装饰器
    def compute_path_length(path):
        return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)) # 计算路径总长

    def plot_3d_field_with_path(
        self, X, Y, Z, path, title="Total Potential Field with Path"
    ): # 绘制3D势场和路径图
        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                z=Z, x=X, y=Y, colorscale="Viridis", opacity=0.9, name="Potential Field"
            ) # 设置颜色和透明度
        ) # X,Y表示坐标值，Z表示势能值

        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=[0.5] * len(path), # 路径高度略高于势场以便观察
                mode="lines+markers",
                line=dict(color="red", width=5), # 路径红色粗线
                marker=dict(size=4, color="red"), # 路径红色标记
                name="Path",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[self.start[0]],
                y=[self.start[1]],
                z=[0.5],
                mode="markers+text",
                marker=dict(size=6, color="green"), # 起点绿色标记
                text=["Start"],
                textposition="top center",
                name="Start",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[self.goal[0]],
                y=[self.goal[1]],
                z=[0.5],
                mode="markers+text",
                marker=dict(size=6, color="black"), # 终点黑色标记
                text=["Goal"],
                textposition="top center",
                name="Goal",
            )
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Potential",
                aspectratio=dict(x=1, y=1, z=0.4), # 调整比例
            ),
        )

        fig.show()

    def plot_2d_path(self, path, title="2D Path with Obstacles"): # 绘制2D路径和障碍
        path_len = self.compute_path_length(path)
        steps = len(path) - 1
        full_title = f"{title}<br>Steps: {steps}, Path Length: {path_len:.2f} units"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                line=dict(color="red", width=3), # 路径红色粗线
                marker=dict(size=4, color="red"), # 路径红色标记
                name="Path",
            )
        )

        for ox, oy, r in self.obstacles: # 绘制障碍
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = ox + r * np.cos(theta)
            circle_y = oy + r * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.2)", # 半透明红色填充
                    line=dict(color="firebrick"), # 边界线绘制
                    name="Obstacle",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[self.start[0]],
                y=[self.start[1]],
                mode="markers+text",
                marker=dict(size=10, color="green"), # 起点绿色标记
                text=["Start"],
                textposition="top center",
                name="Start",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[self.goal[0]],
                y=[self.goal[1]],
                mode="markers+text",
                marker=dict(size=10, color="black"), # 终点黑色标记
                text=["Goal"],
                textposition="top center",
                name="Goal",
            )
        )

        fig.update_layout(
            title=full_title,
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(scaleanchor="y", range=[-1, 22]), # 固定比例
            yaxis=dict(range=[-1, 22]),
            showlegend=True,
            width=750,
            height=750,
        )

        fig.show()


if __name__ == "__main__": # 示例
    start_point = (0, 0)
    goal_point = (20, 20)
    obstacles = [
        (3, 3, 1.0),
        (6, 3, 1.0),
        (9, 6, 1.0),
        (13, 6, 1.0),
        (5, 7, 1.0),
        (7, 10, 1.0),
        (15, 9, 1.0),
        (4, 12, 1.0),
        (10, 12, 1.0),
        (14, 13, 1.0),
        (19, 13, 1.0),
        (16, 17, 1.0),
        (19, 17, 1.0),
    ]

    planner = PotentialFieldPlanner(
        start=start_point,
        goal=goal_point,
        obstacles=obstacles,
        att_gain=2.0,
        rep_gain=80.0,
    ) # 取一些参数

    X, Y, Z = planner.generate_field_grid((0, 21), (0, 21), resolution=100)
    path = planner.compute_path(step_size=0.01, noise_gain=0.03)
    planner.plot_3d_field_with_path(X, Y, Z, path)
    planner.plot_2d_path(path)