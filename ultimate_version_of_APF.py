import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PotentialFieldPlanner:
    def __init__(
        self, start, goal, obstacles, att_gain=1.0, rep_gain=100.0, rep_range_factor=1.0
    ):
        """
        Simple but effective Potential Field Path Planner.

        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacles: List of obstacles [(x, y, radius), ...]
            att_gain: Gain for attractive potential
            rep_gain: Gain for repulsive potential
            rep_range_factor: Factor to scale obstacle influence range (smaller = steeper hills)
        """
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.obstacles = obstacles  # list of (x, y, radius)
        self.att_gain = att_gain
        self.rep_gain = rep_gain
        self.rep_range_factor = rep_range_factor

        # Statistics
        self.stats = {"stuck_count": 0}

    def attractive_potential(self, x, y):
        """Calculate attractive potential toward goal"""
        gx, gy = self.goal
        dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)

        # Quadratic potential within threshold, linear potential beyond
        # This helps prevent too strong attraction from far away
        threshold = 5.0
        if dist <= threshold:
            return 0.5 * self.att_gain * dist**2
        else:
            return self.att_gain * (threshold * dist - 0.5 * threshold**2)

    def repulsive_potential(self, x, y):
        """Calculate repulsive potential from obstacles using sharper localized Gaussian fields"""
        U_rep = 0.0
        for ox, oy, r in self.obstacles:
            dx = x - ox
            dy = y - oy
            dist = np.hypot(dx, dy)
            dist = max(dist, 1e-5)  # avoid division by zero

            # Create steeper and more localized potential hills
            influence_range = r * self.rep_range_factor

            # Use a modified Gaussian with sharper dropoff
            # Only apply repulsion within a limited range (2 sigma)
            if dist < influence_range * 2.0:
                # Quadratic dropoff for steeper hills
                normalized_dist = dist / influence_range
                # Use fourth power for even faster dropoff
                U_rep += self.rep_gain * np.exp(-(normalized_dist**4))

        return U_rep

    def total_potential(self, x, y):
        """Calculate total potential at a point"""
        return self.attractive_potential(x, y) + self.repulsive_potential(x, y)

    def compute_gradient(self, x, y, epsilon=1e-4):
        """Compute the gradient of the potential field at a point"""
        # Central difference approximation
        grad_x = (
            self.total_potential(x + epsilon, y) - self.total_potential(x - epsilon, y)
        ) / (2 * epsilon)
        grad_y = (
            self.total_potential(x, y + epsilon) - self.total_potential(x, y - epsilon)
        ) / (2 * epsilon)

        return np.array([grad_x, grad_y])

    def generate_field_grid(self, x_range, y_range, resolution=100):
        """Generate a grid of potential field values for visualization"""
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)
        X, Y = np.meshgrid(x, y)

        # Vectorized computation of potential field
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.total_potential(X[i, j], Y[i, j])

        return X, Y, Z

    def check_collision(self, point):
        """Check if a point collides with any obstacle"""
        for ox, oy, r in self.obstacles:
            if np.hypot(point[0] - ox, point[1] - oy) <= r:
                return True
        return False

    def compute_path(self, step_size=0.01, tol=0.3, max_iter=5000, noise_gain=0.02):
        """Compute path using gradient descent with enhancements for local minima escape"""
        path = [self.start.copy()]
        current_pos = self.start.copy()
        stuck_count = 0
        prev_positions = []  # For oscillation detection
        prev_potential = float("inf")  # To ensure we're always going downhill
        no_progress_count = 0  # Count iterations with no progress

        print("Computing path...")
        for i in range(max_iter):
            # Store previous positions to detect oscillation
            if len(prev_positions) > 10:
                prev_positions.pop(0)
            prev_positions.append(current_pos.copy())

            # Compute gradient and current potential
            grad = self.compute_gradient(current_pos[0], current_pos[1])
            grad_norm = np.linalg.norm(grad)
            current_potential = self.total_potential(current_pos[0], current_pos[1])

            # Direction to goal for reference
            to_goal = self.goal - current_pos
            dist_to_goal = np.linalg.norm(to_goal)
            to_goal = to_goal / max(dist_to_goal, 1e-10)

            # Check if we're making progress (going downhill)
            if current_potential >= prev_potential and i > 0:
                no_progress_count += 1
            else:
                no_progress_count = 0

            prev_potential = current_potential

            # Check if stuck (small gradient or no progress)
            if grad_norm < 1e-4 or no_progress_count > 5:
                stuck_count += 1

                # Generate random direction biased toward goal
                random_dir = np.random.uniform(-1, 1, size=2)
                random_dir = random_dir / max(np.linalg.norm(random_dir), 1e-10)

                # Increase goal bias as we get stuck more often
                goal_bias = min(0.3 + stuck_count * 0.05, 0.7)
                escape_dir = (1 - goal_bias) * random_dir + goal_bias * to_goal
                escape_dir = escape_dir / max(np.linalg.norm(escape_dir), 1e-10)

                # Take larger step when stuck
                next_pos = (
                    current_pos
                    + noise_gain * (1 + min(stuck_count * 0.1, 1.0)) * escape_dir
                )

                # Reset no progress counter
                no_progress_count = 0
            else:
                # Always follow the negative gradient (downhill)
                # Use adaptive step size based on gradient magnitude
                adaptive_step = step_size / (
                    1 + 0.1 * grad_norm
                )  # Smaller steps for steeper gradients
                next_pos = current_pos - adaptive_step * grad / max(grad_norm, 1e-10)

            # Ensure the path stays within bounds
            next_pos = np.clip(next_pos, 0, 21)

            # Try the step and check if it actually reduces the potential
            next_potential = self.total_potential(next_pos[0], next_pos[1])

            # If we're not going downhill and not stuck, adjust the step
            if (
                next_potential > current_potential
                and stuck_count == 0
                and no_progress_count == 0
            ):
                # Try a smaller step in the same direction
                direction = next_pos - current_pos
                direction = direction / max(np.linalg.norm(direction), 1e-10)

                # Try multiple step sizes to find one that goes downhill
                for factor in [0.5, 0.25, 0.1, 0.05]:
                    test_pos = current_pos + factor * step_size * direction
                    test_pos = np.clip(test_pos, 0, 21)

                    if not self.check_collision(test_pos):
                        test_potential = self.total_potential(test_pos[0], test_pos[1])
                        if test_potential < current_potential:
                            next_pos = test_pos
                            next_potential = test_potential
                            break

            # Check for collision with obstacles
            collision = self.check_collision(next_pos)
            if collision:
                # Identify the obstacle we're colliding with
                colliding_obstacle = None
                min_dist = float("inf")

                for ox, oy, r in self.obstacles:
                    dist = np.hypot(next_pos[0] - ox, next_pos[1] - oy)
                    if dist < r + 0.1 and dist < min_dist:
                        colliding_obstacle = (ox, oy, r)
                        min_dist = dist

                if colliding_obstacle:
                    ox, oy, r = colliding_obstacle
                    # Vector from obstacle to current position
                    vec_from_obs = current_pos - np.array([ox, oy])
                    dist_from_obs = np.linalg.norm(vec_from_obs)

                    if dist_from_obs < 1e-10:
                        # If exactly at obstacle center, move randomly
                        next_pos = current_pos + noise_gain * 2.0 * np.random.uniform(
                            -1, 1, size=2
                        )
                    else:
                        # Calculate tangent vectors in both directions
                        tangent_cw = (
                            np.array([-vec_from_obs[1], vec_from_obs[0]])
                            / dist_from_obs
                        )
                        tangent_ccw = -tangent_cw

                        # Try both directions and see which one has lower potential
                        pos_cw = current_pos + step_size * 2.0 * tangent_cw
                        pos_ccw = current_pos + step_size * 2.0 * tangent_ccw

                        # Also consider which direction is more aligned with goal
                        goal_alignment_cw = np.dot(tangent_cw, to_goal)
                        goal_alignment_ccw = np.dot(tangent_ccw, to_goal)

                        # Weighted decision: 0.7 * potential + 0.3 * goal alignment
                        pot_cw = self.total_potential(pos_cw[0], pos_cw[1])
                        pot_ccw = self.total_potential(pos_ccw[0], pos_ccw[1])

                        score_cw = 0.7 * pot_cw - 0.3 * goal_alignment_cw
                        score_ccw = 0.7 * pot_ccw - 0.3 * goal_alignment_ccw

                        if score_cw < score_ccw:
                            next_pos = pos_cw
                        else:
                            next_pos = pos_ccw

            # Check if the path is oscillating (revisiting positions)
            oscillating = False
            if len(prev_positions) > 5:
                for prev_pos in prev_positions[:-3]:
                    if np.linalg.norm(next_pos - prev_pos) < 0.05:
                        oscillating = True
                        break

            if oscillating:
                # Break oscillation with a random move biased toward goal
                # Use a larger step and more goal bias
                random_dir = np.random.uniform(-1, 1, size=2)
                random_dir = random_dir / max(np.linalg.norm(random_dir), 1e-10)
                next_pos = current_pos + step_size * 3.0 * (
                    0.8 * to_goal + 0.2 * random_dir
                )

                # Reset potential tracking to allow some uphill movement after oscillation
                prev_potential = float("inf")

            # Add to path and update current position
            path.append(next_pos.copy())
            current_pos = next_pos

            # Check if reached goal
            if dist_to_goal < tol:
                print(f"Goal reached in {i + 1} steps!")
                break

            # Give up if stuck too many times
            if stuck_count > 50:
                print(f"Giving up after {i + 1} steps. Couldn't reach goal.")
                break

        self.stats["stuck_count"] = stuck_count
        print(f"Path planning completed with {stuck_count} stuck incidents")
        return np.array(path)

    @staticmethod
    def compute_path_length(path):
        """Compute the total length of a path"""
        return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))

    def plot_field_and_path(
        self, path, x_range=(0, 21), y_range=(0, 21), resolution=100
    ):
        """Create visualizations for the potential field and path"""
        # Generate field
        X, Y, Z = self.generate_field_grid(x_range, y_range, resolution)

        # Create figure with two subplots: 3D and 2D
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=["3D Potential Field", "2D Path View"],
            horizontal_spacing=0.05,
        )

        # 3D Surface plot
        fig.add_trace(
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Potential", len=0.5, y=0.5),
                showscale=True,
            ),
            row=1,
            col=1,
        )

        # 3D Path plot
        path_z = []
        for point in path:
            try:
                # Get Z value at path point
                i = min(
                    max(0, int((point[0] - X[0, 0]) / (X[0, 1] - X[0, 0]))),
                    Z.shape[1] - 1,
                )
                j = min(
                    max(0, int((point[1] - Y[0, 0]) / (Y[1, 0] - Y[0, 0]))),
                    Z.shape[0] - 1,
                )
                z_val = Z[j, i] + 5  # Lift above surface
                path_z.append(z_val)
            except (IndexError, ValueError):
                path_z.append(100)  # Fallback

        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path_z,
                mode="lines",
                line=dict(color="red", width=5),
                name="Path",
            ),
            row=1,
            col=1,
        )

        # 3D Start and goal points
        fig.add_trace(
            go.Scatter3d(
                x=[self.start[0]],
                y=[self.start[1]],
                z=[path_z[0]],
                mode="markers+text",
                marker=dict(size=8, color="green"),
                text=["Start"],
                textposition="top center",
                name="Start",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=[self.goal[0]],
                y=[self.goal[1]],
                z=[path_z[-1]],
                mode="markers+text",
                marker=dict(size=8, color="black"),
                text=["Goal"],
                textposition="top center",
                name="Goal",
            ),
            row=1,
            col=1,
        )

        # 2D Path plot
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                line=dict(color="red", width=3),
                marker=dict(size=4, color="red"),
                name="Path",
            ),
            row=1,
            col=2,
        )

        # 2D Contour plot
        fig.add_trace(
            go.Contour(
                z=Z,
                x=X[0, :],
                y=Y[:, 0],
                colorscale="Viridis",
                opacity=0.5,
                showscale=False,
                contours=dict(
                    start=np.min(Z),
                    end=np.percentile(Z, 95),
                    size=(np.percentile(Z, 95) - np.min(Z)) / 20,
                ),
            ),
            row=1,
            col=2,
        )

        # Add obstacles to 2D view
        for ox, oy, r in self.obstacles:
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = ox + r * np.cos(theta)
            circle_y = oy + r * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.2)",
                    line=dict(color="firebrick"),
                    name="Obstacle",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # 2D Start and goal points
        fig.add_trace(
            go.Scatter(
                x=[self.start[0]],
                y=[self.start[1]],
                mode="markers+text",
                marker=dict(size=10, color="green"),
                text=["Start"],
                textposition="top center",
                name="Start",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[self.goal[0]],
                y=[self.goal[1]],
                mode="markers+text",
                marker=dict(size=10, color="black"),
                text=["Goal"],
                textposition="top center",
                name="Goal",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Path statistics
        path_length = self.compute_path_length(path)
        steps = len(path) - 1

        # Update layout
        fig.update_layout(
            title=f"Potential Field Path Planning<br>Steps: {steps}, Path Length: {path_length:.2f} units",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Potential",
                aspectratio=dict(x=1, y=1, z=0.4),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            ),
            xaxis=dict(scaleanchor="y", range=[-1, 22]),
            yaxis=dict(range=[-1, 22]),
        )

        return fig


if __name__ == "__main__":
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

    # Create planner with optimized parameters
    planner = PotentialFieldPlanner(
        start=start_point,
        goal=goal_point,
        obstacles=obstacles,
        att_gain=1.0,  # Attractive force gain
        rep_gain=150.0,  # Repulsive force gain
        rep_range_factor=1.0,  # Smaller value = more localized obstacle influence
    )

    # Compute path
    path = planner.compute_path(
        step_size=0.01,  # Small steps for smooth path
        noise_gain=0.02,  # Noise for local minima escape
    )

    # Plot results
    fig = planner.plot_field_and_path(path)
    fig.show()