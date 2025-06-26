import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random


class SimpleRRT:
    def __init__(
        self, start, goal, obstacles, step_size=0.5, max_iter=5000, goal_sample_rate=0.1
    ):
        """
        Simple RRT (Rapidly-exploring Random Tree) path planner.

        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacles: List of obstacles [(x, y, radius), ...]
            step_size: Maximum step size for extending the tree
            max_iter: Maximum number of iterations
            goal_sample_rate: Probability of sampling the goal
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate

        # Define the boundary of the workspace
        self.x_min, self.x_max = 0, 21
        self.y_min, self.y_max = 0, 21

        # Tree representation: nodes and edges
        self.nodes = [self.start]  # List of all nodes
        self.parents = {0: None}  # Dictionary mapping node index to parent index

        # For statistics
        self.stats = {"iterations": 0, "nodes_added": 0, "time_taken": 0}

    def check_collision(self, p1, p2=None):
        """
        Check if a point or line segment collides with any obstacle.

        Args:
            p1: First point [x, y]
            p2: Second point [x, y], if provided, checks the line segment p1-p2

        Returns:
            True if collision occurs, False otherwise
        """
        # Check if p1 is in collision
        for ox, oy, r in self.obstacles:
            if np.hypot(p1[0] - ox, p1[1] - oy) <= r:
                return True

        # If p2 is provided, check the line segment
        if p2 is not None:
            # Check multiple points along the segment
            for t in np.linspace(0, 1, 10):
                pt = p1 * (1 - t) + p2 * t
                for ox, oy, r in self.obstacles:
                    if np.hypot(pt[0] - ox, pt[1] - oy) <= r:
                        return True

        return False

    def sample_free(self):
        """
        Sample a random point in the free space.

        Returns:
            A random point [x, y] in free space
        """
        if random.random() < self.goal_sample_rate:
            return self.goal.copy()

        while True:
            # Sample a random point
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            point = np.array([x, y])

            # Check if the point is collision-free
            if not self.check_collision(point):
                return point

    def nearest_node(self, point):
        """
        Find the nearest node in the tree to the given point.

        Args:
            point: Target point [x, y]

        Returns:
            Index of the nearest node
        """
        distances = [np.linalg.norm(point - node) for node in self.nodes]
        return np.argmin(distances)

    def steer(self, from_point, to_point):
        """
        Steer from from_point toward to_point, respecting the step_size.

        Args:
            from_point: Starting point [x, y]
            to_point: Target point [x, y]

        Returns:
            New point [x, y] after steering
        """
        dist = np.linalg.norm(to_point - from_point)

        if dist <= self.step_size:
            return to_point
        else:
            direction = (to_point - from_point) / dist
            return from_point + direction * self.step_size

    def plan(self):
        """
        Execute the RRT algorithm to find a path from start to goal.

        Returns:
            path: List of points from start to goal, or None if no path is found
            iterations: Number of iterations performed
        """
        start_time = time.time()

        for i in range(self.max_iter):
            # Sample a random point
            random_point = self.sample_free()

            # Find the nearest node
            nearest_idx = self.nearest_node(random_point)
            nearest_point = self.nodes[nearest_idx]

            # Steer toward the random point
            new_point = self.steer(nearest_point, random_point)

            # Check for collision
            if not self.check_collision(nearest_point, new_point):
                # Add the new node to the tree
                self.nodes.append(new_point)
                new_idx = len(self.nodes) - 1
                self.parents[new_idx] = nearest_idx
                self.stats["nodes_added"] += 1

                # Check if we can connect to the goal
                dist_to_goal = np.linalg.norm(new_point - self.goal)
                if dist_to_goal <= self.step_size:
                    if not self.check_collision(new_point, self.goal):
                        # Add goal to the tree
                        self.nodes.append(self.goal)
                        goal_idx = len(self.nodes) - 1
                        self.parents[goal_idx] = new_idx

                        # Path found!
                        self.stats["iterations"] = i + 1
                        self.stats["time_taken"] = time.time() - start_time

                        return self.extract_path(goal_idx), self.stats["iterations"]

        # No path found after max_iter iterations
        self.stats["iterations"] = self.max_iter
        self.stats["time_taken"] = time.time() - start_time
        print("Failed to find a path!")
        return None, self.max_iter

    def extract_path(self, goal_idx):
        """
        Extract the path from start to goal by backtracking.

        Args:
            goal_idx: Index of the goal node

        Returns:
            path: List of points from start to goal
        """
        path = [self.nodes[goal_idx]]
        current_idx = goal_idx

        while self.parents[current_idx] is not None:
            current_idx = self.parents[current_idx]
            path.append(self.nodes[current_idx])

        # Reverse the path to get it from start to goal
        return path[::-1]

    def path_pruning(self, path):
        """
        Prune a path by removing unnecessary waypoints.
        Checks if a direct connection between non-adjacent waypoints is possible.

        Args:
            path: Original path as a list of points

        Returns:
            Pruned path as a list of points
        """
        if path is None or len(path) <= 2:
            return path

        # Start with the first point in the pruned path
        pruned_path = [path[0]]
        i = 0

        # Try to connect to the furthest possible point
        while i < len(path) - 1:
            # Start from the furthest possible point
            for j in range(len(path) - 1, i, -1):
                # Use more intensive collision checking for safety
                if not self.check_collision_intensive(path[i], path[j]):
                    # Add this point to the pruned path
                    pruned_path.append(path[j])
                    i = j
                    break
            else:
                # If no direct connection is possible, use the next point
                i += 1
                if i < len(path):
                    pruned_path.append(path[i])

        return pruned_path

    def path_smoothing(self, path, smoothing_iterations=50):
        """
        Smooth a path using random point replacement method with rigorous collision checking.

        Args:
            path: Original path as a list of points
            smoothing_iterations: Number of smoothing iterations

        Returns:
            Smoothed path as a list of points
        """
        if path is None or len(path) <= 2:
            return path

        # Make a copy of the path
        smoothed_path = [np.array(p) for p in path]
        path_length = len(smoothed_path)

        for iteration in range(smoothing_iterations):
            # Select two random indices (excluding start and goal for safety)
            if path_length <= 3:  # Not enough points to smooth
                break

            # Select random indices with at least one node between them
            i = random.randint(0, path_length - 3)
            j = random.randint(i + 2, path_length - 1) if i + 2 < path_length else i + 2

            if j >= path_length:
                continue

            # Generate a random point between points i and j
            alpha = random.random()
            new_point = smoothed_path[i] * (1 - alpha) + smoothed_path[j] * alpha

            # Check for collision with obstacles at the new point
            if self.check_collision(new_point):
                continue

            # Test a new path replacing points between i and j with the new point
            test_path = smoothed_path.copy()
            test_path[i + 1 : j] = [new_point]

            # Verify every segment in the test path
            collision_free = True
            for k in range(len(test_path) - 1):
                if self.check_collision_intensive(test_path[k], test_path[k + 1]):
                    collision_free = False
                    break

            # Only update the path if all segments are collision-free
            if collision_free:
                smoothed_path = test_path
                path_length = len(smoothed_path)

        # Final validation of the entire path
        for i in range(len(smoothed_path) - 1):
            if self.check_collision_intensive(smoothed_path[i], smoothed_path[i + 1]):
                print(
                    f"Warning: Found collision in final smoothed path between points {i} and {i+1}."
                )
                print("Reverting to original path for safety.")
                return path

        return smoothed_path

    def smooth_corners(self, path, radius=1.0, resolution=5):
        """
        Smooth sharp corners in a path by adding interpolated points around corners.

        Args:
            path: List of waypoints [x,y]
            radius: Radius of the corner smoothing
            resolution: Number of points to use for each corner

        Returns:
            New path with smoothed corners
        """
        if path is None or len(path) <= 2:
            return path

        # Make a copy of the path using numpy arrays
        points = [np.array(p) for p in path]
        smooth_path = [points[0]]  # Start with the first point

        # Process each corner (a point with a point before and after it)
        for i in range(1, len(points) - 1):
            prev_pt = points[i - 1]
            corner_pt = points[i]
            next_pt = points[i + 1]

            # Vectors from the corner to the adjacent points
            v1 = prev_pt - corner_pt
            v2 = next_pt - corner_pt

            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            # Skip if points are too close together
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                smooth_path.append(corner_pt)
                continue

            v1 = v1 / v1_norm
            v2 = v2 / v2_norm

            # Angle between vectors
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

            # Only smooth sharp corners
            if abs(angle) < np.pi / 6:  # If angle is small (less than 30 degrees)
                smooth_path.append(corner_pt)
                continue

            # Calculate the maximum allowed radius for this corner
            max_radius = min(v1_norm, v2_norm) * 0.5
            actual_radius = min(radius, max_radius)

            # Points along each segment at 'radius' distance from the corner
            p1 = corner_pt + v1 * actual_radius
            p2 = corner_pt + v2 * actual_radius

            # Generate intermediate points along an arc
            smooth_path.append(p1)

            # Calculate the center of the circular arc
            # We find the point equidistant from p1, p2, and corner_pt
            # This is complex geometry, so we'll use a simpler approximation
            for j in range(1, resolution):
                # Simple linear interpolation for now
                t = j / resolution
                # Use a circular arc approximation
                intermediate = p1 * (1 - t) + p2 * t

                # Ensure intermediate points are collision-free
                if not self.check_collision(intermediate):
                    # Verify the connections are collision-free
                    if len(smooth_path) > 0 and not self.check_collision_intensive(
                        smooth_path[-1], intermediate
                    ):
                        smooth_path.append(intermediate)

            smooth_path.append(p2)

        # Add the last point
        smooth_path.append(points[-1])

        # Final check for collisions along the entire path
        for i in range(len(smooth_path) - 1):
            if self.check_collision_intensive(smooth_path[i], smooth_path[i + 1]):
                # If any collision is found, return the original path
                print(
                    "Warning: Collision detected in corner-smoothed path. Reverting to original."
                )
                return path

        return smooth_path

    def post_process_path(self, path, corner_radius=0.5):
        """
        Apply full path post-processing with pruning, smoothing, and corner smoothing.

        Args:
            path: Original RRT path
            corner_radius: Radius for corner smoothing

        Returns:
            Processed path
        """
        if path is None or len(path) < 2:
            return path

        print("Applying path pruning...")
        pruned_path = self.path_pruning(path)

        print("Applying path smoothing...")
        smoothed_path = self.path_smoothing(pruned_path, smoothing_iterations=50)

        print("Smoothing sharp corners...")
        corner_smoothed_path = self.smooth_corners(
            smoothed_path, radius=corner_radius, resolution=3
        )

        # Calculate improvement statistics
        original_length = self.compute_path_length(path)
        pruned_length = self.compute_path_length(pruned_path)
        smoothed_length = self.compute_path_length(smoothed_path)
        corner_length = self.compute_path_length(corner_smoothed_path)

        print(f"Original path: {original_length:.2f} units, {len(path)} nodes")
        print(f"Pruned path: {pruned_length:.2f} units, {len(pruned_path)} nodes")
        print(f"Smoothed path: {smoothed_length:.2f} units, {len(smoothed_path)} nodes")
        print(
            f"Corner-smoothed path: {corner_length:.2f} units, {len(corner_smoothed_path)} nodes"
        )

        return pruned_path, smoothed_path, corner_smoothed_path

    def check_collision_intensive(self, p1, p2):
        """
        More intensive collision check for a line segment using more sample points.

        Args:
            p1: First point [x, y]
            p2: Second point [x, y]

        Returns:
            True if collision occurs, False otherwise
        """
        # Determine number of checks based on distance
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        num_checks = max(
            20, int(dist * 5)
        )  # Minimum 20 checks, or 5 checks per unit distance

        # Check multiple points along the segment
        for t in np.linspace(0, 1, num_checks):
            pt = np.array(p1) * (1 - t) + np.array(p2) * t
            for ox, oy, r in self.obstacles:
                # Add a small safety margin to the radius
                if np.hypot(pt[0] - ox, pt[1] - oy) <= r + 0.05:
                    return True

        return False

    def get_tree(self):
        """
        Get the tree as a list of line segments.

        Returns:
            List of line segments [(x1, y1, x2, y2), ...]
        """
        edges = []
        for idx, parent_idx in self.parents.items():
            if parent_idx is not None:
                edges.append(
                    (
                        self.nodes[parent_idx][0],
                        self.nodes[parent_idx][1],
                        self.nodes[idx][0],
                        self.nodes[idx][1],
                    )
                )
        return edges

    @staticmethod
    def compute_path_length(path):
        """
        Compute the total length of a path.

        Args:
            path: List of points [x, y]

        Returns:
            Total path length
        """
        return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))

    def plot_results(self, path=None, show_tree=True, title_suffix=""):
        """
        Plot the RRT tree, obstacles, and path.

        Args:
            path: List of points forming the path
            show_tree: Whether to show the RRT tree
            title_suffix: Additional text for the title

        Returns:
            Plotly figure
        """
        # Create figure
        fig = make_subplots(rows=1, cols=1)

        # Plot obstacles
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
                )
            )

        # Plot the RRT tree
        if show_tree:
            edges = self.get_tree()
            for x1, y1, x2, y2 in edges:
                fig.add_trace(
                    go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode="lines",
                        line=dict(color="rgba(0,0,255,0.5)", width=1),
                        showlegend=False,
                    )
                )

        # Plot the start and goal points
        fig.add_trace(
            go.Scatter(
                x=[self.start[0]],
                y=[self.start[1]],
                mode="markers+text",
                marker=dict(size=10, color="green"),
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
                marker=dict(size=10, color="black"),
                text=["Goal"],
                textposition="top center",
                name="Goal",
            )
        )

        # Plot the path if found
        if path is not None:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            fig.add_trace(
                go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode="lines+markers",
                    line=dict(color="red", width=3),
                    marker=dict(size=4, color="red"),
                    name="Path",
                )
            )

        # Add statistics
        path_length = self.compute_path_length(path) if path is not None else 0
        path_nodes = len(path) if path is not None else 0

        title = f"RRT Path Planning{title_suffix}<br>"
        if path is not None:
            title += f"Length: {path_length:.2f} units, Nodes: {path_nodes}"
        else:
            title += "No path found"

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(scaleanchor="y", range=[-1, 22]),
            yaxis=dict(range=[-1, 22]),
            showlegend=True,
        )

        return fig


def run_rrt_demo():
    """Run a demo of the RRT algorithm with the example scenario."""
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

    # Create RRT planner
    rrt = SimpleRRT(
        start=start_point,
        goal=goal_point,
        obstacles=obstacles,
        step_size=1.0,  # Larger step size for faster exploration
        max_iter=5000,  # Maximum iterations
        goal_sample_rate=0.1,  # 10% chance to sample the goal
    )

    # Plan a path
    print("Planning path with RRT...")
    original_path, iterations = rrt.plan()

    if original_path is not None:
        print(f"Path found in {iterations} iterations!")
        original_length = rrt.compute_path_length(original_path)
        print(f"Original path length: {original_length:.2f} units")
        print(f"Original path nodes: {len(original_path)}")
        print(f"Number of tree nodes: {rrt.stats['nodes_added']}")
        print(f"Time taken: {rrt.stats['time_taken']:.4f} seconds")

        # Verify the original path is collision-free
        orig_collision = False
        for i in range(len(original_path) - 1):
            if rrt.check_collision_intensive(original_path[i], original_path[i + 1]):
                orig_collision = True
                print("WARNING: Original path has collision!")
                break

        # Post-process the path
        pruned_path, smoothed_path, corner_smoothed_path = rrt.post_process_path(
            original_path, corner_radius=0.6
        )

        pruned_length = rrt.compute_path_length(pruned_path)
        smoothed_length = rrt.compute_path_length(smoothed_path)
        corner_length = rrt.compute_path_length(corner_smoothed_path)

        # Calculate reduction percentages
        pruned_reduction = (1 - pruned_length / original_length) * 100
        smoothed_reduction = (1 - smoothed_length / original_length) * 100
        corner_reduction = (1 - corner_length / original_length) * 100

        # Verify paths are collision-free
        pruned_collision = False
        for i in range(len(pruned_path) - 1):
            if rrt.check_collision_intensive(pruned_path[i], pruned_path[i + 1]):
                pruned_collision = True
                print("WARNING: Pruned path has collision!")
                break

        smoothed_collision = False
        for i in range(len(smoothed_path) - 1):
            if rrt.check_collision_intensive(smoothed_path[i], smoothed_path[i + 1]):
                smoothed_collision = True
                print(f"WARNING: Smoothed path has collision!")
                break

        corner_collision = False
        for i in range(len(corner_smoothed_path) - 1):
            if rrt.check_collision_intensive(
                corner_smoothed_path[i], corner_smoothed_path[i + 1]
            ):
                corner_collision = True
                print(f"WARNING: Corner-smoothed path has collision!")
                break

        # Create a figure with multiple plots for comparison
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Original RRT Path",
                "Pruned Path",
                "Smoothed Path",
                "Corner-Smoothed Path",
            ),
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

        # Original Path
        orig_title = f" (Original, Length: {original_length:.2f})"
        if orig_collision:
            orig_title += " - HAS COLLISION!"
        fig1 = rrt.plot_results(original_path, title_suffix=orig_title)

        # Pruned Path
        pruned_title = (
            f" (Pruned, Length: {pruned_length:.2f}, {pruned_reduction:.1f}% shorter)"
        )
        if pruned_collision:
            pruned_title += " - HAS COLLISION!"
        fig2 = rrt.plot_results(pruned_path, show_tree=False, title_suffix=pruned_title)

        # Smoothed Path
        smoothed_title = f" (Smoothed, Length: {smoothed_length:.2f}, {smoothed_reduction:.1f}% shorter)"
        if smoothed_collision:
            smoothed_title += " - HAS COLLISION!"
        fig3 = rrt.plot_results(
            smoothed_path, show_tree=False, title_suffix=smoothed_title
        )

        # Corner-Smoothed Path
        corner_title = f" (Corner-Smoothed, Length: {corner_length:.2f}, {corner_reduction:.1f}% shorter)"
        if corner_collision:
            corner_title += " - HAS COLLISION!"
        fig4 = rrt.plot_results(
            corner_smoothed_path, show_tree=False, title_suffix=corner_title
        )

        # Extract traces from each figure and add to subplots
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)

        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)

        for trace in fig3.data:
            fig.add_trace(trace, row=2, col=1)

        for trace in fig4.data:
            fig.add_trace(trace, row=2, col=2)

        # Main title with summary
        main_title = "RRT Path Planning and Optimization<br>"
        main_title += f"Original: {original_length:.2f} units | "
        main_title += f"Pruned: {pruned_length:.2f} units | "
        main_title += f"Smoothed: {smoothed_length:.2f} units | "
        main_title += f"Corner-Smoothed: {corner_length:.2f} units"

        fig.update_layout(title_text=main_title)

        # Update axes for all subplots
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    range=[-1, 22], scaleanchor=f"y{i}", scaleratio=1, row=i, col=j
                )
                fig.update_yaxes(range=[-1, 22], row=i, col=j)

        fig.show()

        return rrt, original_path, pruned_path, smoothed_path, corner_smoothed_path

    else:
        # Plot the results even if no path is found
        fig = rrt.plot_results(None)
        fig.show()

        return rrt, None, None, None, None


if __name__ == "__main__":
    run_rrt_demo()