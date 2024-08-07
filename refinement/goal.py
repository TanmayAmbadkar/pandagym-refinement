import numpy as np
from scipy.spatial import ConvexHull

class AbstractState():
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius
        self.current_center = None

    def reset(self):
        # Generate a random point in a sphere of radius 1 centered at the origin
        # Using spherical coordinates and then scale to the specified radius and shift to center
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        r = self.radius * np.cbrt(np.random.uniform(0, 1))  # cube root for uniform distribution in a sphere

        # Spherical to Cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Shift to the specified center
        point = np.array([x, y, z]) + self.center
        self.current_center = point

class Goal(AbstractState):
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(center=center, radius=radius)

    def predicate(self, state: np.ndarray):
        return np.linalg.norm(state - self.current_center) <= self.radius / 2
    
    def reward(self, state: np.ndarray, last_state: np.ndarray):
        if self.predicate(state):
            return 10
        else:
            return 0.1/np.sum((state - self.current_center)**2)

class ModifiedGoal(Goal):
    def __init__(self, center: np.ndarray, radius: float, hull, reachable=False):
        super().__init__(center=center, radius=radius)
        self.hull = hull
        self.reachable = reachable

    def reset(self):
        while True:
            super().reset()
            if self.in_goal_region(self.current_center):
                break

    def in_goal_region(self, point):
        new_points = np.vstack([self.hull.points, np.array(point).reshape(1, -1)])
        new_hull = ConvexHull(new_points)
        return list(new_hull.vertices) == list(self.hull.vertices)

    def predicate(self, state: np.ndarray):
        if super().predicate(state):
            return self.in_goal_region(state) == self.reachable
        return False
