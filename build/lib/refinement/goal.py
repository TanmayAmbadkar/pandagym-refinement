import numpy as np


class AbstractState():

    def __init__(self, center:np.ndarray, radius:np.ndarray):

        self.center = center
        self.radius = radius

    def reset(self):
        point = None
        while True:
            # Generate a random point in a cube of side 2*radius centered at the origin
            # and then shift it to the specified center
            random_point = np.random.uniform(-self.radius, self.radius, size=3) + self.center
            
            # Check if the point is inside the sphere
            distance_to_center = np.linalg.norm(random_point - self.center)
            if distance_to_center <= self.radius:
                point = random_point
                break
        
        self.current_center = point

class Goal(AbstractState):

    def __init__(self, center:np.ndarray, radius:np.ndarray):

        super().__init__(center=center, radius=radius)
        self.center = center
        self.radius = radius

    def predicate(self, state:np.ndarray):

        return np.linalg.norm(state - self.current_center) <= self.radius/2
    
    def reward(self, state:np.ndarray):
        if self.predicate(state):
            return 10
        else:
            return 0.1/np.linalg.norm(state - self.current_center)

class ModifiedGoal(Goal):
    def __init__(self, center:np.ndarray, radius:np.ndarray, classifier, reachable=False):

        super().__init__(center = center, radius=radius)
        self.classifier = classifier
        self.reachable = reachable
    
    def reset(self):
        super().reset()
        while not self.classifier.predict(self.current_center.reshape(1, -1)):
            super().reset()

    def predicate(self, state:np.ndarray):
        if super().predicate(state):
            prediction = self.classifier.predict(state.reshape(1, -1))
            return prediction == self.reachable
        else:
            return False