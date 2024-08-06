import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
from scipy.spatial import ConvexHull
from collections import deque

class CacheStates:
    def __init__(self, maxlen: int = 10000):
        self.states = deque(maxlen=maxlen)
        self.reachable = deque(maxlen=maxlen)
        
    
    def __len__(self):
        return len(self.states)
    
    def insert(self, state: np.ndarray, reach: bool):
        self.states.append(state)
        self.reachable.append(reach)
    
    def return_dataset(self):
        return np.array(self.states), np.array(self.reachable)
    
    def clear(self):
        self.states.clear()
        self.reachable.clear()

def train_model(cached_states: CacheStates):
    """
    Create a convex hull from the reachable states.

    Args:
        cached_states (CacheStates): Cached states object containing states and reachable flags.

    Returns:
        ConvexHull: Convex hull object created from scipy.spatial.ConvexHull.
    """
    # Extract the reachable points
    dataset = cached_states.return_dataset()
    points = dataset[0][dataset[1] == 1]
    
    # Compute the convex hull of reachable points
    hull = ConvexHull(points)
    
    return hull
