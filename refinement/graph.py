
from __future__ import annotations

import numpy as np
import gymnasium as gym
from ppo.policy_sb3 import train_policy, test_policy
from refinement.utils import CacheStates, train_model
from refinement.goal import Goal, ModifiedGoal
class Node():

    def __init__(self, goal:np.ndarray, splittable:bool = True, final:bool=False, name:str = ""):
        self.goal = goal
        self.splittable = splittable
        self.children = {}
        self.final = final
        self.name = name
        self.idx = 0 

    def sample_state(self):
        return self.goal.sample_state()
    
    def __iter__ (self):
        return self

    def add_child(self, child:Node):
        self.children[id(child)] = {
            "child": child, 
            "reach_probability": 0, 
            "policy": None
        }

    def __next__(self):
        
        keys = list(self.children.keys())
        if self.idx == len(keys):
            raise StopIteration
        else:
            self.idx+=1
            return self.children[keys[self.idx-1]]

    def remove_child(self, child:Node):
        self.children.pop(id(child))
    
    def print_graph(self):
        pass

def split_goal(goal:Goal, cached_states:CacheStates):

    model = train_model(cached_states)

    goal_nr = ModifiedGoal(
        center = goal.center, 
        radius=goal.radius, 
        hull=model,
        reachable=False
    )

    goal_r = ModifiedGoal(
        center = goal.center, 
        radius=goal.radius, 
        hull=model,
        reachable=True
    )

    return goal_nr, goal_r


def depth_first_traversal(head: Node, env: gym.Env, minimum_reach: float = 0.9, n_episodes: int = 3000, n_episodes_test: int = 3000, path: str = ""):

    edges = []
    
    file = open(path + "/result.txt", "w")
    explore(head, None, env, None, minimum_reach, edges, n_episodes, n_episodes_test, file)

def explore(parent: Node, grandparent: Node, env: gym.Env, stored_states: list = None, minimum_reach: float = 0.9, edges: list = [], n_episodes: int = 3000, n_episodes_test: int = 3000, file = None):

    if parent.final:
        return False

    for child in parent:
        if parent.name+"_"+child['child'].name not in edges:
            
            print(f"Evaluating edge ({parent.name}, {child['child'].name})")
            policy = train_policy(env, child['child'], n_episodes, stored_states)
            reach, cached_states, final_states = test_policy(policy, env, child['child'], n_episodes_test, stored_states)

            print(f"Edge ({parent.name}, {child['child'].name}) reach probability: {reach}")

            # print(f"{parent.name}, {child['child'].name}: {reach}", file=file)
            # if reach < minimum_reach and parent.splittable and grandparent is not None:
            if reach < minimum_reach and child['child'].splittable:

                print(f"Edge ({parent.name}, {child['child'].name}) not realised: {reach}")
                _, goal_r = split_goal(goal = child['child'].goal, cached_states = cached_states)

                goal_r_node = Node(
                    goal = goal_r, 
                    splittable=False,
                    final = child['child'].final,
                    name = child['child'].name + "_r"
                )
                goal_r_node.children = child['child'].children

                # goal_nr_node = Node(
                #     goal = goal_nr, 
                #     splittable=False,
                #     final = parent.final,
                #     name = parent.name + "_nr"
                # )

                # goal_nr_node.add_child(goal_r_node)
                # goal_r_node.add_child(child['child'])
                parent.add_child(goal_r_node)

                # grandparent.add_child(goal_r_node)
                # grandparent.add_child(goal_nr_node)
            
            parent.children[id(child['child'])]['reach_probability'] = reach
            parent.children[id(child['child'])]['policy'] = policy
            edges.append(parent.name+"_"+child['child'].name)

            del cached_states
            status = explore(child['child'], parent, env, final_states, minimum_reach, edges, n_episodes, n_episodes_test, file)
            
            if status:
                return False
        
        if child['child'].final and reach>=minimum_reach:
            return True
            






# from __future__ import annotations

# import numpy as np
# import gymnasium as gym
# from ppo.policy import train_policy
# from refinement.utils import CacheStates, train_model
# from refinement.goal import Goal, ModifiedGoal
# class Node():

#     def __init__(self, goal:np.ndarray, splittable:bool = True, final:bool=False, name:str = ""):
#         self.goal = goal
#         self.splittable = splittable
#         self.children = {}
#         self.final = final
#         self.name = name
#         self.idx = 0 

#     def sample_state(self):
#         return self.goal.sample_state()
    
#     def __iter__ (self):
#         return self

#     def add_child(self, child:Node):
#         self.children[id(child)] = {
#             "child": child, 
#             "reach_probability": 0, 
#             "policy": None
#         }

#     def __next__(self):
        
#         keys = list(self.children.keys())
#         if self.idx == len(keys):
#             raise StopIteration
#         else:
#             self.idx+=1
#             return self.children[keys[self.idx-1]]

#     def remove_child(self, child:Node):
#         self.children.pop(id(child))
    
#     def print_graph(self):
#         pass

# def split_goal(goal:Goal, cached_states:CacheStates):

#     model = train_model(cached_states)

#     goal_nr = ModifiedGoal(
#         center = goal.center, 
#         radius=goal.radius, 
#         classifier=model,
#         reachable=False
#     )

#     goal_r = ModifiedGoal(
#         center = goal.center, 
#         radius=goal.radius, 
#         classifier=model,
#         reachable=True
#     )

#     return goal_nr, goal_r

# def depth_first_traversal(head: Node, env: gym.Env, minimum_reach: float = 0.9):

#     edges = []
    
#     explore(head, None, env, minimum_reach, edges)

# def explore(parent: Node, grandparent: Node, env: gym.Env, minimum_reach: float = 0.9, edges: list = [], start_states: list = None):

#     if parent.final:
#         return

#     for child in parent:
#         if parent.name+"_"+child['child'].name not in edges:
            
#             print(f"Evaluating edge ({parent.name}, {child['child'].name})")
#             reach, policy, cached_states, final_states = train_policy(env, parent, child['child'], start_states)

#             print(f"Edge ({parent.name}, {child['child'].name}) reach probability: {reach}")
#             if reach < minimum_reach and parent.splittable and grandparent is not None:

#                 print(f"Edge ({parent.name}, {child['child'].name}) not realised: {reach}")
#                 goal_nr, goal_r = split_goal(goal = parent.goal, cached_states = cached_states)

#                 goal_r_node = Node(
#                     goal = goal_r, 
#                     splittable=False,
#                     final = parent.final,
#                     name = parent.name + "_r"
#                 )

#                 goal_nr_node = Node(
#                     goal = goal_nr, 
#                     splittable=False,
#                     final = parent.final,
#                     name = parent.name + "_nr"
#                 )

#                 goal_nr_node.add_child(goal_r_node)
#                 goal_r_node.add_child(child['child'])

#                 grandparent.add_child(goal_r_node)
#                 grandparent.add_child(goal_nr_node)
#                 # grandparent.add_child(goal_r_node)
            
#             parent.children[id(child['child'])]['reach_probability'] = reach
#             parent.children[id(child['child'])]['policy'] = policy
#             edges.append(parent.name+"_"+child['child'].name)

#             del cached_states
#             explore(child['child'], parent, env, minimum_reach, edges, final_states)


