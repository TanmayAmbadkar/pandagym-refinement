from refinement.graph import Node, depth_first_traversal
from refinement.goal import Goal
import gymnasium as gym
import panda_gym_test
import numpy as np
import argparse
import random
import torch
import os
from env.env import CustomEnv
from stable_baselines3.common.utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-e", "--exp-id", type=int, default=0)
parser.add_argument("-g", "--grid-size", type=int, default=3)
parser.add_argument("-n", "--n-episodes", type=int, default=100000)
parser.add_argument("-t", "--n-episodes-test", type=int, default=3000)
parser.add_argument("-m", "--min-reach", type=float, default=0.9)

if __name__ == "__main__":
    args = parser.parse_args()
    print("seed:", args.seed)
    print("exp_id:", args.exp_id)
    print("grid_size:", args.grid_size)
    print("timesteps:", args.n_episodes)
    
    path = f"results/{args.grid_size}_grid-exp_{args.exp_id}-n_ep_{args.n_episodes}-seed_{args.seed}"
    if not os.path.exists(path):
        os.makedirs(path)

    set_random_seed(seed=args.seed, using_cuda=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    

    start_region = Goal(center = np.array([0, 0, 0]), radius = 0.1)
    mid_region = Goal(center = np.array([0.25, 0, 0.2]), radius = 0.1)
    goal_region = Goal(center = np.array([0.38, 0.0, 0.15]), radius = 0.05)

    start_node = Node(start_region, False, False, "start")
    mid_node = Node(mid_region, True, False, "mid")
    goal_node = Node(goal_region, True, True, "goal")

    start_node.add_child(mid_node)
    mid_node.add_child(goal_node)



    panda_env = gym.make("PandaReach-v3", render_mode = "rgb_array")
    panda_env = CustomEnv(panda_env)
    depth_first_traversal(start_node, panda_env, args.min_reach, args.n_episodes, args.n_episodes_test, path)