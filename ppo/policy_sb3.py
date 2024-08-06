import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import trange
import statistics
from stable_baselines3.common.callbacks import EvalCallback

from refinement.goal import Goal
# from refinement.graph import Node
from refinement.utils import CacheStates
from stable_baselines3 import PPO

def sample_policy(env: gym.Env, observation:np.ndarray, policy:PPO):
    
    final_terminated = False
    total_reward = 0
    traj = [observation]
    while True:
        action, _ = policy.predict(observation)
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        traj.append(observation)
        total_reward+=reward

        final_terminated = info['is_success']
        

        if final_terminated or terminated or truncated:
            break
    
    return final_terminated, total_reward, info, traj


def train_policy(env: gym.Env, end_node, n_episodes=3000, stored_states: list = None):


    # env.set_abstract_states(start_node, end_node)
    env.set_abstract_states(stored_states, end_node)
    eval_callback = EvalCallback(env, eval_freq=1000,
                             deterministic=True, render=True, )
    
    
    model = PPO("MultiInputPolicy", env, verbose=0)
    model.learn(total_timesteps = n_episodes, progress_bar=True)
    model.save("ppo_cartpole")
    return model


def test_policy( policy: PPO, env: gym.Env, end_node, n_episodes_test, stored_states: list = None):
    
    cached_states = CacheStates()
    env.set_abstract_states(stored_states, end_node)

    reach = []
    rewards = [0]
    episodes = trange(n_episodes_test, desc='reach')
    # episodes = range(3000)
    final_states = []
    for episode in episodes:
        
        observation, info = env.reset()
            
        
        # start_observation = observation['achieved_goal']
        
        reached, reward, final_observation, info = sample_policy(env, observation, policy)
        reach.append(reached)
        rewards.append(reward)
        cached_states.insert(end_node.goal.current_center, reached)
        state_id = env.save_state()
        
        if reached:
            final_states.append((final_observation, state_id))
            
        episodes.set_description(f"Current reach: {sum(reach)/len(reach):.2f}, total_reach: {sum(reach)}, reward: {statistics.mean(rewards):.2f}±{statistics.stdev(rewards):.1f}")
        
    

    return sum(reach)/len(reach), cached_states, final_states
