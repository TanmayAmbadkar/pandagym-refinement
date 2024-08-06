from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import trange
import statistics
import random


from ppo.utils import RolloutBuffer
from ppo.actor_critic import ActorCritic
from refinement.goal import Goal
# from refinement.graph import Node
from refinement.utils import CacheStates

class PPO:
    def __init__(self, state_dim, action_dim, continuous, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.optimizer = torch.optim.RMSprop([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())


    def __call__(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.cpu().numpy() 


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*F.mse_loss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        del old_states
        del old_actions
        del old_logprobs
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
      
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def sample_policy(env: gym.Env, observation:np.ndarray, policy:PPO, goal:Goal):
    
    final_terminated = False
    total_reward = 0
    ep_len = 0
    while True:

        action = policy(observation['observation'])
        observation, reward, terminated, truncated, info = env.step(action)
        reward += goal.reward(observation['achieved_goal'])
        total_reward+=reward

        policy.buffer.rewards.append(reward)

        if goal.predicate(observation['achieved_goal']):
            final_terminated = True
        
        policy.buffer.is_terminals.append(final_terminated)

        if terminated or truncated or final_terminated:
            break
        
        ep_len+=1
    
    return final_terminated, total_reward, observation, ep_len, info

def train_policy(env: gym.Env, start_node, end_node, stored_states: List = None):

    policy = PPO(
        state_dim = env.observation_space['observation'].shape[0],
        action_dim = env.action_space.shape[0],
        continuous=True,
        lr_actor=0.0003,
        lr_critic=0.0003,
        gamma = 0.95,
        K_epochs = 40,
        eps_clip = 0.2,
        device = "cpu"
    )

    cached_states = CacheStates()

    reach = []
    rewards = [0]
    ep_lens = [0]
    episodes = trange(2000, desc='reach')
    # episodes = range(3000)
    final_states = []
    for episode in episodes:
        
        if stored_states is None:
            observation, _ = env.reset()
        else:
            _, _ = env.reset()
            observation, state_id = random.sample(stored_states, 1)[0]
            env.restore_state(state_id)
            
        end_node.goal.reset()
        
        start_observation = observation['achieved_goal']
        
        reached, reward, final_observation, ep_len, info = sample_policy(env, observation, policy, end_node.goal)
        ep_lens.append(ep_len)
        reach.append(reached or info['is_success'])
        rewards.append(reward)
        cached_states.insert(start_observation, reached or info['is_success'])
        state_id = env.save_state()
        
        if reached or info['is_success']:
            final_states.append((final_observation, state_id))
            
        episodes.set_description(f"Current reach: {sum(reach[-1000:])/1000:.2f}, total_reach: {sum(reach)}, reward: {statistics.mean(rewards):.2f}±{statistics.stdev(rewards):.1f}, ep_len: {statistics.mean(ep_lens):.2f}")
        
        
        if sum(reach[-1000:])/1000 > 0.70:
            break

        if episode % 5 == 0:
            policy.update()
    

    return sum(reach[-1000:])/1000, policy, cached_states, final_states


