import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, panda_env):
        super().__init__()
        
        self.panda_env = panda_env
        
        self.action_space = panda_env.action_space
        self.observation_space = panda_env.observation_space
        self.goal_node = None
        self.start_states = None
        self.avoid = None
        
    def step(self, action):
        
        observation, reward, terminated, truncated, info = self.panda_env.step(action)
        
        if self.goal_node is not None:
            reward+=self.goal_node.goal.reward(observation['achieved_goal'])
            terminated = terminated or self.goal_node.goal.predicate(observation['achieved_goal'])
            info['is_success'] = self.goal_node.goal.predicate(observation['achieved_goal'])
            
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        
        if self.start_states is None:
            observation, info = self.panda_env.reset(seed=seed, options=options)
        else:
            _, _ = self.env.reset()
            observation, state_id = random.sample(self.start_states, 1)[0]
            self.env.restore_state(state_id)    
        
        if self.goal_node is not None:
            self.goal_node.goal.reset()
            observation['desired_goal'] = self.goal_node.goal.current_center
        
        return observation, info

    def render(self):
        self.panda_env.render()
        
    def close(self):
        self.panda_env.close()
        
    def set_abstract_states(self, start_states, goal_node, avoid = None):
        
        self.start_states = start_states
        self.goal_node = goal_node    
        self.avoid = avoid
        self.goal_node.goal.reset()
        
    def save_state(self) -> int:
        """Save the current state of the envrionment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        return self.panda_env.save_state()
    
    def restore_state(self, state_id: int) -> None:
        """Resotre the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.panda_env.restore_state(state_id)
