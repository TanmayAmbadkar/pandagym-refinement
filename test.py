import gymnasium as gym
import panda_gym_test


env = gym.make("PandaReach-v3", render_mode = "human")

print("State Space", env.observation_space)
print("Action Space", env.action_space.shape)