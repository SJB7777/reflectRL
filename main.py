import gymnasium as gym
from stable_baselines3 import PPO


def main():
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

if __name__ == "__main__":
    main()
