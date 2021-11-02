import gym
from alg_plotter import plotter


# env = gym.make("CartPole-v1")
env = gym.make("MountainCarContinuous-v0")

observation = env.reset()
rewards = 0
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    rewards += reward
    plotter.plots_update_data({'Reward': reward, 'Total Reward': rewards})

    if done:
        observation = env.reset()
        print(f'Done! rewards: {rewards}')
        rewards = 0

    plotter.plots_online()
env.close()