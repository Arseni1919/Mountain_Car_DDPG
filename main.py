import gym

# env = gym.make("CartPole-v1")
env = gym.make("MountainCarContinuous-v0")

observation = env.reset()
rewards = 0
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    rewards += reward

    if done:
        observation = env.reset()
        print(f'Done! rewards: {rewards}')
        rewards = 0
env.close()