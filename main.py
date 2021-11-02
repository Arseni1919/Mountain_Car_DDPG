import gym
import numpy as np
import torch

from alg_plotter import plotter
from alg_env_wrapper import env
from alg_nets import *
# env = gym.make("CartPole-v1")
# env = gym.make("MountainCarContinuous-v0")
observation = env.reset()
critic = CriticNet(obs_size=2, n_actions=1, n_agents=1)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
actor = ActorNet(obs_size=2, n_actions=1)
actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
rewards = 0
for i in range(100000):
    env.render()
    # --------------------------- # ACTOR # -------------------------- #
    actor_optim.zero_grad()
    observation_i = observation.detach()
    action = actor(observation_i)
    actor_loss = - critic(observation_i, action)
    actor_loss.backward()
    actor_optim.step()
    # ---------------------------------------------------------------- #
    # action = env.sample_action()  # your agent here (this takes random actions)
    new_observation, reward, done, info = env.step(action)
    rewards += reward.item()
    # -------------------------- # CRITIC # -------------------------- #

    critic_optim.zero_grad()
    output_value = critic(observation, action.detach())
    loss = nn.MSELoss()
    with torch.no_grad():
        next_action = actor(new_observation)
        next_critic_value = critic(new_observation, next_action)
        target = reward + GAMMA * next_critic_value if not done else reward
    critic_loss = loss(output_value, target)
    critic_loss.backward()
    critic_optim.step()
    # ---------------------------------------------------------------- #
    # obs1 = np.linspace(-1.2, 0.6, 20)
    # obs2 = np.linspace(-0.07, 0.07, 20)
    # critic_values = np.zeros(shape=[20, 20])
    # with torch.no_grad():
    #     for k, ob1 in enumerate(obs1):
    #         for j, ob2 in enumerate(obs2):
    #             state = Variable(torch.tensor([ob1, ob2]).float().unsqueeze(0))
    #             critic_values[k][j] = critic(state, action).item()


    plotter.plots_update_data({
        'Reward': reward.item(),
        'Output': output_value.item(),
        'critic_loss': critic_loss.item(),
        'actor_loss': actor_loss.item(),
        'obs1': observation[0][0],
        'obs2': observation[0][1],
    })
    # plotter.plots_update_data({
    #     'obs1': obs1,
    #     'obs2': obs2,
    #     'critic_values': critic_values
    # }, no_list=True)
    observation = new_observation
    if done:
        observation = env.reset()
        print(f'Done! rewards: {rewards}')
        rewards = 0

    if i % 10 == 0:
        plotter.plots_online()

plotter.close()
env.close()
