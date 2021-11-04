from alg_plotter import plotter
from alg_env_wrapper import env
from alg_nets import *
observation = env.reset()
critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size(), n_agents=1)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

# for step in range(100):
#     sample = env.sample_observation()
#     action = actor(sample)
#     print(f'({action.detach().numpy()}), ', end='')
# print()

rewards = 0
for step in range(N_STEPS):
    env.render()
    # --------------------------- # ACTOR # -------------------------- #
    if random.random() > EPSILON:
        if step % 10 == 0:
            actor_optim.zero_grad()
            observation_i = observation.detach()
            action = actor(observation_i)
            actor_loss = - critic(observation_i, action)
            actor_loss.backward()
            actor_optim.step()
        else:
            observation_i = observation.detach()
            action = actor(observation_i)
            actor_loss = torch.tensor(0)
    else:
        action = env.sample_action()
        actor_loss = torch.tensor(0)
    # ---------------------------------------------------------------- #
    # action = env.sample_action()  # your agent here (this takes random actions)
    new_observation, reward, done, info = env.step(action)
    rewards += reward.item()
    # -------------------------- # CRITIC # -------------------------- #
    critic_optim.zero_grad()
    output_value = critic(observation, action.detach())
    if step % 100 == 0:
        with torch.no_grad():
            next_action = actor(new_observation)
            next_critic_value = critic(new_observation, next_action) if not done else 0
            # next_critic_value = 0
            target = reward + GAMMA * next_critic_value
            curr_critic_value = critic(observation, action.detach())
            delta = target - curr_critic_value
        critic_loss = delta * output_value
        critic_loss.backward()
        critic_optim.step()
    else:
        critic_loss = torch.tensor(0)
    # ---------------------------------------------------------------- #


    plotter.plots_update_data({
        'Reward': reward.item(),
        'critic value': output_value.item(),
        'critic_loss': critic_loss.item(),
        'actor_loss': actor_loss.item(),
        # 'action': action.item()
    })
    observation = new_observation
    if done:
        observation = env.reset()
        print(f'Done! rewards: {rewards}')
        rewards = 0

    if step % 100 == 0:
        plotter.plots_online()

plotter.close()
env.close()

