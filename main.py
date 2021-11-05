import torch

from alg_plotter import plotter
from alg_env_wrapper import env
from alg_nets import *
from alg_replay_buffer import ReplayBuffer
from play import play


torch.autograd.set_detect_anomaly(True)
# --------------------------- # NETS # -------------------------- #
critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size(), n_agents=1)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
target_critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size(), n_agents=1)
target_critic.load_state_dict(critic.state_dict())
target_actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
target_actor.load_state_dict(actor.state_dict())

# --------------------------- # REPLAY BUFFER # -------------------------- #
replay_buffer = ReplayBuffer()

# --------------------------- # NORMAL DISTRIBUTION # -------------------------- #
current_sigma = SIGMA
normal_distribution = Normal(torch.tensor(0.0), torch.tensor(current_sigma))
# normal_distribution.sample()

# --------------------------- # FIRST OBSERVATION # -------------------------- #
observation = env.reset()

rewards = 0
episode = 0
for step in range(N_STEPS):
    print(f'\r(step {step - REPLAY_BUFFER_SIZE})', end='')
    # --------------------------- # STEP # -------------------------- #
    # action = env.sample_action()  # your agent here (this takes random actions)
    with torch.no_grad():
        noisy_action = actor(observation) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(current_sigma))
        clipped_action = torch.clamp(noisy_action, min=-1, max=1)
        new_observation, reward, done, info = env.step(clipped_action)

    # --------------------------- # STORE # -------------------------- #
    replay_buffer.append((observation, clipped_action, reward, done, new_observation))
    rewards += reward.item()

    # --------------------------- # UPDATE # -------------------------- #
    observation = new_observation
    if done:
        observation = env.reset()
        plotter.plots_update_data({'rewards': rewards})
        if step > REPLAY_BUFFER_SIZE:
            plotter.debug(f'episode: {episode}')
            plotter.debug(f'Done! rewards: {rewards}')
        episode += 1
        rewards = 0

    if step > REPLAY_BUFFER_SIZE:
        current_sigma = current_sigma - (SIGMA/(N_STEPS - len(replay_buffer)))
        if episode % 5 == 0:
            env.render()
        # print(f'step: {step}')
        # --------------------------- # MINIBATCH # -------------------------- #
        minibatch = replay_buffer.sample(n=BATCH_SIZE)

        # --------------------------- # Y # -------------------------- #
        b_observations, b_actions, b_rewards, b_dones, b_next_observations = zip(*minibatch)
        b_observations = torch.stack(b_observations).squeeze()
        b_actions = torch.stack(b_actions).squeeze(1)
        b_rewards = torch.stack(b_rewards).squeeze()
        b_dones = torch.stack(b_dones).squeeze()
        b_next_observations = torch.stack(b_next_observations).squeeze()
        with torch.no_grad():
            next_q = target_critic(state=b_next_observations, action=target_actor(b_next_observations)).squeeze()
            next_q = (~b_dones) * next_q
            y = b_rewards + GAMMA * next_q

        # --------------------------- # UPDATE CRITIC # -------------------------- #
        loss = nn.MSELoss()
        critic_optim.zero_grad()
        critic_loss_input = critic(state=b_observations, action=b_actions).squeeze()
        critic_loss = loss(critic_loss_input, y)
        critic_loss.backward()
        critic_optim.step()

        # --------------------------- # UPDATE ACTOR # -------------------------- #
        actor_optim.zero_grad()
        actor_loss = - critic(b_observations, actor(b_observations)).mean()
        actor_loss.backward()
        actor_optim.step()

        # --------------------------- # UPDATE TARGET NETS # -------------------------- #
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)

        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)

        # --------------------------- # PLOTTER # -------------------------- #
        plotter.plots_update_data({
            # 'reward': reward.item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            # 'action': clipped_action.item(),
            'current_sigma': current_sigma,
        })

        if step % 1000 == 0:
            plotter.plots_online()

        # ---------------------------------------------------------------- #

plotter.close()
env.close()
plotter.info('Finished train.')

# Save Results
if SAVE_RESULTS:
    plotter.info('Saving results...')
    torch.save(actor, f'{SAVE_PATH}/actor.pt')
    torch.save(target_actor, f'{SAVE_PATH}/target_actor.pt')
    # example runs
    plotter.info('Example run...')
    model = torch.load(f'{SAVE_PATH}/target_actor.pt')
    model.eval()
    play(10, model=model)

