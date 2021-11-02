from GLOBALS import *

from alg_plotter import plotter


class SingleAgentEnv:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        observation = self.env.reset()
        observation = Variable(torch.tensor(observation, requires_grad=True).float().unsqueeze(0))
        return observation

    def render(self):
        self.env.render()

    def sample_action(self):
        action = self.env.action_space.sample()
        action = Variable(torch.tensor(action, requires_grad=True).float().unsqueeze(0))
        return action

    def sample_observation(self):
        observation = self.env.observation_space.sample()
        observation = Variable(torch.tensor(observation, requires_grad=True).float().unsqueeze(0))
        return observation

    def step(self, action):
        action = action.item()
        observation, reward, done, info = self.env.step([action])
        observation = Variable(torch.tensor(observation, requires_grad=True).float().unsqueeze(0))
        reward = Variable(torch.tensor(reward, requires_grad=True).float().unsqueeze(0))
        return observation, reward, done, info

    def close(self):
        self.env.close()


class MultiAgentEnv:
    def __init__(self):
        pass


env = SingleAgentEnv(env_name=SINGLE_AGENT_ENV_NAME)
