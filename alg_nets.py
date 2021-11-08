import torch

from GLOBALS import *


class ActorNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, n_actions)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

        self.net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            # nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            # nn.ReLU(),
            self.fc2,
            nn.Tanh(),
            # nn.Sigmoid(),
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        # if type(state) is np.ndarray:
        #     state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = state.float()
        value = self.net(state)

        return value


class CriticNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, n_agents: int):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(obs_size * n_agents, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, obs_size * n_agents)
        self.fc3 = nn.Linear(obs_size * n_agents + n_actions * n_agents, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, 1)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)

        self.obs_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            # nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            # nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            self.fc3,
            nn.ReLU(),
            # nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            # nn.ReLU(),
            self.fc4,
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.n_agents = n_agents
        self.entropy_term = 0

    def forward(self, state, action):
        # if type(state) is not torch.Tensor:
        #     # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        #     state = Variable(torch.tensor(state, requires_grad=True).float().unsqueeze(0))
        #     action = Variable(torch.tensor(action, requires_grad=True).float().unsqueeze(0))

        state = state.float()
        obs = self.obs_net(state)
        cat = torch.cat([obs, action], dim=1)
        value = self.out_net(cat)

        return value
