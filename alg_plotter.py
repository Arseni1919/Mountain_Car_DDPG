from GLOBALS import *
from alg_functions import *
import logging
"""
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
"""


class ALGPlotter:
    """
    This object is responsible for plotting, logging and neptune updating.
    """
    def __init__(self, plot_life=True, plot_neptune=False, name='', tags=None):

        if tags is None:
            tags = []
        self.plot_life = plot_life
        self.plot_neptune = plot_neptune
        self.run = {}
        self.name = name
        self.tags = tags
        self.fig, self.actor_losses, self.critic_losses, self.ax, self.agents_list = {}, {}, {}, {}, {}
        self.total_reward, self.val_total_rewards = [], []

        self.neptune_init()
        self.logging_init()

        if self.plot_life:
            self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
            self.data_to_plot = {}

        self.info("ALGPlotter instance created.")

    def plots_update_data(self, data_dict, no_list=False):
        if self.plot_life:
            for key_name, value in data_dict.items():
                if no_list:
                    self.data_to_plot[key_name] = value
                else:
                    if key_name not in self.data_to_plot:
                        self.data_to_plot[key_name] = deque(maxlen=50000)
                    self.data_to_plot[key_name].append(value)

    def plots_update_entropy(self, net1, net2, name=''):
        if self.plot_life:
            key_name = f'kl_div_{name}'
            if key_name not in self.data_to_plot:
                self.data_to_plot[key_name] = deque(maxlen=50000)

            kl_div = 0
            list_params_net1 = list(net1.parameters())
            list_params_net2 = list(net2.parameters())
            for p1, p2 in zip(list_params_net1, list_params_net2):
                for i in range(len(p1)):
                    a, b = p1[i].softmax(dim=0).detach().numpy(), p2[i].softmax(dim=0).detach().numpy()
                    kl_div += kl_divergence(a, b)

            self.data_to_plot[key_name].append(kl_div)

    def plots_online(self):
        # plot live:
        if self.plot_life:
            def plot_graph(ax, indx_r, indx_c, list_of_values, label, color='b', cla=True):
                if cla:
                    ax[indx_r, indx_c].cla()
                ax[indx_r, indx_c].plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                # ax[indx_r, indx_c].set_title(f'Plot: {label}')
                # ax[indx_r, indx_c].set_xlabel('iters')
                ax[indx_r, indx_c].set_ylabel(f'{label}')
                ax[indx_r, indx_c].axhline(0, color='gray')

            def plot_graph_axes(axes, list_of_values, label, color='b'):
                axes.cla()
                axes.plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                # axes.set_title(f'Plot: {label}')
                # axes.set_xlabel('iters')
                axes.set_ylabel(f'{label}')

            # counter = 0
            # for key_name, list_of_values in self.data_to_plot.items():
            #     plot_graph_axes(self.fig.axes[counter], list_of_values, key_name)
            #     counter += 1
            if 'reward' in self.data_to_plot:
                plot_graph(self.ax, 0, 0, self.data_to_plot['reward'], 'Reward')
            if 'critic value' in self.data_to_plot:
                plot_graph(self.ax, 0, 0, self.data_to_plot['critic value'], 'critic value', color='red', cla=False)
            if 'current_sigma' in self.data_to_plot:
                plot_graph(self.ax, 0, 0, self.data_to_plot['current_sigma'], 'current_sigma')
            if 'critic_loss' in self.data_to_plot:
                steps = len(self.data_to_plot['critic_loss'])
                list_to_show = moving_average(self.data_to_plot['critic_loss'], steps / 100)
                plot_graph(self.ax, 0, 1, list_to_show, 'critic_loss')
            if 'actor_loss' in self.data_to_plot:
                steps = len(self.data_to_plot['actor_loss'])
                list_to_show = moving_average(self.data_to_plot['actor_loss'], steps / 100)
                plot_graph(self.ax, 1, 0, list_to_show, 'actor_loss')
                # plot_graph(self.ax, 1, 0, self.data_to_plot['actor_loss'], 'actor_loss')
            if 'rewards' in self.data_to_plot:
                plot_graph(self.ax, 1, 1, self.data_to_plot['rewards'], 'Rewards')
            if 'action' in self.data_to_plot:
                plot_graph(self.ax, 1, 1, self.data_to_plot['action'], 'action')
            if 'kl_div_actor' in self.data_to_plot:
                plot_graph(self.ax, 1, 1, self.data_to_plot['kl_div_actor'], 'kl_div_actor')
            if 'kl_div_critic' in self.data_to_plot:
                plot_graph(self.ax, 1, 1, self.data_to_plot['kl_div_critic'], 'kl_div_critic', color='red', cla=False)

            # X = self.data_to_plot['obs1']
            # Y = self.data_to_plot['obs2']
            # Z = self.data_to_plot['critic_values']
            # self.ax[1, 1].cla()
            # self.ax[1, 1] = self.fig.add_subplot(2,2,4, projection='3d')
            # self.ax[1, 1].plot_surface(X, Y, Z, linewidth=0, antialiased=False)

            plt.pause(0.05)

    def plot_summary(self):
        pass

    def neptune_init(self):
        if self.plot_neptune:
            self.run = neptune.init(project='1919ars/MountainCarDDPG', api_token=os.environ['NEPTUNE_API_TOKEN'],
                                    tags=self.tags, name=f'{self.name}_mountainCar',
                                    # source_files=['alg_constrants_amd_packages.py'],
                                    )

    def neptune_set_parameters(self, params_dict):
        self.run['parameters'] = params_dict

    def neptune_update(self, update_dict: dict):
        if self.plot_neptune:
            for k, v in update_dict.items():
                self.run[k].log(v)
                # self.run[k].log(f'{v}')

    def close(self):
        if NEPTUNE:
            self.run.stop()
        if PLOT_LIVE:
            plt.close()

    @staticmethod
    def logging_init():
        # logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        # logging.basicConfig(level=logging.DEBUG)
        pass

    def info(self, message, print_info=True, end='\n'):
        # logging.info('So should this')
        if print_info:
            print(colored(f'~[INFO]: {message}', 'green'), end=end)

    def debug(self, message, print_info=True, end='\n'):
        # logging.debug('This message should go to the log file')
        if print_info:
            print(colored(f'~[DEBUG]: {message}', 'cyan'), end=end)

    def warning(self, message, print_info=True, end='\n'):
        # logging.warning('And this, too')
        if print_info:
            print(colored(f'\n~[WARNING]: {message}', 'yellow'), end=end)

    def error(self, message):
        # logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
        raise RuntimeError(f"~[ERROR]: {message}")


# plotter = ALGPlotter(
#     plot_life=PLOT_LIVE,
#     plot_neptune=NEPTUNE,
#     name='my_run'
# )

