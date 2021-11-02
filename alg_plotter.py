from alg_constrants_and_packages import *
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
    def __init__(self, plot_life=True, plot_neptune=False):

        self.plot_life = plot_life
        self.plot_neptune = plot_neptune
        self.fig, self.actor_losses, self.critic_losses, self.ax, self.agents_list = {}, {}, {}, {}, {}
        self.total_reward, self.val_total_rewards = [], []

        self.neptune_init()
        self.logging_init()

        if self.plot_life:
            self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
            self.data_to_plot = {}

        self.info("ALGPlotter instance created.")

    def plots_set(self, env_module):
        if self.plot_life:
            self.agents_list = env_module.get_agent_list()
            self.actor_losses = {agent: deque(maxlen=100) for agent in self.agents_list}
            self.critic_losses = {agent: deque(maxlen=100) for agent in self.agents_list}

    def plots_update_data(self, data_dict):
        if self.plot_life:
            for key_name, value in data_dict.items():
                if key_name not in self.data_to_plot:
                    self.data_to_plot[key_name] = deque(maxlen=100)
                self.data_to_plot[key_name].append(value)

    def plots_online(self):
        # plot live:
        if self.plot_life:
            def plot_graph(ax, indx_r, indx_c, list_of_values, label, color='b'):
                ax[indx_r, indx_c].cla()
                ax[indx_r, indx_c].plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                # ax[indx_r, indx_c].set_title(f'Plot: {label}')
                ax[indx_r, indx_c].set_xlabel('iters')
                ax[indx_r, indx_c].set_ylabel(f'{label}')

            def plot_graph_axes(axes, list_of_values, label, color='b'):
                axes.cla()
                axes.plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                # axes.set_title(f'Plot: {label}')
                axes.set_xlabel('iters')
                axes.set_ylabel(f'{label}')

            counter = 0
            for key_name, list_of_values in self.data_to_plot.items():
                plot_graph_axes(self.fig.axes[counter], list_of_values, key_name)
                counter += 1

            # plot_graph(self.ax, 0, 1, self.val_total_rewards, 'val_rewards')

            plt.pause(0.05)

    def plot_summary(self):
        pass

    def neptune_init(self):
        if self.plot_neptune:
            self.run = neptune.init(project='1919ars/PettingZoo',
                                    tags=['DDPG'],
                                    name=f'DDPG_{time.asctime()}',
                                    # source_files=['alg_constrants_amd_packages.py'],
                                    )
            # Neptune.ai Logger
            PARAMS = {
                # 'GAMMA': GAMMA,
                # 'LR': LR,
                # 'CLIP_GRAD': CLIP_GRAD,
                # 'MAX_STEPS': MAX_STEPS,
            }
            self.run['parameters'] = PARAMS
        else:
            self.run = {}

    def neptune_update(self, update_dict: dict):
        if self.plot_neptune:
            for k, v in update_dict.items():
                self.run[k].log(v)
                self.run[k].log(f'{v}')

    def stop(self):
        self.run.stop()
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


plotter = ALGPlotter(
    plot_life=PLOT_LIVE,
    plot_neptune=NEPTUNE
)