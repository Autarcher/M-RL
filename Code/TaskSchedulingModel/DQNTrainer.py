from Code.SystemModel.TaskModel import TaskNode
from Code.SystemModel.TaskDAG import TaskDAG
from Code.SystemModel.DeviceModel import Device
from Code.SystemModel.EnvInit import  RLState,EnvInit
from Code.TaskSchedulingModel.BrainDQN import BrainDQN
from Code.TaskSchedulingModel.util.prioritized_replay_memory import  PrioritizedReplayMemory
import numpy as np
import argparse

#  definition of constants
MEMORY_CAPACITY = 50
GAMMA = 1
STEP_EPSILON = 5000.0
UPDATE_TARGET_FREQUENCY = 500
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
MAX_BETA = 10
MIN_VAL = -1000000
MAX_VAL = 1000000

class Trainer:
    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """

        self.args = args
        np.random.seed(self.args.seed)
        # self.n_action = args.n_task * args.n_device * (args.n_device + args.n_h_max)

        self.num_node_feats = args.num_node_feats
        self.reward_scaling = 0.001

        self.brain = BrainDQN(self.args, self.num_node_feats)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)


        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_node_feat: %d" % self.num_node_feats)
        print("***********************************************************")

    def run_training(self, env, state):
        """
        Run de main loop for training the model
        """
        #  Generate a random instance
        if self.args.plot_training:
            iter_list = []
            reward_list = []

        self.initialize_memory(env)
        print('[INFO]', 'iter', 'time', 'avg_reward_learning', 'loss', "beta")

        cur_best_reward = MIN_VAL

        for i in range(self.args.n_episode):

            loss, beta = self.run_episode(i, False, env)

            #  We first evaluate the validation step every 10 episodes, until 100, then every 100 episodes.
            if (i % 10 == 0 and i < 101) or i % 100 == 0:

                avg_reward = 0.0
                for j in range(self.validation_len):
                    avg_reward += self.evaluate_instance(j, env)

                avg_reward = avg_reward / self.validation_len

                cur_time = round(time.time() - start_time, 2)

                print('[DATA]', i, cur_time, avg_reward, loss, beta)

                sys.stdout.flush()

                if self.args.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()

                    plt.plot(iter_list, reward_list, linestyle="-", label="DQN", color='y')

                    plt.legend(loc=3)
                    out_fig_file = '%s/training_curve_reward.png' % self.args.save_dir
                    out_csv_file = '%s/training_data.csv' % self.args.save_dir
                    if not os.path.exists(self.args.save_dir):
                        os.makedirs(self.args.save_dir)
                    plt.savefig(out_fig_file)

                    # save the data
                    with open(out_csv_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        # 写入标题
                        writer.writerow(['Iteration', 'Reward'])
                        # 写入数据
                        for i, reward in zip(iter_list, reward_list):
                            writer.writerow([i, reward])

                fn = "iter_%d_model.pth.tar" % i

                #  We record only the model that is better on the validation set wrt. the previous model
                #  We nevertheless record a model every 10000 episodes
                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(folder=self.args.save_dir, filename=fn)
                elif i % 10000 == 0:
                    self.brain.save(folder=self.args.save_dir, filename=fn)

    def initialize_memory(self, env):
        """
        Initialize the replay memory with random episodes and a random selection
        """

        while self.init_memory_counter < MEMORY_CAPACITY:
            self.run_episode(0, True, env)

        print("[INFO] Memory Initialized")




    def data_initialization(self, state, env):
        """
        Initialize data for training.

        Parameters:
        state (State): The current state of the environment.
        env (Environment): The environment instance.
        """
        task_features = self.aggregate_task_features(state)

    def aggregate_appDAG_task_features(self, state, env, app):
        app_DAG, app_type, app_device, arriving_time = app
        task_features = []
        for node_id in range(app_DAG.num_nodes):
            task_node = app_DAG.nodes[node_id][1]
            task_device = env.devices[app_device][0]
            arr = [
                task_node.data_size,
                task_node.computation_size,
                task_node.deadline,
                app_DAG.in_degree[node_id],
                app_DAG.out_degree[node_id],
                arriving_time,
                node_id,
                app_type,
                task_device.coordinates[0],
                task_device.coordinates[1],
                task_device.computing_speed,
                task_device.channel_gain,
                task_device.available_time,
            ]
            task_features.append(arr)
        return task_features

    def print_task_features(self, task_features):
        """
        Print the task features.

        Parameters:
        task_features (list): A list of task features to print.
        """
        print("Task Features:")
        for index, features in enumerate(task_features):
            print(f"Task {index + 1}: {features}")

        print(f"Task features shape: (1,{len(task_features[0])})")


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Device parameters for device type 1
    parser.add_argument('--device_type1', type=int, default=1)
    parser.add_argument('--coordinates1', default=[0, 0])
    parser.add_argument('--computing_speed1', type=float, default=1e9)
    parser.add_argument('--channel_gain1', type=float, default=0.85)
    parser.add_argument('--available_time1', type=float, default=0.0)

    # Device parameters for device type 2
    parser.add_argument('--device_type2', type=int, default=2)
    parser.add_argument('--coordinates2', default=[10, 20])
    parser.add_argument('--computing_speed2', type=float, default=2e9)
    parser.add_argument('--channel_gain2', type=float, default=0.9)
    parser.add_argument('--available_time2', type=float, default=0.0)

    # Device parameters for device type 3
    parser.add_argument('--device_type3', type=int, default=3)
    parser.add_argument('--coordinates3', default=[30, 40])
    parser.add_argument('--computing_speed3', type=float, default=1.5e9)
    parser.add_argument('--channel_gain3', type=float, default=0.8)
    parser.add_argument('--available_time3', type=float, default=0.0)

    # TaskDAG parameters for dag type 1
    parser.add_argument('--task_type1', type=int, default=1, help="DAG图的种类")
    parser.add_argument('--seed1', type=int, default=1, help="随机种子")
    parser.add_argument('--num_nodes1', type=int, default=5, help="DAG图的节点数量")

    # TaskDAG parameters for dag type 2
    parser.add_argument('--task_type2', type=int, default=2, help="DAG图的种类")
    parser.add_argument('--seed2', type=int, default=2, help="随机种子")
    parser.add_argument('--num_nodes2', type=int, default=7, help="DAG图的节点数量")

    # TaskDAG parameters for dag type 3
    parser.add_argument('--task_type3', type=int, default=3, help="DAG图的种类")
    parser.add_argument('--seed3', type=int, default=3, help="随机种子")
    parser.add_argument('--num_nodes3', type=int, default=12, help="DAG图的节点数量")

    # Common TaskDAG parameters
    parser.add_argument('--data_range', type=tuple, default=(100, 1000), help="数据量范围：100字节到1000字节")
    parser.add_argument('--computation_range', type=tuple, default=(1e8, 1e10), help="计算量范围：1亿FLOPs到100亿FLOPs")
    parser.add_argument('--deadline_range', type=tuple, default=(100, 1000), help="截止时间范围：100秒到1000秒")

    # Hyper parameters
    parser.add_argument('--seed', type=int, default=1, help="模型的初始化种子")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.000005)
    parser.add_argument('--n_step', type=int, default=600)
    parser.add_argument('--n_time_slot', type=int, default=300)
    parser.add_argument('--max_softmax_beta', type=int, default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64, help='dimension of latent layers')
    parser.add_argument('--num_node_feats', type=int, default=13, help="The features dimension of a node")

    # Argument for Trainer
    parser.add_argument('--n_episode', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./result-default')
    parser.add_argument('--plot_training', type=int, default=1)
    parser.add_argument('--mode', default='gpu', help='cpu/gpu')

    return parser.parse_args()


# 使用示例
if __name__ == "__main__":
    args = parse_arguments()
    env = EnvInit(config_path="config.yaml")
    env.initialize(args=args)
    trainer = Trainer(args)

    # 初始化三个设备每类设备各一种
    env.initialize_devices('device_type1', args)
    # env.initialize_devices('device_type1', args)
    env.initialize_devices('device_type2', args)
    # env.initialize_devices('device_type2', args)
    # env.initialize_devices('device_type3', args)
    env.initialize_devices('device_type3', args)

    # device0发出三个任务请求初始化三个任务DAG每类DAG各一种
    device0 = env.devices[0][0]
    device0.initialize_task_dag('task_type1', args, env)



    #初始化state
    state = env.get_state()
    actions = state.generate_actions()
    state.print_actions(actions)
    env.update_running_tasks(state, 0)
    state = env.get_next_state()
    actions = state.generate_actions()
    for task_tuple in state.app_list:
        app_DAG, app_type, app_device, arriving_time = task_tuple
        print("*************************+++++++++其中一个任务++++++++**************************")
        print("app种类" + str(app_type))
        print("所属设备" + str(app_device))
        print("到达时间" + str(arriving_time))
        app_DAG.print_adjacency_list()
        print("scheduled flag:")
        print(app_DAG.task_node_scheduled_flag)
        print("finished flag:")
        print(app_DAG.task_node_finished_flag)
        print("finished time:")
        print(app_DAG.task_node_finished_time)
        print("scheduled seq:")
        print(app_DAG.task_node_scheduled_seq)
        print("app finished time")
        print(app_DAG.app_finished_time)

    print("***************************************************")
    for task in state.ready_tasks:
        task_node, app_device, arriving_time, node_id, appID = task
        print("所属设备" + str(app_device))
        print("到达时间" + str(arriving_time))
        print("任务ID:" + str(node_id))
        print("所属AppID:" + str(appID))
        task_node.print_task_info()
        print("***************************************************")
        actions = state.generate_actions()
        state.print_actions(actions)


    task_features = trainer.aggregate_appDAG_task_features(state, env, env.tasks[0])

    trainer.print_task_features(task_features)

    print("*******************************测试aggregate_appDAG_task_features函数完毕*******************************")