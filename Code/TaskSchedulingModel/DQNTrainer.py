from Code.SystemModel.EnvInit import EnvInit
from Code.TaskSchedulingModel.BrainDQN import BrainDQN
from Code.TaskSchedulingModel.util.prioritized_replay_memory import  PrioritizedReplayMemory
import torch
import argparse
import random
import os
import numpy as np

import dgl




#  definition of constants
MEMORY_CAPACITY = 300   #300
GAMMA = 1
STEP_EPSILON = 5000.0
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
MAX_BETA = 10
MIN_VAL = -1000000
MAX_VAL = 1000000


#测试训练轮次的参数
UPDATE_TARGET_FREQUENCY = 20 #每学习多少次模型传递依次参数  # 20

EPISODE = 10000        #1000

SYSTEM_MAX_TIMES = 200 #200

LEARNING_STEPS = 5  #5



# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """
        self.n_step = args.n_step
        self.steps_done = 0
        self.args = args
        self.app_type_dict ={
            'task_type1': 1,
            'task_type2': 2,
            'task_type3': 3
        }
        np.random.seed(self.args.seed)
        # self.n_action = args.n_task * args.n_device * (args.n_device + args.n_h_max)

        self.num_node_feats = args.num_node_feats
        self.reward_scaling = 0.001

        self.brain = BrainDQN(self.args, self.num_node_feats, self.num_node_feats)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)


        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_node_feat: %d" % self.num_node_feats)
        print("***********************************************************")

    def run_training(self, env):
        """
        Run de main loop for training the model
        """
        #  Generate a random instance
        # state.print_actions(actions)

        self.initialize_memory(env)



        print("------------------------------测试强化学习训练开始--------------------------------------")
        # env.update_running_tasks(state, 0)
        # state = env.get_next_state()
        # env.update_running_tasks(state, 0)
        # env.update_running_tasks(state, 1)
        for i in range(EPISODE):
            print(f"++++++++++++++++++++++-------------第{i+1}次(系统共运行{SYSTEM_MAX_TIMES}时间)训练开始-------------++++++++++++++++++")

            total_reward = self.training_round(env)


            sum_time = 0
            finished_tasks_num = 0
            for index, app in enumerate(env.tasks):
                appDAG, _, _, arriving_time = app
                print(f"第{index}个任务的到达时刻是{arriving_time}的完成时刻是{appDAG.app_finished_time},"
                    f"响应时间(任务完成时间-任务到达时间)是{appDAG.app_finished_time - arriving_time}")
                if appDAG.app_finished_time > 0:
                    sum_time += appDAG.app_finished_time - arriving_time
                    finished_tasks_num += 1
            print(f"++++++++++++++++++++++++++++++++第{i+1}次(系统共运行{SYSTEM_MAX_TIMES}时间)训练结果++++++++++++++++++++++++++++")
            if finished_tasks_num == 0:
                print(f"完成的任务数{finished_tasks_num}")
                print(f"平均响应时间:-1(没有任务完成)")
                print(f"总奖励为:{total_reward}")
            else:
                print(f"完成的任务数{finished_tasks_num}")
                print(f"平均响应时间:{sum_time / finished_tasks_num}")
                print(f"总奖励为:{total_reward}")

            #记录结果
            # 确保 result 文件夹存在
            result_folder = 'result'
            os.makedirs(result_folder, exist_ok=True)

            # 构造文件名
            file_path = os.path.join(result_folder, 'training_results.txt')

            with open(file_path, 'a') as result_file:
                result_file.write(
                    f"++++++++++++++++++++++++++++++++第{i+1}次(系统共运行{SYSTEM_MAX_TIMES}时间)训练结果++++++++++++++++++++++++++++\n")
                if finished_tasks_num == 0:
                    result_file.write(f"完成的任务数:{finished_tasks_num}\n")
                    result_file.write(f"平均响应时间:-1(没有任务完成)\n")
                    result_file.write(f"总奖励为:{total_reward}\n")
                else:
                    result_file.write(f"完成的任务数:{finished_tasks_num}\n")
                    result_file.write(f"平均响应时间:{sum_time / finished_tasks_num}\n")
                    result_file.write(f"总奖励为:{total_reward}\n")

            # 打印存储的文件的绝对路径
            print(f"结果文件已保存至: {os.path.abspath(file_path)}")


            env.env_clear()
            print(f"++++++++++++++++-------------第{i+1}次(系统共运行{SYSTEM_MAX_TIMES}时间)训练结束-------------++++++++++++++++++")


        return 0


    def training_round(self, env):
        device0 = env.devices[0][0]
        device0.initialize_task_dag('task_type1', args, env)
        device0.initialize_task_dag('task_type2', args, env)
        device0.initialize_task_dag('task_type3', args, env)
        device0.initialize_task_dag('task_type1', args, env)
        device0.initialize_task_dag('task_type2', args, env)
        device0.initialize_task_dag('task_type3', args, env)

        state = env.get_state()
        actions = state.generate_actions()

        total_reward = 0
        for i in range(SYSTEM_MAX_TIMES):
            print(f"---------------第{i+1}轮的强化学习开始----------------")
            # 假设这个部分在每秒的循环中执行

            # 任务随即到达
            seed = 4
            random.seed(seed)
            random1 = random.random()
            random2 = random.random()
            random3 = random.random()
            actions = state.generate_actions()
            actions_len = len(actions)
            count = 0
            if(actions_len <= 30):
                if random1 < 0.33:  # 10% 的概率
                    device0.initialize_task_dag('task_type1', args, env)
                    count = count + 1

                if random2 < 0.33:  # 10% 的概率
                    device0.initialize_task_dag('task_type2', args, env)
                    count = count + 1

                if random3 < 0.33:  # 10% 的概率
                    device0.initialize_task_dag('task_type3', args, env)
                    count = count + 1
                if count < 1:
                    device0.initialize_task_dag('task_type1', args, env)
                # print(f"+++到达任务数+++{count}")
                count = 0


            env.current_time += 1
            state = env.get_state()
            actions = state.generate_actions()

            print(f"+++++++++++++++++++current_time:{env.current_time}+++++++++++++++++++++++")
            loss, reward = self.run_episode(i, False, env)
            total_reward = total_reward + reward
            if i % (UPDATE_TARGET_FREQUENCY/LEARNING_STEPS) == 0:
                self.brain.update_target_model()
                print("成功更新网络参数")

            print(f"第{i+1}轮的强化学习loss:{loss}\n")
        return total_reward
    def select_action(self, action_graphs, action_nodeIDs, target=True):
        # 预测动作值
        predictions = self.brain.predict(action_graphs, action_nodeIDs, target)


        # 将 predictions 转换为 numpy 数组
        predictions = np.array(predictions)
        print("+++++++predictions+++++++")
        print(predictions)

        # 计算 softmax，将 predictions 转化为概率分布
        max_value = np.max(predictions)
        stabilized_predictions = predictions - max_value
        # print("+++++++stabilized_predictions+++++++")
        # print(stabilized_predictions)
        softmax_probs = np.exp(stabilized_predictions) / np.sum(np.exp(stabilized_predictions))

        print("+++++++softmax_probs+++++++")
        print(softmax_probs)
        # 根据 softmax 概率选择动作
        action = np.random.choice(len(predictions), p=softmax_probs)

        return action

    def get_state_feature(self, state):
        action_graphs = []
        action_nodeIDs = []

        actions = state.generate_actions()
        for index, action in enumerate(actions):
            task = env.tasks[action.get("appID")]
            action_nodeIDs.append(action.get("node_id"))
            # print(f"----------第{index}个ready------------")
            task_features, adj_matrix = trainer.aggregate_appDAG_task_features(state, env, task)
            # state, reward = env.get_next_state_and_reward(state, task.task_id)  # 使用 task_id 获取状态和奖励
            graph = trainer.make_nn_input(task_features, adj_matrix)
            action_graphs.append(graph)


            # trainer.print_task_features(task_features, adj_matrix)


            # print(graph)
        return (action_graphs, action_nodeIDs)


    def run_episode(self, episode_idx, memory_initialization, env):
        state = env.get_state()
        action_graphs = []
        action_nodeIDs = []
        actions = state.generate_actions()

        for index, action in enumerate(actions):
            task = env.tasks[action.get("appID")]
            action_nodeIDs.append(action.get("node_id"))

            task_features, adj_matrix = trainer.aggregate_appDAG_task_features(state, env, task)
            # state, reward = env.get_next_state_and_reward(state, task.task_id)  # 使用 task_id 获取状态和奖励
            graph = trainer.make_nn_input(task_features, adj_matrix)
            action_graphs.append(graph)

            # print(f"----------第{index}个ready------------")
            # trainer.print_task_features(task_features, adj_matrix)

            # print(graph)

        #得到action
        # print("+++++++select_action++++++++")
        # print(action_graphs)
        # print(action_nodeIDs)
        action = self.select_action(action_graphs, action_nodeIDs)
        print(f"选择的任务{action}")
        state_features = (action_graphs, action_nodeIDs)

        next_state, reward = env.get_next_state_and_reward(state, action)



        next_state_features = self.get_state_feature(next_state)
        # print("++++++++++++++++next_state_features++++++++++++")
        # print(next_state_features)


        sample = (state_features, action, reward, next_state_features)


        x, y, errors = self.get_targets([(0, sample, 0)])

        #可以学习多次
        for i in range(LEARNING_STEPS):
            loss = self.learning()
            print(f"第{episode_idx+1}轮的第{i+1}次学习的loss:{loss}")


        self.memory.random_add(errors, sample)
        return loss, reward


    def get_targets(self, batch):
        """
        Compute the TD-errors using the n-step Q-learning function and the model prediction
        :param batch: the batch to process
        :return: the state input, the true y, and the error for updating the memory replay
        """
        #这里的batch要改，因为我的每一个batch里的当sample有多个Graph
        batch_len = len(batch)
        # print("+++++++batch+++++++")
        # print(batch[0][1][1])

        graphs_list, nodeIDs_list = list(zip(*[e[1][0] for e in batch]))
        # graphs_list = [g for graph in graphs_list for g in (graph if isinstance(graph, list) else [graph])]

        # print("+++++++++graphs_list[0]++++++++")
        # print(graphs_list)
        # print("+++++++++nodeIDs_list[0]++++++++")
        # print(nodeIDs_list)

        # graphs_batch = dgl.batch(graphs_list)

        next_graphs_list, next_nodeIDs_list = list(zip(*[e[1][3] for e in batch]))
        # next_graphs_list = [g for graph in next_graphs_list for g in (graph if isinstance(graph, list) else [graph])]


        p = self.brain.predict(graphs_list[0], nodeIDs_list[0], target=False)


        #改为判断下一个状态的read_task数是0也就是没有任务

        p_ = self.brain.predict(next_graphs_list[0], next_nodeIDs_list[0], target=False)
        p_target_ = self.brain.predict(next_graphs_list[0], next_nodeIDs_list[0], target=True)
            # print("p_", p_)
            # print("p_target_", p_target_)


        x = []
        y = []
        errors = np.zeros(len(batch))

        for i in range(batch_len):

            sample = batch[i][1]
            state_graphs, state_nodeIDs = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state_graphs, next_state_nodeIDs = sample[3]

            q_value_prediction = p[action]

            #nextactions
            if len(next_state_nodeIDs) == 0:
                td_q_value = reward
                t = td_q_value

            else:
                # predictions = self.brain.model.predict(next_state_graphs, next_state_nodeIDs)
                # select_action_index = predictions.index(max(predictions))
                best_valid_next_action_id = p_.index(max(p_))
                td_q_value = reward + GAMMA * p_target_[best_valid_next_action_id]
                t = td_q_value

            state = (state_graphs, state_nodeIDs)
            x.append(state)
            y.append(t)

            errors[i] = abs(q_value_prediction - td_q_value)

        return x[0], y[0], errors[0]

    def learning(self):
        """
        execute a learning step on a batch of randomly selected experiences from the memory
        :return: the subsequent loss
        """


        # batch = self.memory.sample(self.args.batch_size)
        batch = self.memory.sample(1)

        x, y, errors = self.get_targets(batch)

        self.memory.update(batch[0][0],errors)
        #  update the errors
        # for i in range(len(batch)):
        #     idx = batch[i][0]
        #     self.memory.update(idx, errors[i])
        loss = self.brain.train(x, y)

        # print("--- learn_loss:", loss, "---")
        return round(loss, 4)


    def initialize_memory(self, env):
        """
        Initialize the replay memory with random episodes and a random selection
        """
        device0 = env.devices[0][0]
        device0.initialize_task_dag('task_type1', args, env)
        device0.initialize_task_dag('task_type2', args, env)
        device0.initialize_task_dag('task_type3', args, env)
        device0.initialize_task_dag('task_type1', args, env)
        device0.initialize_task_dag('task_type2', args, env)
        device0.initialize_task_dag('task_type3', args, env)
        # state = env.get_state()
        # actions = state.generate_actions()

        #任务到达的随机种子
        seed = 3
        for i in range(MEMORY_CAPACITY):
            # 假设这个部分在每秒的循环中执行
            #系统是忙的
            random.seed(seed)
            random1 = random.random()
            random2 = random.random()
            random3 = random.random()
            count = 0
            if random1 < 0.33:  # 10% 的概率
                device0.initialize_task_dag('task_type1', args, env)
                count = count + 1

            if random2 < 0.33:  # 10% 的概率
                device0.initialize_task_dag('task_type2', args, env)
                count = count + 1

            if random3 < 0.33:  # 10% 的概率
                device0.initialize_task_dag('task_type3', args, env)
                count = count + 1
            if count < 1:
                device0.initialize_task_dag('task_type1', args, env)
            # print(f"+++到达任务数+++{count}")
            count = 0

            # 得到action
            state = env.get_state()
            state_features = self.get_state_feature(state)
            actions = state.generate_actions()

            if actions:
                # random.seed(args.seed)
                random_action = random.randint(0, len(actions) - 1)
                action = random_action
                # action = 0
            else:
                action = -2

            next_state, reward = env.get_next_state_and_reward(state, action)  # 这个里面会判断app是否已经完成
            next_state_features = self.get_state_feature(state)
            actions = next_state.generate_actions()
            # print("++++++++++++++++next_state_features++++++++++++")
            # print(next_state_features)

            sample = (state_features, action, reward, next_state_features)


            error = abs(reward)
            self.memory.random_add(error, sample)
            env.current_time += 1

            explore_step = 1
            if env.current_time >= 0:
                # reward = (env.t_finished_apps_num * 10) + (env.t_finished_tasks_num * 1) + (
                #             env.t_add_runnings_tasks_num * 1)
                print(f"第{env.current_time / explore_step}次完成探索时的奖励：{reward} ")
                env.clear_t_record()
            print(f"+++++++++++++++++++current_time:{env.current_time}+++++++++++++++++++++++")

        init_loss = self.learning()  # learning procedure
        MAX_TIME = 1000
        running_time = 0
        while running_time >= MAX_TIME and (len(env.running_tasks) > 0):
            print(f"++++++++++current_time:{env.current_time}, current_running_tasks:{len(env.running_tasks)}++++++++++++")
            running_time += 1
            env.current_time += 1
            for running_task in env.running_tasks:
                if running_task[2] <= env.current_time:
                    task_node, _, finished_time, node_id, appID = running_task
                    env.tasks[appID][0].task_node_finished_flag[node_id] = True
                    env.tasks[appID][0].task_node_finished_time[node_id] = finished_time
                    env.t_finished_tasks_num += 1

                    # 如果时出口任务那么记录这个任务的结束时间
                    if node_id == len(env.tasks[appID][0].nodes) - 1:
                        env.tasks[appID][0].app_finished_time = finished_time
                        env.t_finished_apps_num += 1

                    env.running_tasks.remove(running_task)


        sum_time = 0
        finished_tasks_num = 0
        for index, app in enumerate(env.tasks):
            appDAG, _, _, arriving_time = app
            if(index%1 == 0):
                print(f"第{index}个任务的到达时刻是{arriving_time}的完成时刻是{appDAG.app_finished_time},"
                    f"响应时间(任务完成时间-任务到达时间)是{appDAG.app_finished_time - arriving_time}")
            if appDAG.app_finished_time > 0:
                sum_time += appDAG.app_finished_time - arriving_time
                finished_tasks_num += 1

        print(f"完成的任务数{finished_tasks_num}")
        if(finished_tasks_num == 0):
            print(f"平均响应时间: -1(无任务完成)")
        else:
            print(f"平均响应时间:{sum_time / finished_tasks_num}")

        env.env_clear()
        print(f"+++++初始化的one_learning loss:{init_loss}+++")
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
        #得到节点特征
        app_DAG, app_type, app_device, arriving_time = app
        task_features = []
        for node_id in range(app_DAG.num_nodes+2):
            task_node = app_DAG.nodes[node_id][1]
            task_device = env.devices[app_device][0]
            # task_ready_time = env.running_tasks[]
            arr = [
                task_node.data_size,
                task_node.computation_size,
                task_node.deadline,
                app_DAG.in_degree[node_id],
                app_DAG.out_degree[node_id],
                arriving_time,
                node_id,
                self.app_type_dict[app_type],
                task_device.coordinates[0],
                task_device.coordinates[1],
                task_device.computing_speed,
                task_device.channel_gain,
                task_device.available_time,
                task_node.is_end_task,
                task_node.is_start_task
                # task_ready_time
            ]
            task_features.append(arr)
        #得到邻接矩阵
        adjacency_list = app_DAG.adjacency_list
        num_nodes = len(adjacency_list)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for node, dependencies in adjacency_list.items():
            for dep in dependencies:
                adj_matrix[node][dep] = 1  # node 指向 dep
                adj_matrix[dep][node] = 1  # dep 也指向 node（无向）

        return task_features, adj_matrix

    def make_nn_input(self, task_features, adj_matrix):
        # 将任务特征从列表转换为 NumPy 的 ndarray
        if isinstance(task_features, list):
            task_features = np.array(task_features, dtype=np.float32)

        # 将 NumPy 矩阵转换为 PyTorch 的张量
        if isinstance(task_features, np.ndarray):
            task_features = torch.tensor(task_features, dtype=torch.float)

        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)

        # 创建 DGL 图对象，使用邻接矩阵
        edges = adj_matrix.nonzero(as_tuple=True)
        g = dgl.graph(edges)

        # 确保节点特征数量与图中的节点数一致
        if task_features.shape[0] != g.num_nodes():
            raise ValueError(f"节点特征数量 ({task_features.shape[0]}) 与图中的节点数量 ({g.num_nodes()}) 不匹配")

        # 将任务特征分配给节点
        g.ndata['n_feat'] = task_features

        # 创建边特征，与节点特征维度一致，并用1填充
        edge_features = torch.ones((g.num_edges(), task_features.shape[1]), dtype=torch.float)

        # 将边特征分配给边
        g.edata['e_feat'] = edge_features

        return g

    def print_task_features(self, task_features, adj_matrix):
        """
        Print the task features.

        Parameters:
        task_features (list): A list of task features to print.
        """
        print("Task Features:")
        for index, features in enumerate(task_features):
            print(f"Task {index}: {features}")

        print(f"Task features shape: (1,{len(task_features[0])})")
        print("邻接矩阵:\n")
        print(adj_matrix)


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
    #default=(1e8, 1e10)
    parser.add_argument('--computation_range', type=tuple, default=(1e6, 1e8), help="计算量范围：1亿FLOPs到100亿FLOPs")
    parser.add_argument('--deadline_range', type=tuple, default=(100, 1000), help="截止时间范围：100秒到1000秒")

    # Hyper parameters
    parser.add_argument('--seed', type=int, default=1, help="模型的初始化种子")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--n_step', type=int, default=600)
    parser.add_argument('--n_time_slot', type=int, default=300)
    parser.add_argument('--max_softmax_beta', type=int, default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64, help='dimension of latent layers')
    parser.add_argument('--num_node_feats', type=int, default=15, help="The features dimension of a node")

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
    device0.initialize_task_dag('task_type2', args, env)
    device0.initialize_task_dag('task_type3', args, env)
    device0.initialize_task_dag('task_type3', args, env)



    #初始化state
    state = env.get_state()
    actions = state.generate_actions()
    state.print_actions(actions)
    env.update_running_tasks(state, 0)
    state, reward = env.get_next_state_and_reward(state, -2)
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


    task_features, adj_matrix = trainer.aggregate_appDAG_task_features(state, env, env.tasks[0])

    state, reward = env.get_next_state_and_reward(state, 0)  # 这个里面会判断app是否已经完成
    graph = trainer.make_nn_input(task_features, adj_matrix)
    trainer.print_task_features(task_features, adj_matrix)
    # print("tensor_data:")
    print(graph)
    print("++++++++++++++run_episode++++++++++++++++++")
    # print(trainer.run_episode(1, False, env))
    # loss, _ = trainer.run_episode_test(1, True, env)
    # print(f"test_Loss:{loss}")
    print("*******************************测试aggregate_appDAG_task_features函数完毕*******************************")
    print("*******************************测试init_memory函数*******************************")
    env.env_clear()
    # trainer.initialize_memory(env)
    trainer.run_training(env)