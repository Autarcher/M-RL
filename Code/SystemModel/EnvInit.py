from Code.SystemModel.TaskModel import TaskNode
from Code.SystemModel.TaskDAG import TaskDAG
from Code.SystemModel.DeviceModel import Device
import argparse
import random

class RLState:
    def __init__(self, device_list, app_list, cur_time):
        """
        Build a Reinforcement Learning State
        :param device_list: the list of devices
        :param app_list: the list of applications
        :param pending_tasks: the list of pending tasks
        :param ready_tasks: the list of ready tasks
        :param cur_time: the current time slot
        """
        self.device_list = device_list
        self.app_list = app_list
        """
        pending_tasks: 每个元素都是是一个五元组，包括任务节点本身,任务所属设备，任务的到达时间，任务在AppDAG图内的ID和任务所属的AppID
        ready_tasks: 每个元素是一个五元组，包括任务节点本身,任务所属设备，任务的到达时间，任务在AppDAG图内的ID和任务所属的AppID
        """
        self.pending_tasks = []  #根据app_list计算等待任务，依赖任务没有全部完成。
        self.ready_tasks = [] #根据app_list计算就绪任务，依赖任务都已经完成
        self.cur_time = cur_time

        self._update_tasks() #初始化 pending_tasks和ready_tasks




    def _update_tasks(self):
        """
        更新 ready_tasks 和 pending_tasks 列表
        """
        for appID, task_tuple in enumerate(self.app_list):
            app_DAG, app_type, app_device, arriving_time = task_tuple

            # 遍历所有的 TaskNode
            for node_id, task_node in app_DAG.nodes:
                if not app_DAG.task_node_finished_flag[node_id]:  # 任务未完成
                    # 获取所有前置依赖的任务 ID
                    dependencies = app_DAG.adjacency_list[node_id]

                    # 检查所有依赖任务是否都已完成
                    all_dependencies_finished = all(app_DAG.task_node_finished_flag[dep] for dep in dependencies)

                    # 如果任务已经在Running_tasks队列但还未完成则不进入ready_tasks队列
                    if all_dependencies_finished and app_DAG.task_node_scheduled_flag[node_id] == False:
                        # 所有依赖任务完成，将任务添加到 ready_tasks 中
                        self.ready_tasks.append((task_node, app_device, arriving_time, node_id, appID))
                    elif all_dependencies_finished == False:
                        # 如果任务有未完成的依赖任务，放入 pending_tasks
                        self.pending_tasks.append((task_node, app_device, arriving_time, node_id, appID))

    def generate_actions(self):
        """
        生成与 ready_tasks 一一对应的动作数组
        :return: list of actions corresponding to ready_tasks
        """
        actions = []

        for index, task in enumerate(self.ready_tasks):
            task_node, app_device, arriving_time, node_id, appID = task
            # 创建一个与 ready_tasks 对应的动作，假设动作是将任务分配给设备
            action = {
                'node_id': node_id,  # 表示在 DAG 图里面的编号
                'appID': appID,      # 所属 DAG 图编号
                'device': app_device,  # DAG 图所属设备编号
                'action_index': index  # 任务在 ready_tasks 中的编号
            }
            actions.append(action)

        return actions

    def print_actions(self, actions):
        """
        打印动作数组
        :param actions: list of actions to be printed
        """
        for action in actions:
            print(f"Action - node_id: {action['node_id']}, appID: {action['appID']}, device: {action['device']}, action_index: {action['action_index']}")

    def __repr__(self):
        return (f"RLState: cur_time={self.cur_time}, "
                f"pending_tasks={self.pending_tasks}, ready_tasks={self.ready_tasks}, "
                f"app_list={self.app_list}")
class EnvInit:
    def __init__(self, config_path=None):
        """
        初始化环境配置。
        :param config_path: 配置文件的路径（可选）。
        """
        self.config_path = config_path
        self.config = None
        self.devices = []       #其中每个元素是一个三元组，第一个元素是Device类，第二个元素是Device的类别，第三个元素表示Device的ID
        self.tasks = []         #其中每个元素是一个四元组，第一个元素是任务的DAG图，第二个元素表示任务种类，第三个元素表示任务的所属设备，第四个变量表示为任务的到达时间
        self.running_tasks = [] #每个元素是一个五元组，包括任务节点本身,任务执行的设备，任务的完成执行时间，任务在AppDAG图内的ID和任务所属的AppID
        self.current_time = 0   #系统初始参考时间是0

        self.t_add_runnings_tasks_num = 0
        self.t_finished_tasks_num = 0
        self.t_finished_apps_num = 0

    def env_clear(self):
        # self.devices = []
        self.tasks = []
        self.running_tasks = []
        self.current_time = 0


        self.t_add_runnings_tasks_num = 0
        self.t_finished_tasks_num = 0
        self.t_finished_apps_num = 0



    def clear_t_record(self):
        self.t_add_runnings_tasks_num = 0
        self.t_finished_tasks_num = 0
        self.t_finished_apps_num = 0


    def load_config(self):
        """
        加载配置文件。
        """
        if self.config_path:
            # 在此处加载配置文件，例如 JSON 或 YAML
            pass

    def setup_environment(self):
        """
        根据配置设置环境变量或其他初始化工作。
        """
        # 这里可以包括设置环境变量、初始化日志等操作
        pass

    def check_requirements(self):
        """
        检查运行环境的必要条件是否满足。
        """
        # 例如检查是否安装了特定的库或工具
        pass

    def initialize_devices(self, device_type, args):
        """
        根据设备类型初始化设备实例。
        :param device_type: 设备类型。
        :param args: 设备参数。
        """
        if device_type == 'device_type1':
            self.devices.append((Device(tuple(args.coordinates1), args.computing_speed1, args.channel_gain1,
                                        available_time=args.available_time1), device_type, len(self.devices) + 1))
        elif device_type == 'device_type2':
            self.devices.append((Device(tuple(args.coordinates2), args.computing_speed2, args.channel_gain2,
                                        available_time=args.available_time2), device_type, len(self.devices) + 1))
        elif device_type == 'device_type3':
            self.devices.append((Device(tuple(args.coordinates3), args.computing_speed3, args.channel_gain3,
                                        available_time=args.available_time3), device_type, len(self.devices) + 1))

    # def initialize_task_dag(self, task_type, args):
    #     """
    #     初始化任务DAG。
    #     :param task_type: DAG类型。
    #     :param args: 任务DAG参数。
    #     """
    #     if task_type == 'task_type1':
    #         task_dag = TaskDAG(num_nodes=args.num_nodes1, data_range=args.data_range,
    #                            computation_range=args.computation_range,
    #                            deadline_range=args.deadline_range, seed=args.seed1)
    #     elif task_type == 'task_type2':
    #         task_dag = TaskDAG(num_nodes=args.num_nodes2, data_range=args.data_range,
    #                            computation_range=args.computation_range,
    #                            deadline_range=args.deadline_range, seed=args.seed2)
    #     elif task_type == 'task_type3':
    #         task_dag = TaskDAG(num_nodes=args.num_nodes3, data_range=args.data_range,
    #                            computation_range=args.computation_range,
    #                            deadline_range=args.deadline_range, seed=args.seed3)
    #
    #     arrival_time = random.randint(0, 100)  # 随机生成任务到达时间
    #     self.tasks.append((task_dag, task_type, 1, arrival_time))  # 第一个元素是任务的DAG图，第二个元素表示任务种类，第三个元素表示任务的所属设备，第四个变量表示为任务的到达时间

    def initialize(self, args):
        """
        运行初始化流程。
        :param args: 设备参数。
        """
        self.load_config()
        self.setup_environment()
        self.check_requirements()

    def get_state(self):
        state = RLState(self.devices, self.tasks, self.current_time)

        return state


    def update_running_tasks(self, state, action_index):

        if -1 <= action_index < len(state.ready_tasks):
            task_node, app_device, arriving_time, node_id, appID = state.ready_tasks[action_index]
            # 在 tasks 列表中找到对应的 TaskDAG

            task_dag, task_type, task_device, task_arrival_time = self.tasks[appID]

            #卸载任务优先级最高的一个任务  这里要记得计算任务的响应时间
            running_time, offload_device = self.devices[app_device][0].offload_task(state.ready_tasks[action_index], self.devices)
            self.running_tasks.append((task_node, offload_device, state.cur_time + running_time, node_id, appID))
            task_dag.task_node_scheduled_flag[node_id] = True
            task_dag.task_node_scheduled_seq.append(node_id)
            self.t_add_runnings_tasks_num += 1
            print("执行的任务")
            print(f"task_node: {task_node}\n"
                  f"app_device: {app_device}\n"
                  f"arriving_time: {arriving_time}\n"
                  f"offload_device: {offload_device}\n"
                  f"finish_time: {state.cur_time + running_time}\n"
                  f"node_id: {node_id}\n"
                  f"appID: {appID}")
            print("------------------")
        else:
            print("Invalid action index.")

        return self.running_tasks
    def get_next_state_and_reward(self, state, action):
        """
        根据动作更新环境的状态
        :param action_index: index of the action in the ready_tasks list
        """
        self.update_running_tasks(state, action)


        for running_task in self.running_tasks:
            if running_task[2] <= self.current_time:
                task_node, _, finished_time, node_id, appID = running_task
                self.tasks[appID][0].task_node_finished_flag[node_id] = True
                self.tasks[appID][0].task_node_finished_time[node_id] = finished_time
                self.t_finished_tasks_num += 1

                #如果时出口任务那么记录这个任务的结束时间
                if node_id == len(self.tasks[appID][0].nodes) - 1:
                    self.tasks[appID][0].app_finished_time = finished_time
                    self.t_finished_apps_num += 1

                self.running_tasks.remove(running_task)
        state = self.get_state()

        #得到奖励
        if action >= -1:
            reward = (self.t_finished_apps_num * 10) + (self.t_finished_tasks_num * 1) + (
                    self.t_add_runnings_tasks_num * 1)
        else:
            reward = -1

        return state, reward



    def print_device_info(self):
        print('**** Device Information ****')
        """
        打印所有设备的信息。
        """
        for device, device_type, device_id in self.devices:
            print('---- Device Start ----')
            print(f"Device ID: {device_id}, Device Type: {device_type}")
            device.print_device_info()
            print('---- Device End ----')





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
    parser.add_argument('--learning_rate', type=float, default=0.000005)
    parser.add_argument('--n_step', type=int, default=600)
    parser.add_argument('--n_time_slot', type=int, default=300)
    parser.add_argument('--max_softmax_beta', type=int, default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128, help='dimension of latent layers')

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
    device0.initialize_task_dag('task_type1', args, env)
    device0.initialize_task_dag('task_type2', args, env)
    device0.initialize_task_dag('task_type3', args, env)


    # # 打印设备信息
    # env.print_device_info()
    #
    # # 打印DAG图的邻接表
    # print('**** Task DAG Information ****')
    # for task in env.tasks:
    #     task_dag, task_type, device, arrival_time = task
    #     print(f"任务DAG种类: {task_type}, 所属设备: {device}, 到达时间: {arrival_time}")
    #     print('---- Task DAG Start ----')
    #     task_dag.print_adjacency_list()
    #     print('---- Task DAG End ----')


    #初始化state
    state = env.get_state()



    # for task_tuple in state.app_list:
    #     app_DAG, app_type, app_device, arriving_time = task_tuple
    #     print("***************************************************")
    #     print("app种类" + str(app_type))
    #     print("所属设备"+str(app_device))
    #     print("到达时间"+str(arriving_time))
    #     app_DAG.print_adjacency_list()
    #     print(app_DAG.task_node_finished_flag)
    #
    # print("-----------------------------------------------------------")
    #
    # for task in state.ready_tasks:
    #     task_node, app_device, arriving_time, node_id, appID = task
    #     print("所属设备"+str(app_device))
    #     print("到达时间"+str(arriving_time))
    #     print("任务ID:" + str(node_id))
    #     print("所属AppID:" + str(appID))
    #     task_node.print_task_info()

    # print("--------------------------------------------------------------------")
    actions = state.generate_actions()
    state.print_actions(actions)
    print("------------------------------测试get_next_state函数--------------------------------------")
    # env.update_running_tasks(state, 0)
    # state = env.get_next_state()
    # env.update_running_tasks(state, 0)
    # env.update_running_tasks(state, 1)
    for i in range(200):
        # 假设这个部分在每秒的循环中执行
        #任务随即到达
        if random.random() < 0.1:  # 10% 的概率
            device0.initialize_task_dag('task_type1', args, env)

        if random.random() < 0.1:  # 10% 的概率
            device0.initialize_task_dag('task_type2', args, env)

        if random.random() < 0.1:  # 10% 的概率
            device0.initialize_task_dag('task_type3', args, env)

        if actions:
            # random.seed(args.seed)
            random_action = random.randint(0, len(actions) - 1)
            env.update_running_tasks(state, random_action)
        else:
            action = -2
        state, reward = env.get_next_state_and_reward(state, random_action) #这个里面会判断app是否已经完成
        actions = state.generate_actions()
        # env.update_running_tasks(state, 0)
        for task_tuple in state.app_list:
            app_DAG, app_type, app_device, arriving_time = task_tuple
            print("*************************+++++++++其中一个任务++++++++**************************")
            print("app种类" + str(app_type))
            print("所属设备"+str(app_device))
            print("到达时间"+str(arriving_time))
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
            print("所属设备"+str(app_device))
            print("到达时间"+str(arriving_time))
            print("任务ID:" + str(node_id))
            print("所属AppID:" + str(appID))
            task_node.print_task_info()
            print("***************************************************")
            actions = state.generate_actions()
            state.print_actions(actions)
        env.current_time += 1
        print(f"+++++++++++++++++++current_time:{env.current_time}+++++++++++++++++++++++")

        print(f"+++++++++++++++++++------c-------+++++++++++++++++++++++")
        actions = state.generate_actions()
        explore_step = 1
        if env.current_time >= 0 :
            # reward = (env.t_finished_apps_num * 10) + (env.t_finished_tasks_num * 1) + (
            #             env.t_add_runnings_tasks_num * 1)
            print(f"第{env.current_time / explore_step}次完成探索时的奖励：{reward} ")
            env.clear_t_record()
    # print("************************执行任务***************************")
    # state = env.get_next_state(state, 0)
    # for task in state.ready_tasks:
    #     task_node, app_device, arriving_time, node_id, appID = task
    #     print("所属设备"+str(app_device))
    #     print("到达时间"+str(arriving_time))
    #     print("任务ID:" + str(node_id))
    #     print("所属AppID:" + str(appID))
    #     task_node.print_task_info()
    # print("***************************************************")
    #
    # for task_tuple in state.app_list:
    #     app_DAG, app_type, app_device, arriving_time = task_tuple
    #     print("***************************************************")
    #     print("app种类" + str(app_type))
    #     print("所属设备"+str(app_device))
    #     print("到达时间"+str(arriving_time))
    #     app_DAG.print_adjacency_list()
    #     print(app_DAG.task_node_finished_flag)
    #     print(app_DAG.task_node_finished_time)
    # actions = state.generate_actions()
    # state.print_actions(actions)
    #Exploration setp
    print("*******************************测试get_next_state函数完毕*******************************")


    sum_time = 0
    finished_tasks_num = 0
    for index, app in enumerate(env.tasks):
        appDAG, _, _, arriving_time = app
        print(f"第{index}个任务的到达时刻是{arriving_time}的完成时刻是{appDAG.app_finished_time},"
              f"响应时间(任务完成时间-任务到达时间)是{appDAG.app_finished_time - arriving_time}")
        if appDAG.app_finished_time > 0:
            sum_time += appDAG.app_finished_time - arriving_time
            finished_tasks_num += 1
    print(f"完成的任务数{finished_tasks_num}")
    print(f"平均响应时间:{sum_time/finished_tasks_num}")

