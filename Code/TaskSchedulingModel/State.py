from Code.SystemModel.TaskDAG import TaskDAG
from Code.SystemModel.DeviceModel import Device
from Code.SystemModel.EnvInit import EnvInit
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

                    if all_dependencies_finished:
                        # 所有依赖任务完成，将任务添加到 ready_tasks 中
                        self.ready_tasks.append((task_node, app_device, arriving_time, node_id, appID))
                    else:
                        # 如果任务有未完成的依赖任务，放入 pending_tasks
                        self.pending_tasks.append((task_node, app_device, arriving_time, node_id, appID))
    def __repr__(self):
        return (f"RLState: cur_time={self.cur_time}, "
                f"pending_tasks={self.pending_tasks}, ready_tasks={self.ready_tasks}, "
                f"app_list={self.app_list}")


# Example usage
# devices = ['Device1', 'Device2']
# apps = ['App1', 'App2']
# state = RLState(devices, apps, ['Task2'], ['Task2'], 0)
# print(state)

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


if __name__ == "__main__":
    args = parse_arguments()
    env = EnvInit(config_path="config.yaml")
    env.initialize(args=args)
    # 初始化三个设备每类设备各一种
    env.initialize_devices('device_type1', args)
    # env.initialize_devices('device_type1', args)
    # env.initialize_devices('device_type2', args)
    # env.initialize_devices('device_type2', args)
    # env.initialize_devices('device_type3', args)
    # env.initialize_devices('device_type3', args)

    # 初始化三个任务DAG每类DAG各一种
    env.initialize_task_dag('task_type1', args)
    env.initialize_task_dag('task_type2', args)
    env.initialize_task_dag('task_type3', args)

    #测试任务0已经完成后是否可以找到正确的read_task
    env.tasks[0][0].task_node_finished_flag[0] = True


    state = RLState(env.devices, env.tasks, 0)

    # for device_tuple in state.device_list:
    #     device, device_type, index = device_tuple  # 解包元组
    #     device.print_device_info()

    for task_tuple in state.app_list:
        app_DAG, app_type, app_device, arriving_time = task_tuple
        print("***************************************************")
        print("app种类" + str(app_type))
        print("所属设备"+str(app_device))
        print("到达时间"+str(arriving_time))
        app_DAG.print_adjacency_list()
        print(app_DAG.task_node_finished_flag)

    print("-----------------------------------------------------------")
    state._update_tasks()
    for task in state.ready_tasks:
        task_node, app_device, arriving_time, node_id, appID = task
        print("所属设备"+str(app_device))
        print("到达时间"+str(arriving_time))
        print("任务ID:" + str(node_id))
        print("所属AppID:" + str(appID))
        task_node.print_task_info()