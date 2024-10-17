from Code.SystemModel.TaskDAG import TaskDAG
from Code.SystemModel.DeviceModel import Device
from Code.SystemModel.EnvInit import EnvInit
import argparse
import random
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


# 使用示例
if __name__ == "__main__":
    args = parse_arguments()
    env = EnvInit(config_path="config.yaml")
    env.initialize(args=args)

    # 初始化三个设备每类设备各一种
    env.initialize_devices('device_type1', args)
    env.initialize_devices('device_type1', args)
    env.initialize_devices('device_type2', args)
    env.initialize_devices('device_type2', args)
    env.initialize_devices('device_type3', args)
    env.initialize_devices('device_type3', args)

    # 初始化三个任务DAG每类DAG各一种
    env.initialize_task_dag('task_type1', args)
    env.initialize_task_dag('task_type2', args)
    env.initialize_task_dag('task_type3', args)

    # 打印设备信息
    env.print_device_info()

    # 打印DAG图的邻接表
    print('**** Task DAG Information ****')
    for task in env.tasks:
        task_dag, task_type, device, arrival_time = task
        print(f"任务DAG种类: {task_type}, 所属设备: {device}, 到达时间: {arrival_time}")
        print('---- Task DAG Start ----')
        task_dag.print_adjacency_list()
        print('---- Task DAG End ----')