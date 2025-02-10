from Code.SystemModel.TaskDAG import TaskDAG
from Code.SystemModel.DeviceModel import Device
from Code.SystemModel.EnvInit import EnvInit
from Code.TaskSchedulingModel.DQNTrainer import Trainer
import argparse
import random
import argparse
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_device_parameters(parser):
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

    return parser


def add_task_parameters(parser):
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
    parser.add_argument('--num_nodes3', type=int, default=9, help="DAG图的节点数量")

    # Common TaskDAG parameters
    parser.add_argument('--data_range', type=tuple, default=(100, 1000), help="数据量范围：100字节到1000字节")
    #default=(1e8, 1e10)
    #n_default=(1e6, 1e7)
    parser.add_argument('--computation_range', type=tuple, default=(1e6, 1e7), help="计算量范围：1亿FLOPs到100亿FLOPs")
    parser.add_argument('--deadline_range', type=tuple, default=(100, 1000), help="截止时间范围：100秒到1000秒")

    parser.add_argument('--task_basis_nodes', type=int, default=5, help="保证任务的计算数量级相近")

    return parser

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Device parameters
    parser = add_device_parameters(parser)

    # TaskDAG parameters for dag type
    parser = add_task_parameters(parser)



    # Hyper parameters
    parser.add_argument('--seed', type=int, default=2024, help="全局种子")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--n_step', type=int, default=600)
    parser.add_argument('--n_time_slot', type=int, default=300)
    parser.add_argument('--max_softmax_beta', type=int, default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64, help='dimension of latent layers')
    parser.add_argument('--num_node_feats', type=int, default=15, help="The features dimension of a node")
    parser.add_argument('--max_actions', type=int, default=30, help="the length of action space")

    # Argument for Trainer
    parser.add_argument('--n_episode', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='./result-default')
    parser.add_argument('--plot_training', type=int, default=1)
    parser.add_argument('--mode', default='gpu', help='cpu/gpu')

    return parser.parse_args()

def run_M_RL():
    args = parse_arguments()
    random.seed(args.seed)
    env = EnvInit(config_path="config.yaml")
    env.initialize(args=args)
    trainer = Trainer(args)

    # 初始化三个设备每类设备各一种
    env.initialize_devices('device_type1', args)
    env.initialize_devices('device_type2', args)
    env.initialize_devices('device_type3', args)

    # device0发出三个任务请求初始化三个任务DAG每类DAG各一种
    device0 = env.devices[0][0]
    device0.initialize_task_dag('task_type1', args, env)
    device0.initialize_task_dag('task_type2', args, env)
    device0.initialize_task_dag('task_type3', args, env)
    device0.initialize_task_dag('task_type3', args, env)



    #initialize state
    state = env.get_state()
    actions = state.generate_actions()
    state.print_actions(actions)
    env.update_running_tasks(state, 0)
    state, reward = env.get_next_state_and_reward(state, -2)
    actions = state.generate_actions()
    for task_tuple in state.app_list:
        app_DAG, app_type, app_device, arriving_time = task_tuple
        print("*************************+++++++++The one task++++++++**************************")
        print("App type" + str(app_type))
        print("Owning device" + str(app_device))
        print("Arriving time" + str(arriving_time))
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
        print("App type" + str(app_type))
        print("Owning device" + str(app_device))
        print("Arriving time" + str(arriving_time))
        print("Owning App`s ID:" + str(appID))
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
    print("*******************************Test: aggregate_appDAG_task_features FINISHED*******************************")
    print("*******************************TEST: init_memory FUNC*******************************")
    env.env_clear()
    # trainer.initialize_memory(env)
    trainer.run_training(env, args)

def run_FIFS(is_FIFS=True):
    args = parse_arguments()
    random.seed(args.seed)
    env = EnvInit(config_path="config.yaml")
    env.initialize(args=args)

    # 初始化三个设备每类设备各一种
    env.initialize_devices('device_type1', args)
    env.initialize_devices('device_type2', args)
    env.initialize_devices('device_type3', args)

    # device0发出三个任务请求初始化三个任务DAG每类DAG各一种
    device0 = env.devices[0][0]
    device0.initialize_task_dag('task_type1', args, env)
    device0.initialize_task_dag('task_type2', args, env)
    device0.initialize_task_dag('task_type3', args, env)
    device0.initialize_task_dag('task_type1', args, env)
    device0.initialize_task_dag('task_type2', args, env)
    device0.initialize_task_dag('task_type3', args, env)



    #初始化state
    state = env.get_state()
    actions = state.generate_actions()
    state.print_actions(actions)
    print("------------------------------测试get_next_state函数--------------------------------------")
    seed = args.seed
    for i in range(200):
        # The system is busy
        # device0.initialize_task_dag('task_type3', args, env)
        #任务随即到达
        random_instance = random.Random(seed)
        random1 = random_instance.random()
        random2 = random_instance.random()
        random3 = random_instance.random()
        actions = state.generate_actions()
        actions_len = len(actions)
        count = 0
        if (actions_len <= 30):
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

        if actions and is_FIFS:
            action = 0
        elif actions and not is_FIFS:
            random_action = random.randint(0, len(actions) - 1)
            action = random_action
        else:
            action = -2

        state, reward = env.get_next_state_and_reward(state, action) #这个里面会判断app是否已经完成
        actions = state.generate_actions()
        env.current_time += 1
        print(f"+++++++++++++++++++current_time:{env.current_time}+++++++++++++++++++++++")

        actions = state.generate_actions()
        explore_step = 1
        if env.current_time >= 0 :
            # reward = (env.t_finished_apps_num * 10) + (env.t_finished_tasks_num * 1) + (
            #             env.t_add_runnings_tasks_num * 1)
            print(f"The {env.current_time / explore_step} time decision's reward：{reward} ")
            env.clear_t_record()

    print("*******************************Test: get_next_state FINFSHED*******************************")
    sum_time = 0
    finished_tasks_num = 0
    finished_task1_num = 0
    finished_task2_num = 0
    finished_task3_num = 0
    for index, app in enumerate(env.tasks):
        appDAG, _, _, arriving_time = app
        print(f"第{index}个任务的到达时刻是{arriving_time}的完成时刻是{appDAG.app_finished_time},"
              f"响应时间(任务完成时间-任务到达时间)是{appDAG.app_finished_time - arriving_time}")
        if appDAG.app_finished_time > 0:
            sum_time += appDAG.app_finished_time - arriving_time
            finished_tasks_num += 1
            if appDAG.task_type == 1:
                finished_task1_num += 1
            elif appDAG.task_type == 2:
                finished_task2_num += 1
            elif appDAG.task_type == 3:
                finished_task3_num += 1
    if finished_tasks_num == 0:
        print(f"完成的任务数{finished_tasks_num}")
        print(f"完成数1类任务：{finished_task1_num},2类任务：{finished_task2_num},3类任务：{finished_task3_num}")
        print(f"平均响应时间:-1(没有任务完成)")

    else:
        print(f"完成的任务数{finished_tasks_num}")
        print(f"1类任务：{finished_task1_num},2类任务：{finished_task2_num},3类任务：{finished_task3_num}")
        print(f"平均响应时间:{sum_time / finished_tasks_num}")
    print(f"seed:{seed}")
# 使用示例
if __name__ == "__main__":
    #run_FIFS(False)
    run_M_RL()