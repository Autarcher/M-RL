from Code.SystemModel.TaskModel import TaskNode
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
    # state.print_actions(actions)
    print("------------------------------测试get_next_state函数--------------------------------------")
    # env.update_running_tasks(state, 0)
    # state = env.get_next_state()
    # env.update_running_tasks(state, 0)
    # env.update_running_tasks(state, 1)
    for i in range(800):
        # 假设这个部分在每秒的循环中执行
        #任务随即到达
        if random.random() < 0.1:  # 10% 的概率
            device0.initialize_task_dag('task_type1', args, env)

        if random.random() < 0.1:  # 10% 的概率
            device0.initialize_task_dag('task_type2', args, env)

        if random.random() < 0.1:  # 10% 的概率
            device0.initialize_task_dag('task_type3', args, env)

        state = env.get_next_state() #这个里面会判断app是否已经完成
        actions = state.generate_actions()
        if actions:
            # random.seed(args.seed)
            random_action = random.randint(0, len(actions) - 1)
            env.update_running_tasks(state, random_action)
        # env.update_running_tasks(state, 0)
        # for task_tuple in state.app_list:
        #     app_DAG, app_type, app_device, arriving_time = task_tuple
        #     print("*************************+++++++++其中一个任务++++++++**************************")
        #     print("app种类" + str(app_type))
        #     print("所属设备"+str(app_device))
        #     print("到达时间"+str(arriving_time))
        #     app_DAG.print_adjacency_list()
        #     print("scheduled flag:")
        #     print(app_DAG.task_node_scheduled_flag)
        #     print("finished flag:")
        #     print(app_DAG.task_node_finished_flag)
        #     print("finished time:")
        #     print(app_DAG.task_node_finished_time)
        #     print("scheduled seq:")
        #     print(app_DAG.task_node_scheduled_seq)
        #     print("app finished time")
        #     print(app_DAG.app_finished_time)


        # print("***************************************************")
        # for task in state.ready_tasks:
        #     task_node, app_device, arriving_time, node_id, appID = task
        #     print("所属设备"+str(app_device))
        #     print("到达时间"+str(arriving_time))
        #     print("任务ID:" + str(node_id))
        #     print("所属AppID:" + str(appID))
        #     task_node.print_task_info()
        #     print("***************************************************")
        #     actions = state.generate_actions()
        #     state.print_actions(actions)
        env.current_time += 1
        print(f"+++++++++++++++++++current_time:{env.current_time}+++++++++++++++++++++++")
        print(f"+++++++++++++++++++------奖励测试-------+++++++++++++++++++++++")
        actions = state.generate_actions()
        explore_step = 20
        if env.current_time % explore_step == 0:
            reward = (env.t_finished_apps_num * 10) + (env.t_finished_tasks_num * 1) + (
                        env.t_add_runnings_tasks_num * 1)
            print(f"调度的任务数：{env.t_add_runnings_tasks_num}")
            print(f"完成的任务数：{env.t_finished_tasks_num}")
            print(f"完成的app数：{env.t_finished_apps_num}")
            print(f"第{env.current_time / explore_step}次完成探索时的奖励：{reward} ")
            env.clear_t_record()
        print(f"+++++++++++++++++++------奖励测试结束-------+++++++++++++++++++++++")
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