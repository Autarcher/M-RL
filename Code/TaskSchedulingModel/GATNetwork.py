import argparse
# from Code.TaskSchedulingModel.DQNTrainer import Trainer
from Code.SystemModel.EnvInit import  EnvInit
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn



class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GATModel, self).__init__()
        self.gat1 = dglnn.GATConv(in_feats=in_dim,
                                  out_feats=hidden_dim,
                                  num_heads=num_heads,
                                  feat_drop=0.1,
                                  attn_drop=0.1,
                                  activation=F.relu)
        self.gat2 = dglnn.GATConv(in_feats=hidden_dim * num_heads,
                                  out_feats=hidden_dim,
                                  num_heads=num_heads,
                                  feat_drop=0.1,
                                  attn_drop=0.1,
                                  activation=F.relu)
        self.gat3 = dglnn.GATConv(in_feats=hidden_dim * num_heads,
                                  out_feats=out_dim,
                                  num_heads=1,
                                  feat_drop=0.1,
                                  attn_drop=0.1)

    def forward(self, g, features):
        # First GAT layer
        h = self.gat1(g, features)
        h = h.view(h.shape[0], -1)  # Flatten the output

        # Second GAT layer
        h = self.gat2(g, h)
        h = h.view(h.shape[0], -1)  # Flatten the output

        # Third GAT layer
        h = self.gat3(g, h)
        h = h.mean(1)  # Combine heads by averaging
        return h

    def predict(self, action_graphs, action_nodeIDs):
        """
        Predict the Q-values for the given action graphs and node IDs
        :param action_graphs: list of graphs serving as input
        :param action_nodeIDs: list of node IDs for which predictions are needed
        :return: A list of the predictions for each action_nodeID
        """
        with torch.no_grad():
            predictions = []

            self.eval()  # 设置模型为评估模式

            for graph, nodeID in zip(action_graphs, action_nodeIDs):
                features = graph.ndata['n_feat']
                output = self(graph, features)  # 直接调用 self 来前向传播
                predictions.append(output[nodeID].item())

        return predictions


def train_gat_model(action_graphs, action_nodeIDs, reward, num_epochs=1000, lr=0.1):
    # Initialize GAT model
    in_dim = 13
    hidden_dim = 16
    out_dim = 1
    num_heads = 4
    model = GATModel(in_dim, hidden_dim, out_dim, num_heads)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()

        max_value = float('-inf')
        max_graph = None
        max_nodeID = None

        # Forward pass for each graph
        for i, graph in enumerate(action_graphs):
            features = graph.ndata['n_feat']
            output = model(graph, features)

            # Extract the value corresponding to action_nodeID[i]
            value = output[action_nodeIDs[i]]

            # Track the maximum value
            if value > max_value:
                max_value = value
                max_graph = graph
                max_nodeID = action_nodeIDs[i]

        # Calculate loss (mean squared error between max value and reward)
        loss = F.mse_loss(max_value, torch.tensor([reward], dtype=torch.float))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model


# 使用示例
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
# if __name__ == "__main__":
#     args = parse_arguments()
#     env = EnvInit(config_path="config.yaml")
#     env.initialize(args=args)
#     trainer = Trainer(args)
#
#     # 初始化三个设备每类设备各一种
#     env.initialize_devices('device_type1', args)
#     # env.initialize_devices('device_type1', args)
#     env.initialize_devices('device_type2', args)
#     # env.initialize_devices('device_type2', args)
#     # env.initialize_devices('device_type3', args)
#     env.initialize_devices('device_type3', args)
#
#     # device0发出三个任务请求初始化三个任务DAG每类DAG各一种
#     device0 = env.devices[0][0]
#     device0.initialize_task_dag('task_type1', args, env)
#
#
#
#     #初始化state
#     state = env.get_state()
#     actions = state.generate_actions()
#     state.print_actions(actions)
#     env.update_running_tasks(state, 0)
#     state, reward = env.get_next_state_and_reward(state, -2)
#     actions = state.generate_actions()
#     for task_tuple in state.app_list:
#         app_DAG, app_type, app_device, arriving_time = task_tuple
#         print("*************************+++++++++其中一个任务++++++++**************************")
#         print("app种类" + str(app_type))
#         print("所属设备" + str(app_device))
#         print("到达时间" + str(arriving_time))
#         app_DAG.print_adjacency_list()
#         print("scheduled flag:")
#         print(app_DAG.task_node_scheduled_flag)
#         print("finished flag:")
#         print(app_DAG.task_node_finished_flag)
#         print("finished time:")
#         print(app_DAG.task_node_finished_time)
#         print("scheduled seq:")
#         print(app_DAG.task_node_scheduled_seq)
#         print("app finished time")
#         print(app_DAG.app_finished_time)
#
#     print("***************************************************")
#     for task in state.ready_tasks:
#         task_node, app_device, arriving_time, node_id, appID = task
#         print("所属设备" + str(app_device))
#         print("到达时间" + str(arriving_time))
#         print("任务ID:" + str(node_id))
#         print("所属AppID:" + str(appID))
#         task_node.print_task_info()
#         print("***************************************************")
#         actions = state.generate_actions()
#         state.print_actions(actions)
#
#     action_graphs = []
#     action_nodeIDs = []
#     actions = state.generate_actions()
#     for index, action in enumerate(actions):
#         task = env.tasks[action.get("appID")]
#         action_nodeIDs.append(action.get("node_id"))
#         print(f"----------第{index}个ready------------")
#         task_features, adj_matrix = trainer.aggregate_appDAG_task_features(state, env, task)
#         # state, reward = env.get_next_state_and_reward(state, task.task_id)  # 使用 task_id 获取状态和奖励
#         graph = trainer.make_nn_input(task_features, adj_matrix)
#         action_graphs.append(graph)
#         trainer.print_task_features(task_features, adj_matrix)
#         print(graph)
#     # print("tensor_data:")
#     reward = 2
#     model = train_gat_model(action_graphs, action_nodeIDs, reward)
#     print("预测结果：")
#     print(model.predict(action_graphs,action_nodeIDs))
#     # loss, _ = trainer.run_episode_test(1, True, env)
#     # print(f"test_Loss:{loss}")
#     print("*******************************测试aggregate_appDAG_task_features函数完毕*******************************")