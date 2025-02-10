import argparse

import dgl
import numpy as np

# from Code.TaskSchedulingModel.DQNTrainer import Trainer
from Code.SystemModel.EnvInit import  EnvInit
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, activation=F.relu):
        super(GATLayer, self).__init__()
        self.gat = dglnn.GATConv(in_feats=in_dim,
                                 out_feats=out_dim,
                                 num_heads=num_heads,
                                 feat_drop=0.1,
                                 attn_drop=0.1,
                                 activation=activation)

    def forward(self, g, h):
        return self.gat(g, h)
class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GATModel, self).__init__()
        # 前三层 GAT
        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, 1)
        self.gat3 = GATLayer(hidden_dim * 1, hidden_dim, 1)

        # 最后一层 GAT，用于处理选定节点的特征
        self.gat4 = GATLayer(hidden_dim * 1, hidden_dim, 1, activation=F.relu)

        # 用于最后的预测，简单的全连接层
        self.fc = nn.Linear(hidden_dim, 1)  # 每个图一个预测值

        ###old
        # self.gat1 = dglnn.GATConv(in_feats=in_dim,
        #                           out_feats=hidden_dim,
        #                           num_heads=num_heads,
        #                           feat_drop=0.1,
        #                           attn_drop=0.1,
        #                           activation=F.relu)
        # self.gat2 = dglnn.GATConv(in_feats=hidden_dim * num_heads,
        #                           out_feats=hidden_dim,
        #                           num_heads=num_heads,
        #                           feat_drop=0.1,
        #                           attn_drop=0.1,
        #                           activation=F.relu)
        # self.gat3 = dglnn.GATConv(in_feats=hidden_dim * num_heads,
        #                           out_feats=out_dim,
        #                           num_heads=1,
        #                           feat_drop=0.1,
        #                           attn_drop=0.1)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # #0初始化
        # for m in self.modules():
        #     if isinstance(m, dglnn.GATConv):
        #         for param in m.parameters():
        #             nn.init.zeros_(param)

        #Xavier初始化方法
        for m in self.modules():

            torch.manual_seed(4)
            random.seed(4)
            np.random.seed(4)

            if isinstance(m, dglnn.GATConv):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)

    def forward(self, action_graphs, action_nodeIDs):
        # 获取模型的设备 (device)
        device = next(self.parameters()).device

        # 确保 action_nodeIDs 是 torch.tensor 类型，并且是 Long 类型，然后移动到设备上
        action_nodeIDs = torch.tensor(action_nodeIDs, dtype=torch.long, device=device)

        # 保存每个图的输出
        graph_outputs = []

        # 1. 对所有图进行前三层 GAT 操作
        all_graph_embeddings = []
        # print(f"action_graphs: {action_graphs}++++++++++++")
        # print(f"device: {device}++++++++++++")

        for g in action_graphs:
            # 将图移动到正确的设备
            g = g.to(device)

            # 为图添加自环
            g = dgl.add_self_loop(g)

            # 获取每个图的节点特征并移动到正确的设备
            h = g.ndata['n_feat'].to(device)  # 获取每个图的节点特征

            # 第一层 GAT
            h = self.gat1(g, h)
            h = h.flatten(1)  # 展平每个节点的多头输出

            # 第二层 GAT
            h = self.gat2(g, h)
            h = h.flatten(1)

            # 第三层 GAT
            h = self.gat3(g, h)
            h = h.flatten(1)

            # 保存每个图的节点特征
            all_graph_embeddings.append(h)

        # 2. 从每个图中选择对应的节点特征
        selected_node_features = []

        for i, h in enumerate(all_graph_embeddings):
            selected_h = h[action_nodeIDs[i]]  # 选择每个图中特定的节点特征
            selected_node_features.append(selected_h)

        # 3. 将这些选出的节点特征构成一个新的图，作为输入给第四层 GAT
        # 创建一个全连接图
        num_selected_nodes = len(selected_node_features)
        src = torch.arange(0, num_selected_nodes).repeat(num_selected_nodes)
        dst = torch.tile(torch.arange(0, num_selected_nodes), (num_selected_nodes,))  # 完全连接

        new_graph = dgl.graph((src, dst), num_nodes=num_selected_nodes).to(device)

        # 将选出的节点特征作为新图的节点特征，并转移到设备上
        new_graph.ndata['n_feat'] = torch.stack(selected_node_features).to(device)

        # 确保 new_graph 也在正确的设备上
        new_graph = new_graph.to(device)

        # 4. 使用第四层 GAT 处理选定节点的特征
        h = new_graph.ndata['n_feat']
        h = self.gat4(new_graph, h)  # 计算第四层 GAT
        h = h.flatten(1)  # 展平输出

        # 5. 用全连接层生成每个图的预测结果
        prediction = self.fc(h)  # 输出预测结果
        graph_outputs.append(prediction)

        # 将所有图的预测结果拼接成一个 tensor
        return torch.cat(graph_outputs, dim=0)

    # #old
    # def forward(self, g, features):
    #     # First GAT layer
    #     h = self.gat1(g, features)
    #     h = h.view(h.shape[0], -1)  # Flatten the output
    #
    #     # Second GAT layer
    #     h = self.gat2(g, h)
    #     h = h.view(h.shape[0], -1)  # Flatten the output
    #
    #     # Third GAT layer
    #     h = self.gat3(g, h)
    #     h = h.mean(1)  # Combine heads by averaging
    #     return h

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
            print(f"--------------action_nodeIDs:{action_nodeIDs}---------")
            predictions = self.forward(action_graphs, action_nodeIDs)

        return predictions

    # #old
    # def predict(self, action_graphs, action_nodeIDs):
    #     """
    #     Predict the Q-values for the given action graphs and node IDs
    #     :param action_graphs: list of graphs serving as input
    #     :param action_nodeIDs: list of node IDs for which predictions are needed
    #     :return: A list of the predictions for each action_nodeID
    #     """
    #     with torch.no_grad():
    #         predictions = []
    #
    #         self.eval()  # 设置模型为评估模式
    #
    #         for graph, nodeID in zip(action_graphs, action_nodeIDs):
    #             features = graph.ndata['n_feat']
    #             output = self(graph, features)  # 直接调用 self 来前向传播
    #             predictions.append(output[nodeID].item())
    #
    #     return predictions

if __name__ == "__main__":
    num_graphs = 15  # 最多20个图
    num_nodes = 8  # 每个图8个节点
    in_dim = 16  # 输入特征维度
    hidden_dim = 32  # 隐藏层维度
    out_dim = 64  # 输出维度
    num_heads = 4  # 注意力头数
    num_selected_nodes = 1  # 每个图选择3个节点

    # 随机生成每个图需要选择的节点索引 (shape: num_graphs, num_selected_nodes)
    # 使用 Python list 初始化 action_nodeIDs
    action_nodeIDs = []
    for _ in range(num_graphs):
        # 为每个图随机选择 1 个节点
        action_nodeIDs.append([random.choice(range(num_nodes))])  # 使用 random.choice 随机选择一个节点
    print("Action Node IDs:", action_nodeIDs)

    # 创建模型
    model = GATModel(in_dim, hidden_dim, out_dim, num_heads)

    # 假设我们有20个图，每个图有8个节点，且每个图是全连接的
    graphs = []
    for _ in range(num_graphs):
        # 构建全连接图（完全图），即每个节点与其他所有节点都有边
        src = torch.arange(0, num_nodes).repeat(num_nodes)
        dst = torch.tile(torch.arange(0, num_nodes), (num_nodes,))

        # 创建 DGL 图
        g = dgl.graph((src, dst), num_nodes=num_nodes)

        # 随机初始化节点特征
        g.ndata['n_feat'] = torch.randn(num_nodes, in_dim)

        # 添加到图列表
        graphs.append(g)

    # 进行前向传播
    predictions = model(graphs, action_nodeIDs)
    print(predictions.shape)  # 输出预测结果形状 (num_graphs, 1)
    print(predictions)  # 输出预测结果形状 (num_graphs, 1)


