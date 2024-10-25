import argparse
# from Code.TaskSchedulingModel.DQNTrainer import Trainer
from Code.SystemModel.EnvInit import  EnvInit
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn



import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, dglnn.GATConv):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)

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


