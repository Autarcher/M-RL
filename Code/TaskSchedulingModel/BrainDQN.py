
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

import os
import numpy as np
import dgl

from Code.TaskSchedulingModel.GATNetwork import GATModel


class BrainDQN:
    """
    Definition of the DQN Brain, computing the DQN loss
    """

    def __init__(self, args, num_node_feat, num_edge_feat):
        """
        Initialization of the DQN Brain
        :param args: argparse object taking hyperparameters
        :param num_node_feat: number of features on the nodes
        :param num_edge_feat: number of features on the edges
        """
        in_dim = args.num_node_feats
        hidden_dim = 16
        out_dim = 1
        num_heads = 4
        num_epochs = 1000
        self.lr = args.learning_rate

        self.args = args


        self.model = GATModel(in_dim, hidden_dim, out_dim, num_heads)
        self.target_model = GATModel(in_dim, hidden_dim, out_dim, num_heads)


        if self.args.mode == 'gpu':
            self.model.cuda()
            self.target_model.cuda()

    def train(self, x, y):
        """
        Compute the loss between (f(x) and y)
        :param x: the input
        :param y: the true value of y
        :return: the loss
        """
        action_graphs, action_nodeIDs = x

        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        max_value = float('-inf')
        max_graph = None
        max_nodeID = None

        # Forward pass for each graph
        for i, graph in enumerate(action_graphs):
            features = graph.ndata['n_feat']
            output = self.model(graph, features)

            # Extract the value corresponding to action_nodeID[i]
            value = output[action_nodeIDs[i]]

            # Track the maximum value
            if value > max_value:
                max_value = value
                max_graph = graph
                max_nodeID = action_nodeIDs[i]

        # Calculate loss (mean squared error between max value and reward)
        loss = F.mse_loss(max_value, torch.tensor([y], dtype=torch.float))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self, action_graphs, action_nodeIDs, target=False):
        """
        Predict the Q-values for the given action graphs and node IDs
        :param action_graphs: list of graphs serving as input
        :param action_nodeIDs: list of node IDs for which predictions are needed
        :param target: True if the target network must be used for the prediction
        :return: A list of the predictions for each action_nodeID
        """
        with torch.no_grad():
            predictions = []
            if target:
                self.target_model.eval()
                model = self.target_model
            else:
                self.model.eval()
                model = self.model




            # print("++++++++predict:action_graphs+++++++")
            # print(action_graphs)
            # print("+++++++action_nodeIDs+++++++")
            # print(action_nodeIDs)

            for graph, nodeID in zip(action_graphs, action_nodeIDs):
                features = graph.ndata['n_feat']
                output = model(graph, features)
                predictions.append(output[nodeID].item())

        return predictions

    def update_target_model(self):
        """
        Synchronise the target network with the current one
        """

        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder, filename):
        """
        Save the model
        :param folder: Folder requested
        :param filename: file name requested
        """

        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model.state_dict(), filepath)

    def load(self, folder, filename):
        """
        Load the model
        :param folder: Folder requested
        :param filename: file name requested
        """

        filepath = os.path.join(folder, filename)
        # torch.load(self.model.state_dict(), filepath)
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))
