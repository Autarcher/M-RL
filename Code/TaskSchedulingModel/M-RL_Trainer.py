import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import random
import time
import sys
import os
import csv

import numpy as np

import dgl

from src.Environment import Environment
from src.brain_dqn import BrainDQN
from src.DAG_PN import DAGGenerator
from src.util.prioritized_replay_memory import PrioritizedReplayMemory

#  definition of constants
MEMORY_CAPACITY = 50
GAMMA = 1
STEP_EPSILON = 5000.0
UPDATE_TARGET_FREQUENCY = 500
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
MAX_BETA = 10
MIN_VAL = -1000000
MAX_VAL = 1000000


class Trainer_MRL:
    """
    Definition of the Trainer DQN for the PN
    """

    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """

        self.args = args
        np.random.seed(self.args.seed)
        self.n_action = args.n_task

        self.num_node_feats = 2 * (1 + args.k_bound)
        self.num_edge_feats = 2 * (1 + args.k_bound)
        self.reward_scaling = 0.001

        self.brain = BrainDQN(self.args, self.num_node_feats, self.num_edge_feats)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)

        self.steps_done = 0
        self.init_memory_counter = 0
        self.n_step = args.n_step
        self.n_time_slot = args.n_time_slot  # number of time slots to end
        self.validation_len = 3

        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_node_feat: %d" % self.num_node_feats)
        print("[INFO] n_edge_feat: %d" % self.num_edge_feats)
        print("***********************************************************")

    def run_training(self):
        """
        Run de main loop for training the model
        """
        #  Generate a random instance
        instance = DAGGenerator(self.args.n_task, self.args.n_edge, self.args.n_h_max, self.args.k_bound, self.args.n_device)
        env = Environment(instance, self.reward_scaling, self.n_step)
        start_time = time.time()

        if self.args.plot_training:
            iter_list = []
            reward_list = []

        self.initialize_memory(env)
        print('[INFO]', 'iter', 'time', 'avg_reward_learning', 'loss', "beta")

        cur_best_reward = MIN_VAL

        for i in range(self.args.n_episode):

            loss, beta = self.run_episode(i, False, env)

            #  We first evaluate the validation step every 10 episodes, until 100, then every 100 episodes.
            if (i % 10 == 0 and i < 101) or i % 100 == 0:

                avg_reward = 0.0
                for j in range(self.validation_len):
                    avg_reward += self.evaluate_instance(j, env)

                avg_reward = avg_reward / self.validation_len

                cur_time = round(time.time() - start_time, 2)

                print('[DATA]', i, cur_time, avg_reward, loss, beta)

                sys.stdout.flush()

                if self.args.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()

                    plt.plot(iter_list, reward_list, linestyle="-", label="DQN", color='y')

                    plt.legend(loc=3)
                    out_fig_file = '%s/training_curve_reward.png' % self.args.save_dir
                    out_csv_file = '%s/training_data.csv' % self.args.save_dir
                    if not os.path.exists(self.args.save_dir):
                        os.makedirs(self.args.save_dir)
                    plt.savefig(out_fig_file)

                    # save the data
                    with open(out_csv_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        # 写入标题
                        writer.writerow(['Iteration', 'Reward'])
                        # 写入数据
                        for i, reward in zip(iter_list, reward_list):
                            writer.writerow([i, reward])

                fn = "iter_%d_model.pth.tar" % i

                #  We record only the model that is better on the validation set wrt. the previous model
                #  We nevertheless record a model every 10000 episodes
                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(folder=self.args.save_dir, filename=fn)
                elif i % 10000 == 0:
                    self.brain.save(folder=self.args.save_dir, filename=fn)

    def initialize_memory(self, env):
        """
        Initialize the replay memory with random episodes and a random selection
        """

        while self.init_memory_counter < MEMORY_CAPACITY:
            self.run_episode(0, True, env)

        print("[INFO] Memory Initialized")

    def run_episode(self, episode_idx, memory_initialization, env):
        """
        Run a single episode, either for initializing the memory (random episode in this case)
        or for training the model (following DQN algorithm)
        :param episode_idx: the index of the current episode done (without considering the memory initialization)
        :param memory_initialization: True if it is for initializing the memory
        :return: the loss and the current beta of the softmax selection
        """

        cur_state = env.get_initial_environment()

        graph_list = []
        rewards_vector = np.zeros(self.n_step)
        actions_vector = np.zeros(self.n_step, dtype=np.int16)
        available_vector = np.zeros((self.n_step, env.p_num + env.t_num))

        idx = 0
        total_loss = 0

        #  the current temperature for the softmax selection: increase from 0 to MAX_BETA
        temperature = max(0., min(self.args.max_softmax_beta,
                                  (episode_idx - 1) / STEP_EPSILON * self.args.max_softmax_beta))

        #  execute the episode
        while True:
            graph = env.make_nn_input(cur_state, self.args.mode)
            avail, avail_idx = env.get_valid_actions(cur_state)
            if avail_idx.size != 0:
                if memory_initialization:  # if we are in the memory initialization phase, a random episode is selected
                    action = random.choice(avail_idx)
                else:  # otherwise, we do the softmax selection
                    action = self.soft_select_action(graph, avail, temperature)
            else:
                action = 0
            #  each time we do a step, we increase the counter, and we periodically synchronize the target network
            self.steps_done += 1
            if self.steps_done % UPDATE_TARGET_FREQUENCY == 0:
                self.brain.update_target_model()
            # print(cur_state)
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            # print("reward:", reward)

            # graph_list[idx] = graph
            graph_list.append(graph)
            rewards_vector[idx] = reward
            actions_vector[idx] = action
            available_vector[idx] = avail

            idx += 1
            if idx >= self.n_step or cur_state.cur_time >= self.n_time_slot:
                break

        #  compute the n-step values
        for i in range(idx - 1):
            cur_graph = graph_list[i]
            cur_available = available_vector[i]
            next_graph = graph_list[i + 1]
            next_available = available_vector[i + 1]
            #  a state correspond to the graph
            state_features = (cur_graph, cur_available)
            next_state_features = (next_graph, next_available)

            #  the n-step reward
            reward = rewards_vector[i]
            action = actions_vector[i]

            sample = (state_features, action, reward, next_state_features)

            if memory_initialization:
                error = abs(reward)  # the error of the replay memory is equals to the reward, at initialization
                self.init_memory_counter += 1
                step_loss = 0
            else:
                x, y, errors = self.get_targets([(0, sample, 0)])  # feed the memory with the new samples
                error = errors[0]
                step_loss = self.learning()  # learning procedure

            self.memory.add(error, sample)

            total_loss += step_loss

        print("total_loss:", total_loss)
        return total_loss, temperature

    def evaluate_instance(self, idx, env):
        """
        Evaluate an instance with the current model
        :param idx: the index of the instance in the validation set
        :return: the reward collected for this instance
        """

        # instance = self.validation_set[idx]
        cur_state = env.get_initial_environment()

        total_reward = 0
        while True:
            graph = env.make_nn_input(cur_state, self.args.mode)
            # print("interval ————————————")
            avail, avail_idx = env.get_valid_actions(cur_state)
            if avail_idx.size != 0:
                action = self.select_action(graph, avail)
            else:
                action = 0
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)
            # print(cur_state)
            total_reward += reward
            if cur_state.cur_time >= self.n_time_slot :
                break

        return total_reward

    def select_action(self, graph, available):
        """
        Select an action according the to the current model
        :param graph: the graph (first part of the state)
        :param available: the vector of available (second part of the state)
        :return: the action, following the greedy policy with the model prediction
        """

        batched_graph = dgl.batch([graph, ])
        available = available.astype(bool)
        out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)

        action_idx = np.argmax(out[available])

        action = np.arange(len(out))[available][action_idx]

        return action

    # def soft_select_action(self, graph, available, beta):
    #     """
    #     Select an action according the to the current model with a softmax selection of temperature beta
    #     :param available: the vector of available
    #     :param beta: the current temperature
    #     :return: the action, following the softmax selection with the model prediction
    #     """
    #
    #     batched_graph = dgl.batch([graph, ])
    #     available = available.astype(bool)
    #     out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)
    #
    #     if len(out[available]) > 1:
    #         logits = (out[available] - out[available].mean())
    #         div = ((logits ** 2).sum() / (len(logits) - 1)) ** 0.5
    #         logits = logits / div
    #
    #         probabilities = np.exp(beta * logits)
    #         norm = probabilities.sum()
    #
    #         if norm == np.infty:
    #             action_idx = np.argmax(logits)
    #             action = np.arange(len(out))[available][action_idx]
    #             return action, 1.0
    #
    #         probabilities /= norm
    #     else:
    #         probabilities = [1]
    #
    #     action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
    #     action = np.arange(len(out))[available][action_idx]
    #     return action

    def soft_select_action(self, graph, available, beta):
        batched_graph = dgl.batch([graph, ])
        available = available.astype(bool)
        out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)

        if len(out[available]) > 1:
            logits = out[available] - out[available].mean()
            div = ((logits ** 2).sum() / max((len(logits) - 1), 1)) ** 0.5
            if div > 0:
                logits = logits / div

            logits -= logits.max()
            probabilities = np.exp(beta * logits)
            norm = probabilities.sum()

            if norm == 0 or np.isinf(norm):
                action_idx = np.argmax(logits)
                action = np.arange(len(out))[available][action_idx]
                return action, 1.0

            probabilities /= max(norm, 1e-8)  # 防止除以0
        else:
            probabilities = [1]

        action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        action = np.arange(len(out))[available][action_idx]
        return action

    def get_targets(self, batch):
        """
        Compute the TD-errors using the n-step Q-learning function and the model prediction
        :param batch: the batch to process
        :return: the state input, the true y, and the error for updating the memory replay
        """

        batch_len = len(batch)
        graph, avail = list(zip(*[e[1][0] for e in batch]))

        graph_batch = dgl.batch(graph)

        next_graph, next_avail = list(zip(*[e[1][3] for e in batch]))
        next_graph_batch = dgl.batch(next_graph)
        next_copy_graph_batch = dgl.batch(dgl.unbatch(next_graph_batch))
        p = self.brain.predict(graph_batch, target=False)

        if next_graph_batch.number_of_nodes() > 0:
            p_ = self.brain.predict(next_graph_batch, target=False)
            p_target_ = self.brain.predict(next_copy_graph_batch, target=True)
            # print("p_", p_)
            # print("p_target_", p_target_)


        x = []
        y = []
        errors = np.zeros(len(batch))

        for i in range(batch_len):

            sample = batch[i][1]
            state_graph, state_avail = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state_graph, next_state_avail = sample[3]
            next_action_indices = np.argwhere(next_state_avail == 1).reshape(-1)
            t = p[i]

            q_value_prediction = t[action]

            if len(next_action_indices) == 0:

                td_q_value = reward
                t[action] = td_q_value

            else:

                mask = np.zeros(p_[i].shape, dtype=bool)
                mask[next_action_indices] = True

                best_valid_next_action_id = np.argmax(p_[i][mask])
                best_valid_next_action = np.arange(len(mask))[mask.reshape(-1)][best_valid_next_action_id]

                td_q_value = reward + GAMMA * p_target_[i][best_valid_next_action]
                t[action] = td_q_value

            state = (state_graph, state_avail)
            x.append(state)
            y.append(t)

            errors[i] = abs(q_value_prediction - td_q_value)

        return x, y, errors

    def learning(self):
        """
        execute a learning step on a batch of randomly selected experiences from the memory
        :return: the subsequent loss
        """

        batch = self.memory.sample(self.args.batch_size)

        x, y, errors = self.get_targets(batch)

        #  update the errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        loss = self.brain.train(x, y)

        return round(loss, 4)



