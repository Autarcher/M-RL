from Code.SystemModel.TaskDAG import TaskDAG
from Code.SystemModel.DeviceModel import Device
from Code.SystemModel.EnvInit import EnvInit





import torch
import numpy as np
import dgl
import scipy.sparse as sp
import math
import random
import networkx as nx
from matplotlib.style.core import available

def is_sparse(matrix, threshold=0.8):
    """ 判断矩阵是否稀疏 """
    zero_count = np.count_nonzero(matrix == 0)
    total_elements = matrix.size
    zero_ratio = zero_count / total_elements
    return zero_ratio > threshold

def shift_left(vec, start_pos):
    value = vec[start_pos]
    if start_pos < vec.size - 1:
        vec[start_pos:-1] = vec[start_pos + 1:]
    vec[-1] = 0
    return value


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



class Environment:

    def __init__(self, instance, reward_scaling, max_time):
        """
        Initialize the System environment
        """
