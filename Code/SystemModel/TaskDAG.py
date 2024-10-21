import random
from Code.SystemModel.TaskModel import TaskNode



class TaskDAG:
    def __init__(self, num_nodes, data_range, computation_range, deadline_range, seed):
        """
        初始化DAG图，生成指定节点数量的任务节点。
        :param num_nodes: 任务节点的数量
        :param data_range: 数据量范围（如 (min_data_size, max_data_size)）
        :param computation_range: 计算量范围（如 (min_computation_size, max_computation_size)）
        :param deadline_range: 截止时间范围（如 (min_deadline, max_deadline)）
        """
        random.seed(seed)
        self.num_nodes = num_nodes
        self.data_range = data_range
        self.computation_range = computation_range
        self.deadline_range = deadline_range
        self.nodes = []  # 存储所有生成的 TaskNode
        self.task_node_finished_flag = []  # 标识任务是否已经被完成
        self.task_node_finished_time = []  # 记录子任务的完成时间
        self.task_node_scheduled_flag = [] # 标识任务是否已经在running_tasks队列
        self.task_node_scheduled_seq = []  #标识任务的调度顺序
        self.app_finished_time = 0         #标识整个app的完成时间

        self.adjacency_list = {}  # 存储DAG的邻接表，表示任务的前驱节点

        # 新增两个数组记录入度和出度
        self.in_degree = [0] * (num_nodes + 2)  # 每个节点的入度
        self.out_degree = [0] * (num_nodes + 2) # 每个节点的出度

        self._generate_dag()
        self._remove_redundant_dependencies()
        self._calculate_degrees()

    def _generate_dag(self):
        """
        内部函数，用于生成DAG图，包括入口任务和出口任务
        """
        # 生成入口任务，数据量和计算量为0
        entry_task = TaskNode(0, 0, 0)
        self.nodes.append((0, entry_task))  # 节点编号为0
        self.task_node_finished_flag.append(False)
        self.task_node_scheduled_flag.append(False)
        self.task_node_finished_time.append(0)
        self.adjacency_list[0] = []  # 初始化入口任务的依赖关系

        # 生成指定数量的 TaskNode
        for i in range(1, self.num_nodes + 1):
            data_size = random.randint(self.data_range[0], self.data_range[1])
            computation_size = random.uniform(self.computation_range[0], self.computation_range[1])
            deadline = random.randint(self.deadline_range[0], self.deadline_range[1])

            # 创建 TaskNode 实例并添加到节点列表中
            task_node = TaskNode(data_size, computation_size, deadline)
            self.nodes.append((i, task_node))  # 节点编号从1开始
            self.task_node_finished_flag.append(False)
            self.task_node_scheduled_flag.append(False)
            self.task_node_finished_time.append(0)
            self.adjacency_list[i] = []

        # 创建出口任务，数据量和计算量为0
        exit_task = TaskNode(0, 0, 0)
        exit_task_id = self.num_nodes + 1
        self.nodes.append((exit_task_id, exit_task))
        self.task_node_finished_flag.append(False)
        self.task_node_scheduled_flag.append(False)
        self.task_node_finished_time.append(0)
        self.adjacency_list[exit_task_id] = []  # 出口任务的依赖关系

        # 使用新的逻辑生成依赖关系，确保生成合理的DAG结构
        for i in range(1, self.num_nodes + 1):
            possible_dependencies = list(range(0, i))  # 节点可以依赖于之前的所有节点（包括入口任务）
            num_dependencies = random.randint(1, min(3, len(possible_dependencies)))  # 随机选择1到3个前驱节点
            dependencies = random.sample(possible_dependencies, num_dependencies)
            self.adjacency_list[i] = dependencies

        # 所有没有后继任务的节点（即没有其他任务依赖它们的）依赖于出口任务
        for i in range(1, self.num_nodes + 1):
            is_successor = False
            for deps in self.adjacency_list.values():
                if i in deps:
                    is_successor = True
                    break
            if not is_successor:
                self.adjacency_list[exit_task_id].append(i)  # 依赖出口任务

    def _remove_redundant_dependencies(self):
        """
        移除冗余的依赖关系，确保每个任务节点只保留直接依赖关系
        """
        for node in range(1, self.num_nodes + 1):
            direct_dependencies = set(self.adjacency_list[node])
            redundant_dependencies = set()
            for dep in direct_dependencies:
                for other_dep in direct_dependencies:
                    if dep != other_dep and self._has_path(other_dep, dep):
                        redundant_dependencies.add(dep)
            self.adjacency_list[node] = list(direct_dependencies - redundant_dependencies)

    def _calculate_degrees(self):
        """
        计算每个节点的入度和出度
        """
        # 重置入度和出度数组
        self.in_degree = [0] * (self.num_nodes + 2)
        self.out_degree = [0] * (self.num_nodes + 2)

        # 计算入度和出度
        for node, dependencies in self.adjacency_list.items():
            self.in_degree[node] = len(dependencies)
            for dep in dependencies:
                self.out_degree[dep] += 1

    def _has_path(self, start, end):
        """
        检查从节点 start 是否可以到达节点 end
        :param start: 起始节点
        :param end: 目标节点
        :return: 如果存在路径则返回 True，否则返回 False
        """
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == end:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(self.adjacency_list[node])

        return False

    def print_adjacency_list(self):
        """
        打印DAG图的邻接表（依赖关系：存储依赖于哪些任务）
        """
        print(
            f"Adjacency List (Dependencies) of the DAG with {self.num_nodes + 2} nodes (including entry and exit nodes):")
        for node_id, dependencies in self.adjacency_list.items():
            if dependencies:
                print(f"Task {node_id} depends on: {', '.join(map(str, dependencies))}")
            else:
                print(f"Task {node_id} has no dependencies.")

        # 打印每个节点的 TaskNode 信息
        for node_id, task_node in self.nodes:
            print(f"\nTask {node_id} Information:")
            task_node.print_task_info()

        # 打印每个节点的入度和出度
        print("\nIn-degree for each task:")
        for i, in_deg in enumerate(self.in_degree):
            print(f"Task {i}: {in_deg} in-degree")

        print("\nOut-degree for each task:")
        for i, out_deg in enumerate(self.out_degree):
            print(f"Task {i}: {out_deg} out-degree")


# 示例用法
if __name__ == "__main__":
    # 设置数据量、计算量和截止时间的取值范围
    data_range = (100, 1000)  # 数据量范围：100字节到1000字节
    computation_range = (1e8, 1e10)  # 计算量范围：1亿FLOPs到100亿FLOPs
    deadline_range = (100, 1000)  # 截止时间范围：100秒到1000秒

    # 生成一个包含5个任务节点的DAG（总共7个节点，包括入口和出口任务）
    task_dag = TaskDAG(num_nodes=5, data_range=data_range, computation_range=computation_range,
                       deadline_range=deadline_range, seed = 1)

    # 打印DAG图的邻接表（存储依赖于哪些任务）
    task_dag.print_adjacency_list()