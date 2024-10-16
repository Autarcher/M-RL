import random
from TaskModel import TaskNode
random.seed(1)
class TaskDAG:
    def __init__(self, num_nodes, data_range, computation_range, deadline_range):
        """
        初始化DAG图，生成指定节点数量的任务节点。
        :param num_nodes: 任务节点的数量
        :param data_range: 数据量范围（如 (min_data_size, max_data_size)）
        :param computation_range: 计算量范围（如 (min_computation_size, max_computation_size)）
        :param deadline_range: 截止时间范围（如 (min_deadline, max_deadline)）
        """
        self.num_nodes = num_nodes
        self.data_range = data_range
        self.computation_range = computation_range
        self.deadline_range = deadline_range
        self.nodes = []  # 存储所有生成的 TaskNode
        self.adjacency_list = {}  # 存储DAG的邻接表，表示任务依赖关系

        self._generate_dag()

    def _generate_dag(self):
        """
        内部函数，用于生成DAG图，包括入口任务和出口任务
        """
        # 生成入口任务，数据量和计算量为0
        entry_task = TaskNode(0, 0, 0)
        self.nodes.append((0, entry_task))  # 节点编号为0
        self.adjacency_list[0] = []  # 初始化入口任务的依赖关系

        # 生成指定数量的 TaskNode
        for i in range(1, self.num_nodes + 1):
            data_size = random.randint(self.data_range[0], self.data_range[1])
            computation_size = random.uniform(self.computation_range[0], self.computation_range[1])
            deadline = random.randint(self.deadline_range[0], self.deadline_range[1])

            # 创建 TaskNode 实例并添加到节点列表中
            task_node = TaskNode(data_size, computation_size, deadline)
            self.nodes.append((i, task_node))  # 节点编号从1开始
            self.adjacency_list[i] = []

        # 创建出口任务，数据量和计算量为0
        exit_task = TaskNode(0, 0, 0)
        exit_task_id = self.num_nodes + 1
        self.nodes.append((exit_task_id, exit_task))
        self.adjacency_list[exit_task_id] = []  # 出口任务的依赖关系

        # 所有任务节点依赖于入口任务（编号为0）
        for i in range(1, self.num_nodes + 1):
            self.adjacency_list[0].append(i)  # 所有任务都依赖于入口任务

        # 创建随机的依赖关系，确保没有循环（有向无环图的特性）
        for i in range(1, self.num_nodes + 1):
            num_dependencies = random.randint(0, i - 1)
            dependencies = random.sample(range(1, i), num_dependencies)
            for dep in dependencies:
                self.adjacency_list[dep].append(i)

        # 所有没有后继任务的节点依赖于出口任务
        for i in range(1, self.num_nodes + 1):
            if len(self.adjacency_list[i]) == 0:
                self.adjacency_list[i].append(exit_task_id)

    def print_dag_info(self):
        """
        打印DAG图的基本信息，包括每个节点的任务信息和依赖关系
        """
        print(f"DAG with {self.num_nodes + 2} nodes (including entry and exit nodes):")
        for node_id, node in self.nodes:
            if node_id == 0:
                print(f"Task {node_id} (Entry Task):")
            elif node_id == self.num_nodes + 1:
                print(f"Task {node_id} (Exit Task):")
            else:
                print(f"Task {node_id}:")

            node.print_task_info()
            dependents = self.adjacency_list[node_id]
            if dependents:
                print("Dependent on:")
                for dep_id in dependents:
                    if dep_id == 0:
                        print(f"  - Task {dep_id} (Entry Task)")
                    elif dep_id == self.num_nodes + 1:
                        print(f"  - Task {dep_id} (Exit Task)")
                    else:
                        print(f"  - Task {dep_id}")
            else:
                print("No dependencies.")
            print()

# 示例用法
if __name__ == "__main__":
    # 设置数据量、计算量和截止时间的取值范围
    data_range = (100, 1000)  # 数据量范围：100字节到1000字节
    computation_range = (1e8, 1e10)  # 计算量范围：1亿FLOPs到100亿FLOPs
    deadline_range = (100, 1000)  # 截止时间范围：100秒到1000秒

    # 生成一个包含5个任务节点的DAG（总共7个节点，包括入口和出口任务）
    task_dag = TaskDAG(num_nodes=5, data_range=data_range, computation_range=computation_range, deadline_range=deadline_range)

    # 打印DAG图的信息
    task_dag.print_dag_info()
