class TaskNode:
    def __init__(self, data_size, computation_size, deadline):
        """
        初始化任务节点的属性
        :param data_size: 任务的数据量（单位可以是字节）
        :param computation_size: 任务的计算量（单位可以是 FLOP）
        :param deadline: 任务的截止时间（单位可以是秒或时间戳）
        """
        self.ready_time = 0         #任务的可执行时间表示
        self.is_end_task = 0        #任务是否为app出口任务
        self.is_start_task = 0      #任务是否为app入口任务
        self.data_size = data_size  # 任务的数据量
        self.computation_size = computation_size  # 任务的计算量
        self.deadline = deadline  # 任务的截止时间

    def print_task_info(self):
        """
        打印任务节点的基本信息
        """
        print(f"Task Info:")
        print(f"Data Size: {self.data_size} bytes")
        print(f"Computation Size: {self.computation_size} FLOPs")
        print(f"Deadline: {self.deadline} seconds")





# 主函数，创建任务节点并打印任务信息
if __name__ == "__main__":
    # 创建一个任务节点实例
    task = TaskNode(1e9, 5e9, 3600)  # 任务数据量为1GB，计算量为5GFLOPs，截止时间为3600秒（1小时）

    # 打印任务信息
    task.print_task_info()
