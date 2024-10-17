import math
import random
from Code.SystemModel.TaskModel import TaskNode
from Code.SystemModel.TaskDAG import TaskDAG

class Device:
    def __init__(self, coordinates, computing_speed, channel_gain, available_time = 0):
        self.coordinates = coordinates
        self.computing_speed = computing_speed
        self.channel_gain = channel_gain
        self.available_time = available_time  # 可获得时间初始化为 0,表示一开始为空闲的

    def print_device_info(self):
        print(f"Device Info:")
        print(f"Coordinates: {self.coordinates}")
        print(f"Computing Speed: {self.computing_speed} FLOPS")
        print(f"Channel Gain: {self.channel_gain}")

    def calculate_computation_time(self, computation_size):
        """
        计算任务的执行时间（只考虑计算时间）
        :param computation_size: 任务的计算量（单位为 FLOP）
        :return: 任务执行时间（单位为秒）
        """
        if self.computing_speed == 0:
            raise ValueError("Computing speed must be greater than zero.")
        computation_time = computation_size / self.computing_speed
        return computation_time

    def calculate_distance(self, other_coordinates):
        """
        计算设备与另一位置的距离
        :param other_coordinates: 另一个设备的坐标 (x, y)
        :return: 距离（单位为米）
        """
        x1, y1 = self.coordinates
        x2, y2 = other_coordinates
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def calculate_transmission_time(self, data_size, other_coordinates, base_transmission_speed=100e6):
        """
        计算任务的传输时间
        :param data_size: 任务数据量（单位为字节）
        :param other_coordinates: 任务发起方的坐标 (x, y)
        :param base_transmission_speed: 基础传输速度（单位为字节每秒，默认100MB/s）
        :return: 任务传输时间（单位为秒）
        """
        distance = self.calculate_distance(other_coordinates)
        transmission_speed = base_transmission_speed / (1 + distance ** 2)
        transmission_time = data_size / transmission_speed
        return transmission_time

    def calculate_total_task_time(self, task, other_coordinates):
        """
        计算在此设备上完成任务的总时间（传输时间 + 计算时间）
        :param task: TaskNode 对象，包含任务数据量和计算量
        :param other_coordinates: 任务发起方的位置坐标 (x, y)
        :return: 完成任务所需的总时间
        """
        transmission_time = self.calculate_transmission_time(task.data_size, other_coordinates)
        computation_time = self.calculate_computation_time(task.computation_size)
        total_time = transmission_time + computation_time
        return total_time

    def calculate_local_task_time(self, task):
        """
        计算在此设备上本地完成任务的时间（只计算任务执行时间，没有传输时间）
        :param task: TaskNode 对象，包含任务的计算量
        :return: 本地完成任务所需的时间（单位为秒）
        """
        return self.calculate_computation_time(task.computation_size)

    def offload_task(self, ready_task, devices):
        """
        卸载任务到设备，更新设备的可获得时间
        :param task: TaskNode 对象，包含任务数据量和计算量
        :param other_coordinates: 任务发起方的位置坐标 (x, y)
        :return: 任务完成后的新可获得时间
        """
        best_time = float('inf')
        best_device = None
        best_device_index = -1

        task_node, app_device, arriving_time, node_id, appID = ready_task
        #寻找最优卸载设备
        for device_index, device_tuple in enumerate(devices):
            # 获取设备实例和设备信息
            other_device, device_type, device_id = device_tuple
            # 如果卸载设备号和任务所属设备号相等
            if app_device == device_index:
                task_time =other_device.calculate_local_task_time(task_node)
            else:
                # 计算任务完成所需的总时间（传输 + 计算）
                task_time = other_device.calculate_total_task_time(task_node, other_device.coordinates)

            # 更新最优设备和最小总时间
            if task_time < best_time:
                best_time = task_time
                best_device = other_device
                best_device_index = device_index

        # 将任务总时间加到设备的可获得时间上
        if best_device is not None:
            best_device.available_time += best_time
            print(
                f"Task is offloaded to Device {best_device_index} with updated available time: {best_device.available_time:.2f} seconds")
        return best_device.available_time, best_device #用于后面计算任务的响应时间

    def offload_local_task(self, task):
        task_time = self.calculate_local_task_time(task)
        return self.available_time

    def initialize_task_dag(self, task_type, args, env): #产生任务后会成为Env的一部分向Env中
        """
        初始化任务DAG。
        :param task_type: DAG类型。
        :param args: 任务DAG参数。
        """
        if task_type == 'task_type1':
            task_dag = TaskDAG(num_nodes=args.num_nodes1, data_range=args.data_range,
                               computation_range=args.computation_range,
                               deadline_range=args.deadline_range, seed=args.seed1)
        elif task_type == 'task_type2':
            task_dag = TaskDAG(num_nodes=args.num_nodes2, data_range=args.data_range,
                               computation_range=args.computation_range,
                               deadline_range=args.deadline_range, seed=args.seed2)
        elif task_type == 'task_type3':
            task_dag = TaskDAG(num_nodes=args.num_nodes3, data_range=args.data_range,
                               computation_range=args.computation_range,
                               deadline_range=args.deadline_range, seed=args.seed3)

        arrival_time = random.randint(0, 100)  # 随机生成任务到达时间
        env.tasks.append((task_dag, task_type, 0, arrival_time))  # 第一个元素是任务的DAG图，第二个元素表示任务的所属设备，第三个是属于哪个设备，第四给变量表示为任务的到达时间

# 主函数，创建设备和任务，选择最优设备进行任务卸载
if __name__ == "__main__":
    # 创建三个设备实例
    # 任务发起方设备
    task_origin_device = Device((0.0, 0.0), 1e9, 0.85, available_time=0)  # 计算速度为 1 GFLOPS
    # 两个可卸载设备
    device1 = Device((10.0, 20.0), 2e9, 0.9, available_time=0)  # 计算速度为 2 GFLOPS
    device2 = Device((30.0, 40.0), 1.5e9, 0.8, available_time=0)  # 计算速度为 1.5 GFLOPS

    # 创建任务：数据量为100MB，计算量为500 GFLOPs
    task = TaskNode(1e6, 5e11, 3600)  # 100MB 数据量，500 GFLOPs 计算量

    # 计算发起方设备本地计算时间
    local_time = task_origin_device.calculate_computation_time(task.computation_size)
    print(f"Local task time on task origin device: {local_time:.2f} seconds")

    # 计算卸载到 device1 的总时间（传输 + 计算）
    total_time_device1 = device1.calculate_total_task_time(task, task_origin_device.coordinates)
    print(f"Total time to offload task to Device 1: {total_time_device1:.2f} seconds")

    # 计算卸载到 device2 的总时间（传输 + 计算）
    total_time_device2 = device2.calculate_total_task_time(task, task_origin_device.coordinates)
    print(f"Total time to offload task to Device 2: {total_time_device2:.2f} seconds")

    # 选择最优方案
    if local_time < total_time_device1 and local_time < total_time_device2:
        print("The task will be executed locally on the origin device.")
    elif total_time_device1 < total_time_device2:
        print("The task will be offloaded to Device 1.")
    else:
        print("The task will be offloaded to Device 2.")