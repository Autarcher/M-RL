import math
from TaskModel import TaskNode

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

    def offload_task(self, task, other_coordinates):
        """
        卸载任务到设备，更新设备的可获得时间
        :param task: TaskNode 对象，包含任务数据量和计算量
        :param other_coordinates: 任务发起方的位置坐标 (x, y)
        :return: 任务完成后的新可获得时间
        """
        # 计算任务完成所需的总时间（传输 + 计算）
        task_time = self.calculate_total_task_time(task, other_coordinates)

        # 将任务总时间加到设备的可获得时间上
        self.available_time += task_time
        return self.available_time

    def offload_local_task(self, task):
        task_time = self.calculate_local_task_time(task)
        self.available_time += task_time

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