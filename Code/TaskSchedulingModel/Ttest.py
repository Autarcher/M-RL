from Code.SystemModel.TaskDAG import TaskDAG
from Code.SystemModel.DeviceModel import Device
from Code.SystemModel.EnvInit import EnvInit

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
        # self.pending_tasks = pending_tasks
        self.ready_tasks  #根据app_list计算就绪任务，依赖任务都已经完成

        self.cur_time = cur_time

    def __repr__(self):
        return (f"RLState: cur_time={self.cur_time}, "
                f"pending_tasks={self.pending_tasks}, ready_tasks={self.ready_tasks}, "
                f"app_list={self.app_list}")


# Example usage
# devices = ['Device1', 'Device2']
# apps = ['App1', 'App2']
# state = RLState(devices, apps, ['Task2'], ['Task2'], 0)
# print(state)

