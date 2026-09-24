"""OpenPI Runtime 对真实机器人/仿真环境的最小抽象接口。"""

import abc


class Environment(abc.ABC):
    """表示机器人及其所处世界的可交互环境。

    An Environment represents the robot and the environment it inhabits.

    The primary contract of environments is that they can be queried for observations
    about their state, and have actions applied to them to change that state.
    """

    @abc.abstractmethod
    def reset(self) -> None:
        """重置环境，并准备第一帧可读取的观测。

        This will be called once before starting each episode.
        """

    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        """告诉 Runtime 当前 episode 是否成功结束、失败或超时。

        This will be called after each step. It should return `True` if the episode is
        complete (either successfully or unsuccessfully), and `False` otherwise.
        """

    @abc.abstractmethod
    def get_observation(self) -> dict:
        """读取当前状态并转换成 Agent 能理解的观测字典。"""

    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        """把 Agent 的动作应用到真实机器人或仿真环境。"""
