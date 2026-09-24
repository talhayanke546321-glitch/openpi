"""OpenPI 客户端的通用 episode 调度循环。

Runtime 不关心动作来自本地模型还是 WebSocket，也不关心环境是机器人
还是 SAPIEN。它只执行统一的三步协议：读取 observation、向 Agent 请求
action、把 action 应用到 Environment。Galaxea 的具体字段转换由环境和
policy adapter 完成，因此这里保持通用。
"""

import logging
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class Runtime:
    """按固定频率协调 Environment、Agent 和可选 Subscriber。"""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        """保存 episode 数量、最大步数和控制频率配置。"""
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0

    def run(self) -> None:
        """连续运行指定数量的 episode，并在最后 reset 环境。"""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """在后台线程中启动 ``run``，返回线程句柄。"""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """让当前 episode 的 while 循环在本轮结束后退出。"""
        self._in_episode = False

    def _run_episode(self) -> None:
        """执行一个 episode 的 reset、agent reset 和按频率控制循环。"""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """执行一次 ``observation -> action -> apply_action`` 闭环迭代。"""
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
