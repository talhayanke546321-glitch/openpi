"""动作块缓存器：把一次策略预测拆成多个连续控制动作。

π0.5 不只预测下一步，而是一次返回一段 action chunk。Broker 在 chunk
没有耗尽前不会再次调用模型，减少昂贵的图像编码和模型推理；耗尽后才
用最新观测请求下一段动作。Galaxea 项目把模型预测长度设为 15、执行
长度设为 10，因此每 10 个控制周期会触发一次重新规划。
"""

from typing import Dict

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """将策略返回的动作块逐步输出为单步动作。

    Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        """保存底层策略和本次要消费的 chunk 长度。"""
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """获取当前 chunk 的下一步动作，必要时触发一次新推理。

        只有 ``actions`` 这个字段沿第一维切片；``state``、timing 等其它
        NumPy 元数据必须保持原样，否则会被错误地切成标量或第一个时间步。
        """
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        if not isinstance(self._last_results, dict):
            raise ValueError("action chunk policy response must be a dictionary")

        actions = self._last_results.get("actions")
        if not isinstance(actions, np.ndarray):
            raise ValueError("action chunk policy response must contain a NumPy 'actions' array")
        if actions.ndim < 1 or actions.shape[0] < self._action_horizon:
            raise ValueError(
                "action chunk is shorter than the broker horizon: "
                f"shape={actions.shape}, horizon={self._action_horizon}"
            )

        # Only the action sequence is chunked.  A policy response can also
        # contain the current state, timing, or other metadata; indexing every
        # NumPy leaf would silently turn a state vector into a scalar.
        results = dict(self._last_results)
        results["actions"] = actions[self._cur_step, ...]
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        """在 episode 边界丢弃旧动作块，避免跨任务复用动作。"""
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
