import numpy as np
import pytest

from openpi_client import action_chunk_broker
from openpi_client import base_policy


class _Policy(base_policy.BasePolicy):
    def __init__(self, result):
        self.result = result

    def infer(self, obs):
        return self.result


def test_broker_only_slices_actions() -> None:
    actions = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
    state = np.arange(14, dtype=np.float32)
    broker = action_chunk_broker.ActionChunkBroker(
        _Policy({"actions": actions, "state": state, "policy_timing": {"infer_ms": 1.0}}),
        action_horizon=2,
    )

    first = broker.infer({})
    second = broker.infer({})
    np.testing.assert_array_equal(first["actions"], actions[0])
    np.testing.assert_array_equal(second["actions"], actions[1])
    np.testing.assert_array_equal(first["state"], state)
    np.testing.assert_array_equal(second["state"], state)


def test_broker_rejects_short_chunks() -> None:
    policy = _Policy({"actions": np.zeros((1, 2), dtype=np.float32)})
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)
    with pytest.raises(ValueError, match="shorter"):
        broker.infer({})
