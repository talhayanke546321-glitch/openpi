import numpy as np
import pytest

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


class _FakeConnection:
    def __init__(self) -> None:
        self.timeout = None

    def send(self, data: bytes) -> None:
        assert isinstance(data, bytes)

    def recv(self, timeout=None) -> bytes:
        self.timeout = timeout
        return msgpack_numpy.packb({"actions": np.zeros((1, 2), dtype=np.float32)})


def test_infer_passes_response_timeout_to_websocket(monkeypatch) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(
        websocket_client_policy.WebsocketClientPolicy,
        "_wait_for_server",
        lambda self: (connection, {}),
    )
    policy = websocket_client_policy.WebsocketClientPolicy(response_timeout=1.25)
    result = policy.infer({"state": np.zeros(2, dtype=np.float32)})
    assert connection.timeout == 1.25
    assert result["actions"].shape == (1, 2)


def test_response_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="response_timeout"):
        websocket_client_policy.WebsocketClientPolicy(response_timeout=0.0)
