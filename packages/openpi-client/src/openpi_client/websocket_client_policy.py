"""OpenPI WebSocket 策略客户端。

客户端把它自己伪装成一个普通 ``BasePolicy``：上层只调用 ``infer``，
不需要知道策略实际运行在本地进程、远程机器还是容器中。网络协议使用
msgpack + NumPy 扩展编码，以便高效传输图像数组和动作数组。
"""

import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """通过持久 WebSocket 连接调用远程策略。

    Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        """拼接服务 URI、创建编码器并等待策略服务上线。"""
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        """返回建连时服务端发送的机器人/动作协议元数据。"""
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """循环等待服务端，并读取连接建立后的第一条 metadata 消息。"""
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    # A Pi05 JAX server may spend several minutes compiling
                    # the first inference graph.  Keep the connection open
                    # during that one-time warm-up instead of treating the
                    # missing response as a dead server.
                    ping_interval=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """序列化观测、发送二进制消息、等待并解码动作响应。"""
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        """网络策略本身没有本地状态；episode 状态由 Broker/Runtime 管理。"""
        pass
