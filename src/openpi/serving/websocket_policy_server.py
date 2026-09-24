"""OpenPI 策略 WebSocket 服务端。

服务端只负责网络传输和调用一个已经加载好的 ``BasePolicy``：连接建立后
先发送 metadata，随后循环接收观测、调用 ``policy.infer``、附加推理耗时
并返回动作。模型加载、checkpoint 校验和 transform pipeline 位于
``openpi.policies.policy_config``，不在本文件中重复实现。
"""

import asyncio
import hmac
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """通过 WebSocket 暴露一个已经构造好的策略。

    Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        api_key: str | None = None,
        max_request_size: int = 32 * 1024 * 1024,
        ping_interval: float | None = 20,
        ping_timeout: float | None = 60,
        expose_errors: bool = False,
    ) -> None:
        """保存策略、监听地址、鉴权配置和握手 metadata。

        Args:
            policy: 已经完成 checkpoint/transform 装配的策略。
            host: WebSocket 监听地址。公网部署建议监听 ``127.0.0.1``，由
                Caddy/Nginx 提供 TLS；只有在 VPN 或防火墙已配置时才直接
                监听 ``0.0.0.0``。
            port: 监听端口。
            metadata: 客户端连接后收到的协议元数据。
            api_key: 可选共享密钥。配置后，除 ``/healthz`` 外的连接必须
                携带 ``Authorization: Api-Key <key>``。
            max_request_size: 单条 WebSocket 请求的最大字节数，防止无限大
                图像消息耗尽服务端内存。
            ping_interval: WebSocket 心跳间隔；``None`` 表示禁用。
            ping_timeout: 等待 pong 的最长时间。
            expose_errors: 是否把完整 traceback 返回客户端。公网服务应保持
                ``False``，详细错误只写服务器日志。
        """
        if port is not None and not 0 < port < 65536:
            raise ValueError(f"port must be between 1 and 65535, got {port}")
        if max_request_size <= 0:
            raise ValueError("max_request_size must be positive")
        if api_key == "":
            raise ValueError("api_key must not be empty")
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._api_key = api_key
        self._max_request_size = max_request_size
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._expose_errors = expose_errors
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """启动 asyncio 服务并阻塞等待客户端连接。"""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            # systemd 的 KillSignal=SIGINT 和交互式 Ctrl-C 都走这里。服务在
            # asyncio context manager 中已经关闭 socket，无需把正常停机打印
            # 成 traceback 或让 systemd 误判为程序故障。
            logger.info("Policy server stopped")

    async def run(self):
        """创建 WebSocket server，并为每条连接注册 handler。"""
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_request_size,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            process_request=self._process_request,
            # 不在 HTTP 响应中暴露 Python/websockets 版本。
            server_header=None,
        ) as server:
            await server.serve_forever()

    def _process_request(
        self,
        connection: _server.ServerConnection,
        request: _server.Request,
    ) -> _server.Response | None:
        """处理健康检查，并在 WebSocket upgrade 前完成 API key 鉴权。"""
        return _process_request(connection, request, api_key=self._api_key)

    async def _handler(self, websocket: _server.ServerConnection):
        """处理一条客户端连接的 metadata、推理请求和异常。"""
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                error_traceback = traceback.format_exc()
                logger.exception("Policy inference failed for %s", websocket.remote_address)
                client_message = error_traceback if self._expose_errors else "Policy inference failed"
                await websocket.send(client_message)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """向后兼容的无鉴权健康检查回调。"""
    return _process_request(connection, request, api_key=None)


def _process_request(
    connection: _server.ServerConnection,
    request: _server.Request,
    *,
    api_key: str | None,
) -> _server.Response | None:
    """返回健康检查/鉴权响应；``None`` 表示继续 WebSocket 握手。"""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")

    if api_key is not None:
        authorization = request.headers.get("Authorization", "")
        expected = f"Api-Key {api_key}"
        if not hmac.compare_digest(authorization, expected):
            response = connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
            response.headers["WWW-Authenticate"] = "Api-Key"
            return response

    # Continue with the normal request handling.
    return None
