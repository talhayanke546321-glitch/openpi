"""从网络边界验证 Galaxea R1 π0.5 服务的握手和一次完整推理。"""

from __future__ import annotations

import dataclasses
import logging
import os
import time

from openpi_client import websocket_client_policy
import tyro

from openpi.serving import galaxea_r1_deployment


@dataclasses.dataclass
class Args:
    # 本机服务使用 127.0.0.1；TLS 反向代理可传 wss://pi05.example.com。
    host: str = "127.0.0.1"
    # 使用反向代理默认 443 端口时传 None，脚本不会在 URL 后追加端口。
    port: int | None = 8000
    # 客户端从该环境变量读取共享密钥。
    api_key_env: str = "OPENPI_API_KEY"
    # 仅用于合成 smoke 观测，不代表真实机器人任务。
    prompt: str = "pick up the object"


def main(args: Args) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Set non-empty {args.api_key_env} before running the authenticated smoke test")

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=api_key,
    )
    metadata = client.get_server_metadata()
    galaxea_r1_deployment.validate_r1_server_metadata(metadata)

    start = time.monotonic()
    result = client.infer(galaxea_r1_deployment.make_warmup_observation(args.prompt))
    elapsed = time.monotonic() - start
    actions = galaxea_r1_deployment.validate_r1_inference_result(result)
    logging.info(
        "Smoke test passed: protocol=%s, actions=%s, round_trip=%.3f s, server_timing=%s",
        metadata["protocol_version"],
        actions.shape,
        elapsed,
        result.get("server_timing"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
