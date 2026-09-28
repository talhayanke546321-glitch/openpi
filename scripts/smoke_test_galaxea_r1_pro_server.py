"""Validate R1 Pro policy metadata and one 15x16 network inference."""

from __future__ import annotations

import dataclasses
import logging
import os
import time

from openpi_client import websocket_client_policy
import tyro

from openpi.serving import galaxea_r1_pro_deployment


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int | None = 8000
    api_key_env: str = "OPENPI_API_KEY"
    prompt: str = "pick up the bottle and place it on the plate"


def main(args: Args) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Set non-empty {args.api_key_env} before running the smoke test")

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=api_key,
    )
    metadata = client.get_server_metadata()
    galaxea_r1_pro_deployment.validate_server_metadata(metadata)

    started = time.monotonic()
    result = client.infer(
        galaxea_r1_pro_deployment.make_warmup_observation(args.prompt)
    )
    elapsed = time.monotonic() - started
    actions = galaxea_r1_pro_deployment.validate_inference_result(result)
    logging.info(
        "R1 Pro smoke test passed: protocol=%s, actions=%s, round_trip=%.3fs, "
        "range=[%.5f, %.5f], server_timing=%s",
        metadata["protocol_version"],
        actions.shape,
        elapsed,
        float(actions.min()),
        float(actions.max()),
        result.get("server_timing"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
