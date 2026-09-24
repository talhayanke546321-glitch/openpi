"""以云端安全默认值启动 Galaxea R1 π0.5 推理服务。

这个入口固定使用 checkpoint 模式、要求 API key、默认监听 127.0.0.1，并在
开放端口前完成一次模型预热。公网接入应由 Caddy/Nginx 把 ``wss://`` 转发到
本进程；VPN 场景可以显式传 ``--host 0.0.0.0``，再用安全组限制来源地址。
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import tyro

if __package__:
    from scripts import serve_policy
else:
    # Direct execution (`python scripts/serve_galaxea_r1_cloud.py`) places the
    # scripts directory, rather than the repository root, on sys.path.
    import serve_policy


DEFAULT_CHECKPOINT = (
    "checkpoints/pi05_galaxea_r1_multitask_lora/"
    "multi_asset_v2_lora_8k/7999"
)


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = DEFAULT_CHECKPOINT
    config: str = "pi05_galaxea_r1_multitask_lora"
    host: str = "127.0.0.1"
    port: int = 8000
    api_key_env: str = "OPENPI_API_KEY"
    max_request_size_mib: float = 32.0
    warmup_prompt: str = "pick up the object"
    record: bool = False


def _validate_checkpoint_layout(checkpoint_dir: Path) -> None:
    """在加载 9GB 权重前检查 checkpoint 的必要文件是否齐全。"""
    required = (
        checkpoint_dir / "params",
        checkpoint_dir / "assets" / "galaxea_r1_multi_asset_v1" / "norm_stats.json",
        checkpoint_dir / "assets" / "galaxea_r1_multi_asset_v1" / "norm_stats_metadata.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete Galaxea R1 checkpoint; missing: {missing}")


def main(args: Args) -> None:
    """执行 checkpoint/API key 预检，然后交给通用 OpenPI server。"""
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    _validate_checkpoint_layout(checkpoint_dir)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Cloud deployment requires a non-empty {args.api_key_env} environment variable. "
            "Store it in a root-readable systemd EnvironmentFile; do not pass it on the command line."
        )

    serve_policy.main(
        serve_policy.Args(
            host=args.host,
            port=args.port,
            api_key_env=args.api_key_env,
            allow_unauthenticated=False,
            max_request_size_mib=args.max_request_size_mib,
            warmup=True,
            warmup_prompt=args.warmup_prompt,
            expose_errors=False,
            record=args.record,
            policy=serve_policy.Checkpoint(config=args.config, dir=str(checkpoint_dir)),
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
