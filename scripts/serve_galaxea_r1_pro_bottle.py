"""Serve the 10k R1 Pro bottle-place checkpoint with deployment-safe defaults."""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import tyro

if __package__:
    from scripts import serve_policy
else:
    import serve_policy


DEFAULT_CHECKPOINT = (
    "checkpoints/pi05_galaxea_r1_pro_bottle_place_lora/"
    "r1pro_bottle_place_high_v1_bs32_10k/10000"
)
ASSET_ID = Path("galaxea/R1ProBottlePickPlace-v0")


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = DEFAULT_CHECKPOINT
    config: str = "pi05_galaxea_r1_pro_bottle_place_lora"
    host: str = "127.0.0.1"
    port: int = 8000
    api_key_env: str = "OPENPI_API_KEY"
    max_request_size_mib: float = 32.0
    warmup_prompt: str = "pick up the bottle and place it on the plate"
    record: bool = False


def _validate_checkpoint_layout(checkpoint_dir: Path) -> None:
    required = (
        checkpoint_dir / "params",
        checkpoint_dir / "_CHECKPOINT_METADATA",
        checkpoint_dir / "assets" / ASSET_ID / "norm_stats.json",
        checkpoint_dir / "assets" / ASSET_ID / "norm_stats_metadata.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete R1 Pro bottle-place checkpoint; missing: {missing}")


def main(args: Args) -> None:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    _validate_checkpoint_layout(checkpoint_dir)
    if not os.environ.get(args.api_key_env):
        raise ValueError(
            f"Set a non-empty {args.api_key_env} before starting the R1 Pro policy server"
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
