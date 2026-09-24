"""Galaxea R1 Pro π0.5 服务启动前的协议自检和模型预热。"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import time

import numpy as np

from openpi.policies import galaxea_r1_pro_policy


def validate_server_metadata(metadata: Mapping[str, object]) -> None:
    """确认服务端公布的是 R1 Pro 16 维关节位置控制协议。"""
    expected = {
        "protocol_version": "galaxea-r1-pro-openpi-v1",
        "robot": "r1_pro",
        "controller": "bimanual_joint_position",
        "action_dim": galaxea_r1_pro_policy.R1_PRO_STATE_DIM,
        "action_horizon": galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON,
        "execute_horizon": galaxea_r1_pro_policy.R1_PRO_EXECUTE_HORIZON,
        "control_frequency_hz": 15,
        "image_keys": list(galaxea_r1_pro_policy.R1_PRO_IMAGE_NAMES),
        "state_order": list(galaxea_r1_pro_policy.R1_PRO_STATE_ORDER),
    }
    missing = [key for key in expected if key not in metadata]
    if missing:
        raise ValueError(f"R1 Pro policy metadata is missing required fields: {missing}")
    mismatches = {key: (metadata[key], value) for key, value in expected.items() if metadata[key] != value}
    if mismatches:
        raise ValueError(f"R1 Pro policy metadata does not match the deployment contract: {mismatches}")


def make_warmup_observation(prompt: str) -> dict[str, object]:
    """构造三路 224x224 RGB、16维状态和任务文本的合成观测。"""
    if not prompt.strip():
        raise ValueError("warmup prompt must not be empty")
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "images": {name: image.copy() for name in galaxea_r1_pro_policy.R1_PRO_IMAGE_NAMES},
        "state": np.zeros(galaxea_r1_pro_policy.R1_PRO_STATE_DIM, dtype=np.float32),
        "prompt": prompt,
    }


def validate_inference_result(result: object) -> np.ndarray:
    """要求预热输出严格为有限的 15x16 动作块。"""
    if not isinstance(result, dict) or "actions" not in result:
        raise ValueError("R1 Pro policy result must contain 'actions'")
    actions = np.asarray(result["actions"], dtype=np.float32)
    expected = (
        galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON,
        galaxea_r1_pro_policy.R1_PRO_STATE_DIM,
    )
    if actions.shape != expected:
        raise ValueError(f"R1 Pro policy returned {actions.shape}, expected {expected}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("R1 Pro policy returned NaN/Inf during warmup")
    return actions


def warmup_policy(policy: object, *, prompt: str) -> float:
    """在监听端口前完成一次 R1 Pro 推理和 JAX/PyTorch 编译。"""
    metadata = getattr(policy, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Policy has no metadata mapping")
    validate_server_metadata(metadata)
    logging.info("Warming up Galaxea R1 Pro policy before opening the WebSocket port...")
    start = time.monotonic()
    actions = validate_inference_result(policy.infer(make_warmup_observation(prompt)))
    elapsed = time.monotonic() - start
    logging.info(
        "R1 Pro warmup completed in %.3f s; action shape=%s, range=[%.5f, %.5f]",
        elapsed,
        actions.shape,
        float(actions.min()),
        float(actions.max()),
    )
    return elapsed


__all__ = ["make_warmup_observation", "validate_inference_result", "validate_server_metadata", "warmup_policy"]
