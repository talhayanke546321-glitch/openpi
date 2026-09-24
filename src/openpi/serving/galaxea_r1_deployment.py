"""Galaxea R1 π0.5 云端推理服务的启动前自检与模型预热。

这个模块只处理 GPU 服务端，不依赖 ROS 2 或真实机器人 SDK。它在端口开始
接收真机连接之前确认：checkpoint 公布的是标准 R1 14 维协议，并用一帧合成
观测完成 JAX/PyTorch 首次推理和编译。这样真机第一次请求不会承担一次性编译
延迟，也能在服务上线前发现动作形状、NaN/Inf 或错误 checkpoint。
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import time

import numpy as np

from openpi.policies import galaxea_policy


def validate_r1_server_metadata(metadata: Mapping[str, object]) -> None:
    """确认服务端 metadata 足以唯一描述当前 R1 策略协议。"""
    expected = {
        "protocol_version": "galaxea-r1-openpi-v1",
        "robot": "r1",
        "controller": "bimanual_joint_position",
        "action_dim": galaxea_policy.GALAXEA_STATE_DIM,
        "action_horizon": galaxea_policy.GALAXEA_ACTION_HORIZON,
        "execute_horizon": galaxea_policy.GALAXEA_EXECUTE_HORIZON,
        "control_frequency_hz": 15,
        "image_keys": list(galaxea_policy.GALAXEA_IMAGE_NAMES),
    }
    missing = [key for key in expected if key not in metadata]
    if missing:
        raise ValueError(f"R1 policy metadata is missing required fields: {missing}")

    mismatches = {key: (metadata[key], value) for key, value in expected.items() if metadata[key] != value}
    if mismatches:
        raise ValueError(f"R1 policy metadata does not match the cloud deployment contract: {mismatches}")

    state_order = metadata.get("state_order")
    expected_order = list(galaxea_policy.GALAXEA_STATE_ORDER)
    if state_order != expected_order:
        raise ValueError(f"R1 state_order must be {expected_order}, got {state_order}")


def make_warmup_observation(prompt: str) -> dict[str, object]:
    """构造一帧满足在线契约的合成观测。

    合成图像和零状态只用于触发模型编译，不用于评估策略质量。真实客户端仍需
    提供三路 RGB HWC uint8 图像、14 维 float32 状态和明确任务指令。
    """
    if not prompt.strip():
        raise ValueError("warmup prompt must not be empty")
    height, width = 224, 224
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return {
        "images": {name: image.copy() for name in galaxea_policy.GALAXEA_IMAGE_NAMES},
        "state": np.zeros(galaxea_policy.GALAXEA_STATE_DIM, dtype=np.float32),
        "prompt": prompt,
    }


def validate_r1_inference_result(result: object) -> np.ndarray:
    """校验一次服务端推理是否返回有限的 ``15×14`` 动作块。"""
    if not isinstance(result, dict) or "actions" not in result:
        raise ValueError("R1 policy result must be a dictionary containing 'actions'")
    actions = np.asarray(result["actions"], dtype=np.float32)
    expected_shape = (galaxea_policy.GALAXEA_ACTION_HORIZON, galaxea_policy.GALAXEA_STATE_DIM)
    if actions.shape != expected_shape:
        raise ValueError(f"R1 policy returned actions with shape {actions.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("R1 policy returned NaN/Inf during warmup")
    return actions


def warmup_policy(policy: object, *, prompt: str) -> float:
    """执行一次合成推理并返回耗时秒数。"""
    metadata = getattr(policy, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Policy has no metadata mapping")
    validate_r1_server_metadata(metadata)

    logging.info("Warming up Galaxea R1 policy before opening the WebSocket port...")
    start = time.monotonic()
    result = policy.infer(make_warmup_observation(prompt))
    elapsed = time.monotonic() - start
    actions = validate_r1_inference_result(result)
    logging.info(
        "Galaxea R1 policy warmup completed in %.3f s; action shape=%s, range=[%.5f, %.5f]",
        elapsed,
        actions.shape,
        float(actions.min()),
        float(actions.max()),
    )
    return elapsed


__all__ = [
    "make_warmup_observation",
    "validate_r1_inference_result",
    "validate_r1_server_metadata",
    "warmup_policy",
]
