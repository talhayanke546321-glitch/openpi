"""Galaxea R1 Pro 仿真器与 OpenPI π0.5 之间的数据变换。

R1 Pro 与标准 R1 最重要的区别是每只手臂有 7 个关节，因此在线状态和
物理动作均为 16 维：

``[左臂 7 关节, 左夹爪, 右臂 7 关节, 右夹爪]``。

本模块只描述 GalaxeaManipSim 中 R1 Pro 的关节位置控制语义。仿真夹爪的
范围仍是 0.00~0.05 米（0 闭合、0.05 张开）；真机 SDK 使用 0~100 行程，
必须由真机 ROS2 适配层单独转换，不能直接复用这里的米制边界。

π0.5 内部动作缓冲区保持 32 维，以便直接从 ``pi05_base`` 初始化。这里
只负责在模型边界补齐/裁剪：物理接口始终是 16 维，padding 维度绝不发送
给仿真器。
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms

R1_PRO_ARM_DOF = 7
R1_PRO_STATE_DIM = 2 * (R1_PRO_ARM_DOF + 1)
R1_PRO_MODEL_ACTION_DIM = 32
R1_PRO_GRIPPER_MAX = 0.05
R1_PRO_GRIPPER_INDICES = (R1_PRO_ARM_DOF, R1_PRO_STATE_DIM - 1)
R1_PRO_IMAGE_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
R1_PRO_STATE_ORDER = (
    *(f"left_arm_joint{i}" for i in range(1, R1_PRO_ARM_DOF + 1)),
    "left_gripper",
    *(f"right_arm_joint{i}" for i in range(1, R1_PRO_ARM_DOF + 1)),
    "right_gripper",
)
R1_PRO_ACTION_HORIZON = 15
R1_PRO_EXECUTE_HORIZON = 10
R1_PRO_RECORDING_CONTRACT = "pre_action_v1"
R1_PRO_IMAGE_PREPROCESS_CONTRACT = "resize_with_pad_pil_bilinear_v1"
NORM_STATS_METADATA_FILENAME = "norm_stats_metadata.json"


def _finite_array(value: object, *, name: str, minimum_dim: int, exact: bool) -> np.ndarray:
    """把输入变成 float32，并校验最后一维和有限值。"""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, got scalar")
    valid = array.shape[-1] == minimum_dim if exact else array.shape[-1] >= minimum_dim
    if not valid:
        relation = "exactly" if exact else "at least"
        raise ValueError(f"{name} must have {relation} {minimum_dim} values, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _convert_image(image: object, *, name: str) -> np.ndarray:
    """把 CHW/HWC、float/uint8 图片统一成 RGB HWC uint8。"""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3-D image, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"{name} must have three colour channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        if array.size and float(np.max(array)) <= 1.0 + 1e-6:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _sim_grippers_to_model(value: np.ndarray) -> np.ndarray:
    """把仿真米制开度转换成模型的 0 张开、1 闭合语义。"""
    result = value.copy()
    for index in R1_PRO_GRIPPER_INDICES:
        result[..., index] = 1.0 - np.clip(result[..., index] / R1_PRO_GRIPPER_MAX, 0.0, 1.0)
    return result


def _model_grippers_to_sim(value: np.ndarray) -> np.ndarray:
    """把模型夹爪语义转换回仿真器使用的 0~0.05 米开度。"""
    result = value.copy()
    for index in R1_PRO_GRIPPER_INDICES:
        result[..., index] = (1.0 - np.clip(result[..., index], 0.0, 1.0)) * R1_PRO_GRIPPER_MAX
    return result


def make_delta_action_mask() -> tuple[bool, ...]:
    """仅把左右 7 个手臂关节转成 delta action，夹爪保持绝对值。"""
    return transforms.make_bool_mask(R1_PRO_ARM_DOF, -1, R1_PRO_ARM_DOF, -1)


def make_norm_stats_metadata(*, action_horizon: int, use_delta_joint_actions: bool) -> dict[str, object]:
    """记录 R1 Pro 统计量的机器人、维度和变换来源。"""
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    return {
        "schema_version": 1,
        "robot": "r1_pro",
        "controller": "bimanual_joint_position",
        "action_dim": R1_PRO_STATE_DIM,
        "action_horizon": action_horizon,
        "execute_horizon": R1_PRO_EXECUTE_HORIZON,
        "use_delta_joint_actions": use_delta_joint_actions,
        "gripper_convention": "sim_closed_open_meters_to_model_closed_open_unit",
        "recording_contract": R1_PRO_RECORDING_CONTRACT,
        "image_preprocess": R1_PRO_IMAGE_PREPROCESS_CONTRACT,
    }


def make_dataset_metadata() -> dict[str, object]:
    """返回必须出现在 R1 Pro LeRobot 数据集中的来源契约。"""
    return {
        "galaxea_recording_contract": R1_PRO_RECORDING_CONTRACT,
        "galaxea_controller_type": "bimanual_joint_position",
        "galaxea_image_preprocess": R1_PRO_IMAGE_PREPROCESS_CONTRACT,
        "galaxea_robot": "r1_pro",
        "galaxea_action_dim": R1_PRO_STATE_DIM,
        "galaxea_control_freq": 15,
        "galaxea_camera_resolution_scale": 4,
    }


def _validate_metadata(metadata: Mapping[str, object], expected: Mapping[str, object], *, name: str) -> None:
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"R1 Pro {name} metadata does not match the contract: {mismatches}")


def validate_dataset_metadata(metadata: Mapping[str, object], expected: Mapping[str, object]) -> None:
    """拒绝其它机器人、控制器或采样频率的数据集。"""
    _validate_metadata(metadata, expected, name="dataset")


def validate_norm_stats_metadata(metadata: Mapping[str, object], expected: Mapping[str, object]) -> None:
    """拒绝不属于当前 R1 Pro 动作变换的归一化统计。"""
    _validate_metadata(metadata, expected, name="normalization-stat")


@dataclasses.dataclass(frozen=True)
class GalaxeaR1ProInputs(transforms.DataTransformFn):
    """把在线/数据集中的 R1 Pro 观测转换成 OpenPI canonical input。"""

    EXPECTED_IMAGES: ClassVar[tuple[str, ...]] = R1_PRO_IMAGE_NAMES

    def __call__(self, data: dict) -> dict:
        images = data.get("images")
        if not isinstance(images, dict):
            raise ValueError('R1 Pro input must contain an "images" dictionary')
        missing = set(self.EXPECTED_IMAGES) - set(images)
        if missing:
            raise ValueError(f"R1 Pro input is missing images: {sorted(missing)}")

        state = _sim_grippers_to_model(
            _finite_array(data["state"], name="state", minimum_dim=R1_PRO_STATE_DIM, exact=True)
        )
        converted = {name: _convert_image(images[name], name=name) for name in self.EXPECTED_IMAGES}
        result = {
            "image": {
                "base_0_rgb": converted["cam_high"],
                "left_wrist_0_rgb": converted["cam_left_wrist"],
                "right_wrist_0_rgb": converted["cam_right_wrist"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }
        if "actions" in data:
            result["actions"] = _sim_grippers_to_model(
                _finite_array(data["actions"], name="actions", minimum_dim=R1_PRO_STATE_DIM, exact=True)
            )
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class GalaxeaR1ProOutputs(transforms.DataTransformFn):
    """裁剪 π0.5 的 32 维 padding，并恢复 R1 Pro 仿真夹爪单位。"""

    def __call__(self, data: dict) -> dict:
        actions = _finite_array(
            data["actions"], name="actions", minimum_dim=R1_PRO_STATE_DIM, exact=False
        )[..., :R1_PRO_STATE_DIM]
        return {"actions": _model_grippers_to_sim(actions)}


__all__ = [
    "NORM_STATS_METADATA_FILENAME",
    "R1_PRO_ACTION_HORIZON",
    "R1_PRO_ARM_DOF",
    "R1_PRO_EXECUTE_HORIZON",
    "R1_PRO_GRIPPER_INDICES",
    "R1_PRO_GRIPPER_MAX",
    "R1_PRO_IMAGE_NAMES",
    "R1_PRO_MODEL_ACTION_DIM",
    "R1_PRO_STATE_DIM",
    "R1_PRO_STATE_ORDER",
    "GalaxeaR1ProInputs",
    "GalaxeaR1ProOutputs",
    "make_dataset_metadata",
    "make_delta_action_mask",
    "make_norm_stats_metadata",
    "validate_dataset_metadata",
    "validate_norm_stats_metadata",
]
