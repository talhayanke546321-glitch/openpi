"""Galaxea R1 双臂仿真器与 OpenPI π0.5 之间的数据变换。

Data transforms for the Galaxea R1 bimanual simulator.

The simulator and the OpenPI model deliberately use different gripper
conventions.  This module is the single place where that conversion happens
for both LeRobot data and online inference:

* simulator state/action: ``[left 6 joints, left gripper, right 6 joints,
  right gripper]``;
* simulator grippers: ``0.0`` is closed and ``0.05`` is open (metres);
* model grippers: ``0.0`` is open and ``1.0`` is closed.

手臂关节在这个边界上保持绝对弧度值。``LeRobotGalaxeaDataConfig`` 会在
后续 transform 中只把 12 个手臂关节转换为 delta action，两个夹爪仍保持
绝对值。这样可以同时满足：仿真器需要绝对关节目标，OpenPI 训练使用更
适合模型学习的关节增量。

这份文件是整个项目最重要的“语义翻译层”之一。在线推理和离线训练都
必须走同一套夹爪、图片、字段拼接和动作裁剪逻辑，否则即使 tensor 形状
相同，模型看到的物理含义也会不一致。
"""

from collections.abc import Mapping
import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms

GALAXEA_STATE_DIM = 14
GALAXEA_ARM_DOF = 6
GALAXEA_GRIPPER_MAX = 0.05
GALAXEA_GRIPPER_INDICES = (6, 13)
GALAXEA_MODEL_ACTION_DIM = 32
GALAXEA_IMAGE_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
GALAXEA_STATE_ORDER = (
    *(f"left_arm_joint{i}" for i in range(1, GALAXEA_ARM_DOF + 1)),
    "left_gripper",
    *(f"right_arm_joint{i}" for i in range(1, GALAXEA_ARM_DOF + 1)),
    "right_gripper",
)
GALAXEA_ACTION_HORIZON = 15
GALAXEA_EXECUTE_HORIZON = 10
NORM_STATS_METADATA_FILENAME = "norm_stats_metadata.json"
GALAXEA_RECORDING_CONTRACT = "pre_action_v1"
GALAXEA_IMAGE_PREPROCESS_CONTRACT = "resize_with_pad_pil_bilinear_v1"


def _as_last_dim_array(value: object, *, name: str, dim: int) -> np.ndarray:
    """检查数组最后一维恰好是指定维度，并拒绝 NaN/Inf。"""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != dim:
        raise ValueError(f"{name} must have shape (..., {dim}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_at_least_dim_array(value: object, *, name: str, dim: int) -> np.ndarray:
    """检查数组最后一维至少包含指定维度，用于裁剪模型输出。"""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] < dim:
        raise ValueError(f"{name} must have at least {dim} values in its last dimension, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _convert_image(image: object, *, name: str) -> np.ndarray:
    """把 CHW/HWC、float/uint8 图片统一成 RGB HWC uint8。

    Return an RGB HWC uint8 image.

    LeRobot's image loader may return CHW tensors, while the online Galaxea
    client sends HWC images.  Accepting both here keeps the dataset and online
    paths aligned without applying a second, inconsistent image conversion.
    """

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3-D image, got {array.shape}")

    # Convert CHW to HWC when the channel dimension is unambiguously first.
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))

    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"{name} must have three colour channels, got {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        # Support both the common [0, 1] and [0, 255] float conventions.
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0 + 1e-6:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _convert_sim_grippers_to_model(value: np.ndarray) -> np.ndarray:
    """把仿真夹爪米制值转换为模型的 0~1 闭合语义。

    仿真中 0.00 m 表示闭合、0.05 m 表示张开；模型中 1 表示闭合、0
    表示张开，因此需要先除以最大开度，再做 ``1 - x``。
    """
    result = value.copy()
    for index in GALAXEA_GRIPPER_INDICES:
        result[..., index] = 1.0 - np.clip(result[..., index] / GALAXEA_GRIPPER_MAX, 0.0, 1.0)
    return result


def _convert_model_grippers_to_sim(value: np.ndarray) -> np.ndarray:
    """把模型夹爪值反变换回仿真器使用的米制开度。"""
    result = value.copy()
    for index in GALAXEA_GRIPPER_INDICES:
        result[..., index] = (1.0 - np.clip(result[..., index], 0.0, 1.0)) * GALAXEA_GRIPPER_MAX
    return result


def make_delta_action_mask() -> tuple[bool, ...]:
    """返回 delta-action mask。

    mask 为 True 的 12 个手臂关节会经过 ``DeltaActions``；两个夹爪位置
    为 -1，表示保持绝对值，不参与状态差分。
    """

    return transforms.make_bool_mask(GALAXEA_ARM_DOF, -1, GALAXEA_ARM_DOF, -1)


def make_norm_stats_metadata(*, action_horizon: int, use_delta_joint_actions: bool) -> dict[str, object]:
    """生成 normalization stats 的来源说明文件内容。

    单独保存 metadata 是必要的，因为普通 stats 数值本身无法说明它是按
    15 步还是 50 步 action horizon、是否做过 delta action、采用哪种夹爪
    语义计算得到的。
    """

    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    return {
        "schema_version": 1,
        "robot": "r1",
        "controller": "bimanual_joint_position",
        "action_dim": GALAXEA_STATE_DIM,
        "action_horizon": action_horizon,
        "execute_horizon": GALAXEA_EXECUTE_HORIZON,
        "use_delta_joint_actions": use_delta_joint_actions,
        "gripper_convention": "sim_closed_open_meters_to_model_closed_open_unit",
        "recording_contract": GALAXEA_RECORDING_CONTRACT,
        "image_preprocess": GALAXEA_IMAGE_PREPROCESS_CONTRACT,
    }


def make_dataset_metadata() -> dict[str, object]:
    """生成写入 LeRobot ``meta/info.json`` 的数据来源契约。"""

    return {
        "galaxea_recording_contract": GALAXEA_RECORDING_CONTRACT,
        "galaxea_controller_type": "bimanual_joint_position",
        "galaxea_image_preprocess": GALAXEA_IMAGE_PREPROCESS_CONTRACT,
        "galaxea_robot": "r1",
        "galaxea_action_dim": GALAXEA_STATE_DIM,
        "galaxea_control_freq": 15,
        "galaxea_camera_resolution_scale": 4,
    }


def validate_dataset_metadata(metadata: Mapping[str, object], expected: Mapping[str, object]) -> None:
    """拒绝由其它机器人、频率或图像预处理流程生成的数据集。"""

    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Galaxea dataset metadata does not match the contract: {mismatches}")


def validate_norm_stats_metadata(metadata: Mapping[str, object], expected: Mapping[str, object]) -> None:
    """拒绝与当前 Galaxea action/stats 流程不一致的统计量。"""

    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Galaxea normalization-stat metadata does not match the contract: {mismatches}")


@dataclasses.dataclass(frozen=True)
class GalaxeaInputs(transforms.DataTransformFn):
    """把在线/数据集中的 Galaxea 观测转换成 OpenPI canonical input。

    输入字段使用项目自己的短名称 ``images/state/prompt``；输出字段改成
    OpenPI 模型需要的 ``image/image_mask/state``。本类还负责将夹爪语义
    从仿真格式转成模型格式，但不会做 Normalize 或 tokenization，那些
    步骤由外层 policy transform pipeline 完成。
    """

    EXPECTED_IMAGES: ClassVar[tuple[str, ...]] = GALAXEA_IMAGE_NAMES

    def __call__(self, data: dict) -> dict:
        """完成图片命名、夹爪转换、状态校验和 prompt 透传。"""
        images = data.get("images")
        if not isinstance(images, dict):
            raise ValueError('Galaxea input must contain an "images" dictionary')
        missing = set(self.EXPECTED_IMAGES) - set(images)
        if missing:
            raise ValueError(f"Galaxea input is missing images: {sorted(missing)}")

        converted_images = {
            name: _convert_image(images[name], name=name) for name in self.EXPECTED_IMAGES
        }
        state = _convert_sim_grippers_to_model(
            _as_last_dim_array(data["state"], name="state", dim=GALAXEA_STATE_DIM)
        )

        result = {
            "image": {
                "base_0_rgb": converted_images["cam_high"],
                "left_wrist_0_rgb": converted_images["cam_left_wrist"],
                "right_wrist_0_rgb": converted_images["cam_right_wrist"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            result["actions"] = _convert_sim_grippers_to_model(
                _as_last_dim_array(data["actions"], name="actions", dim=GALAXEA_STATE_DIM)
            )
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class GalaxeaOutputs(transforms.DataTransformFn):
    """把 OpenPI canonical action 转回 R1 仿真动作。

    π0/π0.5 内部通常使用固定的 32 维 action buffer；R1 实际只需要前 14
    维。这里先裁剪 padding，再把模型夹爪语义恢复成 0~0.05 m，保证
    多余维度不会越过 WebSocket 进入仿真器。
    """

    def __call__(self, data: dict) -> dict:
        """裁剪模型 padding 并执行模型夹爪到仿真夹爪的反变换。"""
        # Pi0/Pi05 internally use action_dim=32. The R1 contract is the first
        # 14 dimensions; padding dimensions must never reach the simulator.
        actions = _as_at_least_dim_array(data["actions"], name="actions", dim=GALAXEA_STATE_DIM)
        actions = actions[..., :GALAXEA_STATE_DIM]
        return {"actions": _convert_model_grippers_to_sim(actions)}


@dataclasses.dataclass(frozen=True)
class GalaxeaLeRobotRepack(transforms.DataTransformFn):
    """拼接转换器写入的 LeRobot 组件字段。

    Pack the component fields written by the Galaxea LeRobot converter.

    This transform is used only on dataset samples.  Online inference already
    supplies the compact ``images/state`` contract consumed by
    :class:`GalaxeaInputs`.
    """

    image_fields: ClassVar[dict[str, str]] = {
        "cam_high": "observation.images.head_rgb",
        "cam_left_wrist": "observation.images.left_wrist_rgb",
        "cam_right_wrist": "observation.images.right_wrist_rgb",
    }
    state_fields: ClassVar[tuple[str, ...]] = (
        "observation.state.left_arm_joints",
        "observation.state.left_gripper",
        "observation.state.right_arm_joints",
        "observation.state.right_gripper",
    )
    action_fields: ClassVar[tuple[str, ...]] = (
        "action.left_arm_joints",
        "action.left_gripper",
        "action.right_arm_joints",
        "action.right_gripper",
    )
    component_dims: ClassVar[tuple[int, ...]] = (GALAXEA_ARM_DOF, 1, GALAXEA_ARM_DOF, 1)

    def _concat(self, data: dict, fields: tuple[str, ...], *, name: str) -> np.ndarray:
        """按 ``[左臂, 左夹爪, 右臂, 右夹爪]`` 拼接 state/action 字段。

        LeRobot 不同版本可能把单值夹爪保存成标量、``(T,)`` 或
        ``(T, 1)``，这里统一恢复最后一维，避免 batch/action horizon
        维度在拼接时发生歧义。
        """
        reference = np.asarray(data[fields[0]], dtype=np.float32)
        if reference.ndim == 0 or reference.shape[-1] != self.component_dims[0]:
            raise ValueError(
                f"Dataset field {fields[0]!r} must end in dimension {self.component_dims[0]}, got {reference.shape}"
            )
        reference_leading_shape = reference.shape[:-1]
        values = []
        for field, component_dim in zip(fields, self.component_dims, strict=True):
            if field not in data:
                raise ValueError(f"Dataset sample is missing {name} field {field!r}")
            value = np.asarray(data[field], dtype=np.float32)
            if component_dim == 1:
                # LeRobot versions differ here: a one-value feature can be a
                # scalar for state, or (T,) for an action sequence. Restore the
                # explicit component axis expected by the 14-D R1 contract.
                if value.ndim == 0:
                    if reference_leading_shape:
                        value = np.full((*reference_leading_shape, 1), value, dtype=np.float32)
                    else:
                        value = value.reshape(1)
                elif reference.ndim >= 2 and value.ndim == reference.ndim - 1:
                    value = np.expand_dims(value, axis=-1)
                elif value.shape[-1] != 1:
                    raise ValueError(
                        f"Dataset field {field!r} must contain one gripper value per sample, got {value.shape}"
                    )
            elif value.ndim == 0 or value.shape[-1] != component_dim:
                raise ValueError(
                    f"Dataset field {field!r} must end in dimension {component_dim}, got {value.shape}"
                )
            if value.shape[:-1] != reference_leading_shape:
                raise ValueError(
                    f"Dataset field {field!r} has leading shape {value.shape[:-1]}, "
                    f"expected {reference_leading_shape} to match {fields[0]!r}"
                )
            values.append(value)
        result = np.concatenate(values, axis=-1)
        if result.shape[-1] != GALAXEA_STATE_DIM:
            raise ValueError(f"Packed {name} must have last dimension 14, got {result.shape}")
        return result

    def __call__(self, data: dict) -> dict:
        """将 LeRobot 的分字段样本变成 OpenPI 适配器的紧凑字段。"""
        images = {}
        for name, field in self.image_fields.items():
            if field not in data:
                raise ValueError(f"Dataset sample is missing image field {field!r}")
            images[name] = data[field]

        result = {
            "images": images,
            "state": self._concat(data, self.state_fields, name="state"),
            "actions": self._concat(data, self.action_fields, name="action"),
        }
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


__all__ = [
    "GALAXEA_ACTION_HORIZON",
    "GALAXEA_ARM_DOF",
    "GALAXEA_EXECUTE_HORIZON",
    "GALAXEA_GRIPPER_MAX",
    "GALAXEA_IMAGE_NAMES",
    "GALAXEA_IMAGE_PREPROCESS_CONTRACT",
    "GALAXEA_MODEL_ACTION_DIM",
    "GALAXEA_RECORDING_CONTRACT",
    "GALAXEA_STATE_DIM",
    "GALAXEA_STATE_ORDER",
    "NORM_STATS_METADATA_FILENAME",
    "GalaxeaInputs",
    "GalaxeaLeRobotRepack",
    "GalaxeaOutputs",
    "make_dataset_metadata",
    "make_delta_action_mask",
    "make_norm_stats_metadata",
    "validate_dataset_metadata",
    "validate_norm_stats_metadata",
]
