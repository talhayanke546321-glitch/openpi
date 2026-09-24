"""从训练配置和 checkpoint 构造可在线调用的 OpenPI Policy。

这里是训练配置、模型权重、normalization stats、输入/输出 transform 和
WebSocket server 之间的装配点。Galaxea 的特殊逻辑主要在
``galaxea_policy.py``，本文件负责把这些逻辑按正确顺序接到通用 OpenPI
推理流程中。
"""

import json
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.galaxea_policy as _galaxea_policy
import openpi.policies.galaxea_r1_pro_policy as _galaxea_r1_pro_policy
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _validate_checkpoint_norm_stats_metadata(
    checkpoint_dir: pathlib.Path,
    data_config: _config.DataConfig,
) -> None:
    """确认 Galaxea checkpoint 携带与当前数据契约匹配的 stats metadata。

    普通的 norm_stats.json 只有数值，不包含 action horizon、夹爪转换和
    delta-action 信息。如果缺少旁车 metadata，旧 checkpoint 即使形状
    看起来兼容，也可能在物理意义上不兼容，因此这里选择直接拒绝加载。
    """

    if not data_config.require_norm_stats_metadata:
        return
    if data_config.asset_id is None or data_config.norm_stats_metadata is None:
        raise ValueError("Galaxea policy requires an asset id and normalization-stat contract metadata")

    metadata_path = (
        checkpoint_dir
        / "assets"
        / data_config.asset_id
        / _galaxea_policy.NORM_STATS_METADATA_FILENAME
    )
    if not metadata_path.exists():
        raise ValueError(
            f"Checkpoint {checkpoint_dir} has no {metadata_path.name}. "
            "It predates the Galaxea action/stats contract; recompute stats and retrain the checkpoint."
        )
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid checkpoint normalization-stat metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint normalization-stat metadata must be a JSON object: {metadata_path}")
    # R1 与 R1 Pro 的 sidecar 文件名相同，但机器人、动作维度和关节顺序
    # 不同。根据期望契约选择对应校验器，避免把 14 维 R1 stats 错接到
    # 16 维 R1 Pro 策略。
    if data_config.norm_stats_metadata.get("robot") == "r1_pro":
        _galaxea_r1_pro_policy.validate_norm_stats_metadata(metadata, data_config.norm_stats_metadata)
    else:
        _galaxea_policy.validate_norm_stats_metadata(metadata, data_config.norm_stats_metadata)


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """从 checkpoint 构造一个可以被 WebSocket 服务调用的策略。

    Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    # 先创建 DataConfig 并校验契约，再加载模型。这样 checkpoint 不合格时
    # 会在昂贵的模型恢复之前失败，错误信息也更接近根因。
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    _validate_checkpoint_norm_stats_metadata(checkpoint_dir, data_config)

    # 根据 model.safetensors 判断是 PyTorch checkpoint 还是 JAX checkpoint。
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
        if norm_stats is None and data_config.require_norm_stats_metadata:
            raise ValueError(
                f"Checkpoint {checkpoint_dir} has no normalization statistics for "
                f"asset {data_config.asset_id!r}."
            )

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # 输入 transform 顺序：数据集字段重排 -> 默认 prompt -> Galaxea 语义
    # 转换 -> normalization -> 模型专用图片/文本/token 处理。
    # 输出顺序反过来：模型输出 -> 反归一化 -> Galaxea 反变换 -> 重排回
    # 仿真/客户端能理解的动作。
    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
