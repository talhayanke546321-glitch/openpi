"""OpenPI 训练/推理配置定义。

``_CONFIGS`` 是命令行配置名到模型、数据集、transform、stats 和训练
超参数的映射。Galaxea 部分的配置不仅决定训练用哪个 checkpoint，还
定义了在线服务端必须复用的同一套输入/输出语义。
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.galaxea_policy as galaxea_policy
import openpi.policies.galaxea_r1_pro_policy as galaxea_r1_pro_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as _nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """已经解析完成、可以交给 data loader 的数据配置。

    对 Galaxea 来说，``repo_id``/``dataset_root`` 指向 LeRobot 数据，
    ``repack_transforms`` 负责把分字段样本拼回 R1 接口，
    ``data_transforms`` 负责夹爪和 delta action，``model_transforms``
    负责 π0.5 的图片/文本处理。最后三组 transform 会在推理时以相同
    顺序复用。
    """

    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

    # Galaxea 生成的本地 LeRobot 数据集根目录。None 时使用 LeRobot 默认
    # cache/Hub；仿真项目通常通过环境变量显式传入这个路径。
    # This is useful for datasets generated directly by GalaxeaManipSim.
    dataset_root: str | pathlib.Path | None = None
    # normalization stats 的来源契约。它必须单独保存，因为 stats 数值本身
    # 不记录 action horizon 和 transform 顺序。
    # Expected provenance contract for the normalization stats. This is kept
    # separate from ``norm_stats`` because the JSON stats file itself does not
    # encode the action horizon or transform order.
    norm_stats_metadata: dict[str, Any] | None = None
    # Whether the expected stats provenance sidecar was found in the configured
    # assets directory. Training checks this before applying normalization.
    norm_stats_metadata_present: bool = False
    # If true, training/inference must use stats generated with the current
    # Galaxea contract. Generic configs leave this false.
    require_norm_stats_metadata: bool = False
    # Expected provenance fields in the LeRobot ``meta/info.json`` file.
    dataset_metadata: dict[str, Any] | None = None
    # If true, the data loader must find and validate the dataset contract.
    require_dataset_metadata: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """把简洁的训练配置工厂展开成完整 ``DataConfig``。"""

    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """解析 repo/asset/stats 路径，并创建通用基础数据配置。"""
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        """加载标准 normalization stats；找不到时返回 None 供上层报错。"""
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

    def _load_norm_stats_metadata(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, Any] | None:
        """读取 ``norm_stats.json`` 旁边描述来源契约的 JSON sidecar。"""

        if asset_id is None:
            return None
        metadata_url = str(assets_dir / asset_id / galaxea_policy.NORM_STATS_METADATA_FILENAME)
        try:
            metadata_path = _download.maybe_download(metadata_url)
            metadata = json.loads(metadata_path.read_text())
        except FileNotFoundError:
            logging.info(f"Norm stats metadata not found at {metadata_url}, skipping.")
            return None
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid normalization-stat metadata JSON at {metadata_url}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"Normalization-stat metadata at {metadata_url} must be a JSON object")
        return metadata


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotGalaxeaDataConfig(DataConfigFactory):
    """标准 Galaxea R1 的 LeRobot 数据配置。

    LeRobot data configuration for the standard Galaxea R1.

    The Galaxea converter stores arm and gripper values in separate feature
    fields. ``GalaxeaLeRobotRepack`` combines them into the 14-dimensional
    interface consumed by ``GalaxeaInputs``. The same robot-specific
    transforms are then used by training and online inference.
    """

    # A local root can be supplied for datasets produced by GalaxeaManipSim.
    # When omitted, LeRobot resolves ``repo_id`` using its normal cache/Hub
    # behavior.
    dataset_root: str | pathlib.Path | None = None
    # Convert absolute joint targets to deltas. Gripper dimensions stay absolute.
    use_delta_joint_actions: bool = True
    # Real Galaxea training must have a provenance sidecar. The shape-only
    # smoke config below deliberately disables this because it uses built-in
    # UR5e statistics rather than statistics computed from R1 data.
    require_stats_metadata: bool = True
    # The stats command sets this false so it can replace an obsolete sidecar
    # while recomputing statistics for a changed action horizon.
    validate_stats_metadata: bool = True
    default_prompt: str | None = None
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[galaxea_policy.GalaxeaLeRobotRepack()]
        )
    )
    action_sequence_keys: Sequence[str] = galaxea_policy.GalaxeaLeRobotRepack.action_fields

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """构造 R1 专用数据流，并强制检查 15 步/32 维模型契约。"""
        if model_config.action_horizon != galaxea_policy.GALAXEA_ACTION_HORIZON:
            raise ValueError(
                "Galaxea R1 requires model action_horizon="
                f"{galaxea_policy.GALAXEA_ACTION_HORIZON}, got {model_config.action_horizon}"
            )
        if model_config.action_dim != galaxea_policy.GALAXEA_MODEL_ACTION_DIM:
            raise ValueError(
                "Galaxea R1/π0.5 requires internal model action_dim="
                f"{galaxea_policy.GALAXEA_MODEL_ACTION_DIM}, got {model_config.action_dim}"
            )
        # 先把仿真/LeRobot 字段转换成 Galaxea 14 维语义，再按配置选择是否
        # 将 12 个手臂关节变成 delta action；夹爪始终保持绝对值。
        data_transforms = _transforms.Group(
            inputs=[galaxea_policy.GalaxeaInputs()],
            outputs=[galaxea_policy.GalaxeaOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = galaxea_policy.make_delta_action_mask()
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 这里的 model_transforms 会在 normalization 之后处理 224x224 图像、
        # prompt tokenization 和 32 维 padding，训练和在线推理共用。
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        base_config = self.create_base_config(assets_dirs, model_config)
        stats_metadata = None
        stats_metadata_present = False
        if self.require_stats_metadata:
            stats_metadata = galaxea_policy.make_norm_stats_metadata(
                action_horizon=model_config.action_horizon,
                use_delta_joint_actions=self.use_delta_joint_actions,
            )
            actual_metadata = self._load_norm_stats_metadata(
                epath.Path(self.assets.assets_dir or assets_dirs),
                base_config.asset_id,
            )
            stats_metadata_present = actual_metadata is not None
            if actual_metadata is not None and self.validate_stats_metadata:
                galaxea_policy.validate_norm_stats_metadata(actual_metadata, stats_metadata)

        # 把契约元数据放入 DataConfig，供 data loader、stats 计算和 checkpoint
        # 加载阶段反复校验，避免“形状兼容但物理语义错误”的数据混用。
        return dataclasses.replace(
            base_config,
            dataset_root=self.dataset_root if self.dataset_root is not None else base_config.dataset_root,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            norm_stats_metadata=stats_metadata,
            norm_stats_metadata_present=stats_metadata_present,
            require_norm_stats_metadata=self.require_stats_metadata,
            dataset_metadata=galaxea_policy.make_dataset_metadata(),
            require_dataset_metadata=self.require_stats_metadata,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotGalaxeaR1ProDataConfig(DataConfigFactory):
    """R1 Pro 7 轴双臂 LeRobot 数据和在线推理配置。

    GalaxeaManipSim 的通用转换器把 R1 Pro 保存成三个 ``rgb_*`` 图像、一个
    16 维 ``observation.state`` 和一个 16 维 ``action``。本配置先把这些
    字段重排成在线接口，再执行夹爪语义转换、关节 delta action、归一化和
    π0.5 的 32 维 padding。训练与在线推理因此共享同一条变换链。
    """

    dataset_root: str | pathlib.Path | None = None
    use_delta_joint_actions: bool = True
    require_stats_metadata: bool = True
    validate_stats_metadata: bool = True
    default_prompt: str | None = None
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.rgb_head",
                            "cam_left_wrist": "observation.images.rgb_left_hand",
                            "cam_right_wrist": "observation.images.rgb_right_hand",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建严格的 15 步、16 维物理接口和 32 维模型接口。"""
        if model_config.action_horizon != galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON:
            raise ValueError(
                "Galaxea R1 Pro requires model action_horizon="
                f"{galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON}, got {model_config.action_horizon}"
            )
        if model_config.action_dim != galaxea_r1_pro_policy.R1_PRO_MODEL_ACTION_DIM:
            raise ValueError(
                "Galaxea R1 Pro/π0.5 requires internal model action_dim="
                f"{galaxea_r1_pro_policy.R1_PRO_MODEL_ACTION_DIM}, got {model_config.action_dim}"
            )

        data_transforms = _transforms.Group(
            inputs=[galaxea_r1_pro_policy.GalaxeaR1ProInputs()],
            outputs=[galaxea_r1_pro_policy.GalaxeaR1ProOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_mask = galaxea_r1_pro_policy.make_delta_action_mask()
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_mask)],
                outputs=[_transforms.AbsoluteActions(delta_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        stats_metadata = None
        stats_metadata_present = False
        if self.require_stats_metadata:
            stats_metadata = galaxea_r1_pro_policy.make_norm_stats_metadata(
                action_horizon=model_config.action_horizon,
                use_delta_joint_actions=self.use_delta_joint_actions,
            )
            actual_metadata = self._load_norm_stats_metadata(
                epath.Path(self.assets.assets_dir or assets_dirs),
                base_config.asset_id,
            )
            stats_metadata_present = actual_metadata is not None
            if actual_metadata is not None and self.validate_stats_metadata:
                galaxea_r1_pro_policy.validate_norm_stats_metadata(actual_metadata, stats_metadata)

        return dataclasses.replace(
            base_config,
            dataset_root=self.dataset_root if self.dataset_root is not None else base_config.dataset_root,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            norm_stats_metadata=stats_metadata,
            norm_stats_metadata_present=stats_metadata_present,
            require_norm_stats_metadata=self.require_stats_metadata,
            dataset_metadata=galaxea_r1_pro_policy.make_dataset_metadata(),
            require_dataset_metadata=self.require_stats_metadata,
        )


GALAXEA_R1_PRO_POLICY_METADATA = {
    "protocol_version": "galaxea-r1-pro-openpi-v1",
    "robot": "r1_pro",
    "controller": "bimanual_joint_position",
    "action_dim": galaxea_r1_pro_policy.R1_PRO_STATE_DIM,
    "action_horizon": galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON,
    "execute_horizon": galaxea_r1_pro_policy.R1_PRO_EXECUTE_HORIZON,
    "control_frequency_hz": 15,
    "state_order": list(galaxea_r1_pro_policy.R1_PRO_STATE_ORDER),
    "action_order": list(galaxea_r1_pro_policy.R1_PRO_STATE_ORDER),
    "arm_position_unit": "radian",
    "gripper_canonical_closed": 0.0,
    "gripper_canonical_open": galaxea_r1_pro_policy.R1_PRO_GRIPPER_MAX,
    "image_keys": list(galaxea_r1_pro_policy.R1_PRO_IMAGE_NAMES),
    "image_layout": "HWC",
    "image_dtype": "uint8",
    "image_color_space": "RGB",
    "training_domain": "simulation",
    "real_robot_validated": False,
}


GALAXEA_POLICY_METADATA = {
    # 这些字段会随 WebSocket 握手发送给仿真/真机客户端；它们不是模型输入，
    # 而是客户端用来确认机器人、关节顺序、单位和动作块协议的边界契约。
    "protocol_version": "galaxea-r1-openpi-v1",
    "robot": "r1",
    "controller": "bimanual_joint_position",
    "action_dim": galaxea_policy.GALAXEA_STATE_DIM,
    "action_horizon": galaxea_policy.GALAXEA_ACTION_HORIZON,
    "execute_horizon": galaxea_policy.GALAXEA_EXECUTE_HORIZON,
    "control_frequency_hz": 15,
    "state_order": list(galaxea_policy.GALAXEA_STATE_ORDER),
    "action_order": list(galaxea_policy.GALAXEA_STATE_ORDER),
    "arm_position_unit": "radian",
    # 这是模型/仿真兼容的夹爪规范值，不是 R1 SDK 的 0~100 硬件行程。
    # 真机客户端必须在 ROS 边界完成比例映射。
    "gripper_canonical_closed": 0.0,
    "gripper_canonical_open": galaxea_policy.GALAXEA_GRIPPER_MAX,
    "hardware_gripper_mapping_required": True,
    "image_keys": list(galaxea_policy.GALAXEA_IMAGE_NAMES),
    "image_layout": "HWC",
    "image_dtype": "uint8",
    "image_color_space": "RGB",
    # 当前 checkpoint 由仿真数据训练，metadata 明确禁止把“协议兼容”误解为
    # “已通过真机效果/安全验证”。将来使用真机数据训练后应发布新协议元数据。
    "training_domain": "simulation",
    "real_robot_validated": False,
}


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # Optional completed-step numbers at which to save checkpoints exactly. When
    # non-empty, the training loop uses these values instead of save_interval and
    # names checkpoints by the number of completed optimizer updates.
    save_steps: tuple[int, ...] = ()
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.save_steps:
            if any(step <= 0 for step in self.save_steps):
                raise ValueError("save_steps must contain positive completed-step numbers.")
            if tuple(sorted(set(self.save_steps))) != self.save_steps:
                raise ValueError("save_steps must be strictly increasing with no duplicates.")
            if self.save_steps[-1] > self.num_train_steps:
                raise ValueError("save_steps cannot contain a step beyond num_train_steps.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # R1 Pro 单任务配置。它从 pi05_base 初始化，但必须先用 R1 Pro 16 维
    # 数据计算独立的 norm stats，随后完成微调；不能把 base 权重裸输出到
    # 仿真器。GALAXEA_R1PRO_LEROBOT_ROOT 应指向通用 Galaxea 转换器生成的
    # LeRobot 数据集根目录。
    TrainConfig(
        name="pi05_galaxea_r1_pro",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_r1_pro_policy.R1_PRO_MODEL_ACTION_DIM,
            action_horizon=galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON,
        ),
        data=LeRobotGalaxeaR1ProDataConfig(
            repo_id="R1ProDualBottlesPickEasy-v0",
            dataset_root=os.environ.get("GALAXEA_R1PRO_LEROBOT_ROOT"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "OPENPI_PI05_BASE_PARAMS",
                "gs://openpi-assets/checkpoints/pi05_base/params",
            )
        ),
        num_train_steps=30_000,
        policy_metadata=GALAXEA_R1_PRO_POLICY_METADATA,
    ),
    # 低显存 R1 Pro LoRA 配置，与当前 RTX 3080 工作流保持一致。归一化
    # 统计会写入独立 assets 目录，不能与14维标准 R1 共用。
    TrainConfig(
        name="pi05_galaxea_r1_pro_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_r1_pro_policy.R1_PRO_MODEL_ACTION_DIM,
            action_horizon=galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotGalaxeaR1ProDataConfig(
            repo_id="R1ProDualBottlesPickEasy-v0",
            assets=AssetsConfig(
                assets_dir=os.environ.get(
                    "OPENPI_R1PRO_ASSETS_DIR",
                    "/home/vipuser/robotics/openpi/assets/pi05_galaxea_r1_pro",
                )
            ),
            dataset_root=os.environ.get("GALAXEA_R1PRO_LEROBOT_ROOT"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "OPENPI_PI05_BASE_PARAMS",
                "/home/vipuser/robotics/packages/pi05_base/params",
            )
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_r1_pro_policy.R1_PRO_MODEL_ACTION_DIM,
            action_horizon=galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        num_train_steps=30_000,
        policy_metadata=GALAXEA_R1_PRO_POLICY_METADATA,
    ),
    # R1 Pro bottle pick-and-place: the original five table heights plus the
    # supplemental 0.90 m group, with 20 successful episodes per height.  This
    # sibling config keeps the existing dual-bottle config intact while pointing
    # at the new task and its independent norm stats.
    TrainConfig(
        name="pi05_galaxea_r1_pro_bottle_place_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_r1_pro_policy.R1_PRO_MODEL_ACTION_DIM,
            action_horizon=galaxea_r1_pro_policy.R1_PRO_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotGalaxeaR1ProDataConfig(
            repo_id="galaxea/R1ProBottlePickPlace-v0",
            assets=AssetsConfig(
                assets_dir=os.environ.get(
                    "OPENPI_R1PRO_BOTTLE_PLACE_ASSETS_DIR",
                    "/home/vipuser/robotics/openpi/assets/pi05_galaxea_r1_pro_bottle_place",
                )
            ),
            dataset_root=os.environ.get("GALAXEA_R1PRO_BOTTLE_PLACE_LEROBOT_ROOT"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "OPENPI_PI05_BASE_PARAMS",
                "/home/vipuser/robotics/packages/pi05_base/params",
            )
        ),
        # Keep the 10 GB single-GPU path feasible: the standard Pi0 helper
        # also leaves vision/action projection weights trainable when both
        # LLM blocks use LoRA.  This task is intentionally LoRA-only; every
        # base weight is frozen and only paths containing ``lora`` receive
        # gradients and AdamW state.
        freeze_filter=nnx.All(nnx.Not(_nnx_utils.PathRegex(".*lora.*"))),
        ema_decay=None,
        batch_size=32,
        num_train_steps=10_000,
        save_interval=1_000,
        keep_period=2_000,
        save_steps=(6_000, 8_000, 10_000),
        policy_metadata=GALAXEA_R1_PRO_POLICY_METADATA,
    ),
    # 单任务 R1 配置：适合只使用一个 Gym 任务数据集进行微调。
    TrainConfig(
        name="pi05_galaxea_r1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
        ),
        data=LeRobotGalaxeaDataConfig(
            repo_id="R1DualBottlesPickEasy-v0",
            dataset_root=os.environ.get("GALAXEA_LEROBOT_ROOT"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        policy_metadata=GALAXEA_POLICY_METADATA,
    ),
    # 多任务配置：每个 episode 的 language_instruction 写入 LeRobot task，
    # prompt_from_task=True 让训练时使用与在线仿真相同的自然语言条件。
    TrainConfig(
        name="pi05_galaxea_r1_multitask",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
        ),
        data=LeRobotGalaxeaDataConfig(
            repo_id="galaxea_r1_multi_asset_v1",
            dataset_root=os.environ.get(
                "GALAXEA_MULTITASK_LEROBOT_ROOT",
                "/home/vipuser/robotics/GalaxeaManipSim/datasets/galaxea_r1_multi_asset_v1/lerobot",
            ),
            # The merged converter stores each episode's natural-language
            # instruction in LeRobot's task field.
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
        policy_metadata=GALAXEA_POLICY_METADATA,
    ),
    # 重新构建的多资产 R1 数据集的 LoRA 配置（8 个任务 x 50 条 episode）。
    # LoRA variant for the rebuilt multi-asset R1 dataset (8 tasks x 50
    # episodes).  Its stats are generated from the same 15-step action
    # horizon and Galaxea recording contract as the training data.
    TrainConfig(
        name="pi05_galaxea_r1_multitask_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotGalaxeaDataConfig(
            repo_id="galaxea_r1_multi_asset_v1",
            assets=AssetsConfig(
                assets_dir=os.environ.get(
                    "OPENPI_MULTITASK_ASSETS_DIR",
                    "/home/vipuser/robotics/openpi/assets/pi05_galaxea_r1_multitask",
                )
            ),
            dataset_root=os.environ.get(
                "GALAXEA_MULTITASK_LEROBOT_ROOT",
                "/home/vipuser/robotics/GalaxeaManipSim/datasets/galaxea_r1_multi_asset_v1/lerobot",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "OPENPI_PI05_BASE_PARAMS",
                "/home/vipuser/robotics/openpi-data/openpi-assets/checkpoints/pi05_base/params",
            )
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        num_train_steps=10_000,
        policy_metadata=GALAXEA_POLICY_METADATA,
    ),
    # 单独的 upright bottles 数据配置。
    TrainConfig(
        name="pi05_galaxea_r1_upright_bottles_v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
        ),
        data=LeRobotGalaxeaDataConfig(
            repo_id="galaxea_r1_upright_bottles_v1",
            dataset_root=os.environ.get(
                "GALAXEA_UPRIGHT_BOTTLES_LEROBOT_ROOT",
                "/home/vipuser/robotics/GalaxeaManipSim/datasets/galaxea_r1_upright_bottles_v1/lerobot",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
        policy_metadata=GALAXEA_POLICY_METADATA,
    ),
    # 低显存 upright bottles LoRA 微调配置。它必须使用同一 action horizon
    # 计算出的 stats，否则训练与推理的归一化分布不一致。
    # Low-memory Pi05 LoRA fine-tuning for the upright bottle task.  Its
    # normalization stats must be computed with action_horizon=15 before a
    # new checkpoint is trained.
    TrainConfig(
        name="pi05_galaxea_r1_upright_bottles_v1_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotGalaxeaDataConfig(
            repo_id="galaxea_r1_upright_bottles_v1",
            assets=AssetsConfig(
                assets_dir=os.environ.get(
                    "OPENPI_UPRIGHT_BOTTLES_ASSETS_DIR",
                    "/home/vipuser/robotics/openpi/assets/pi05_galaxea_r1_upright_bottles_v1",
                )
            ),
            dataset_root=os.environ.get(
                "GALAXEA_UPRIGHT_BOTTLES_LEROBOT_ROOT",
                "/home/vipuser/robotics/GalaxeaManipSim/datasets/galaxea_r1_upright_bottles_v1/lerobot",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "OPENPI_PI05_BASE_PARAMS",
                "/home/vipuser/robotics/openpi-data/openpi-assets/checkpoints/pi05_base/params",
            )
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        num_train_steps=8_000,
        policy_metadata=GALAXEA_POLICY_METADATA,
    ),
    # 仅用于检查传输/形状的 smoke 配置。``ur5e_dual`` 的 stats 只是形状
    # 兼容，不代表真实 R1 数据分布，不能用于正式训练或性能结论。
    # Shape-compatible smoke test only.  ``ur5e_dual`` is a built-in 14-D
    # dual-arm stats set; it is not a substitute for R1 dataset statistics.
    TrainConfig(
        name="pi05_galaxea_r1_smoke",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=galaxea_policy.GALAXEA_MODEL_ACTION_DIM,
            action_horizon=galaxea_policy.GALAXEA_ACTION_HORIZON,
        ),
        data=LeRobotGalaxeaDataConfig(
            repo_id="R1DualBottlesPickEasy-v0",
            assets=AssetsConfig(asset_id="ur5e_dual"),
            default_prompt="pick up the two bottles simultaneously",
            require_stats_metadata=False,
            base_config=DataConfig(prompt_from_task=False),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        policy_metadata=GALAXEA_POLICY_METADATA,
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
