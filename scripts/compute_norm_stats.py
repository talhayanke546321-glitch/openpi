"""为指定训练配置计算 normalization statistics。

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.policies.galaxea_policy as galaxea_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """创建用于统计 state/actions 分布的 PyTorch 数据加载器。"""
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """创建 RLDS 统计数据加载器。"""
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    """按训练时相同的字段/动作 transform 计算并保存统计量。"""
    config = _config.get_config(config_name)
    # Recomputing stats is the migration path when an existing sidecar was
    # generated for an obsolete action horizon. Do not let that old metadata
    # prevent this command from producing the replacement files.
    data_factory = config.data
    if isinstance(
        data_factory,
        _config.LeRobotGalaxeaDataConfig | _config.LeRobotGalaxeaR1ProDataConfig,
    ):
        data_factory = dataclasses.replace(data_factory, validate_stats_metadata=False)
    data_config = data_factory.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.asset_id is None:
        raise ValueError("The selected config has no asset_id; cannot store normalization stats")

    # ``AssetsConfig.assets_dir`` is the directory that training and
    # inference actually read.  The old code always wrote to
    # ``config.assets_dirs``, which was wrong for configs that deliberately
    # share stats through a custom assets directory (for example the R1 LoRA
    # config).  Use the same root for both reading and writing, and use the
    # asset id rather than repo id (these differ for smoke-test configs).
    configured_assets_dir = getattr(getattr(config.data, "assets", None), "assets_dir", None)
    if configured_assets_dir is not None and "://" not in str(configured_assets_dir):
        assets_root = Path(configured_assets_dir).expanduser()
    elif isinstance(
        data_factory,
        _config.LeRobotGalaxeaDataConfig | _config.LeRobotGalaxeaR1ProDataConfig,
    ) and configured_assets_dir is not None:
        raise ValueError("compute_norm_stats.py writes local files; use a local assets_dir")
    else:
        # Preserve the generic OpenPI behavior for remote/base assets: the
        # stats command writes a local result under the configured assets root.
        assets_root = config.assets_dirs
    output_path = assets_root / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    if data_config.norm_stats_metadata is not None:
        metadata_path = output_path / galaxea_policy.NORM_STATS_METADATA_FILENAME
        metadata_path.write_text(json.dumps(data_config.norm_stats_metadata, indent=2, sort_keys=True) + "\n")
        print(f"Writing stats metadata to: {metadata_path}")


if __name__ == "__main__":
    tyro.cli(main)
