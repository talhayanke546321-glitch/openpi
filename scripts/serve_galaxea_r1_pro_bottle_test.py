from pathlib import Path

import pytest

from scripts import serve_galaxea_r1_pro_bottle


def _make_checkpoint(root: Path) -> Path:
    (root / "params").mkdir(parents=True)
    (root / "_CHECKPOINT_METADATA").touch()
    assets = root / "assets" / serve_galaxea_r1_pro_bottle.ASSET_ID
    assets.mkdir(parents=True)
    (assets / "norm_stats.json").touch()
    (assets / "norm_stats_metadata.json").touch()
    return root


def test_validate_checkpoint_layout_accepts_complete_layout(tmp_path: Path) -> None:
    serve_galaxea_r1_pro_bottle._validate_checkpoint_layout(_make_checkpoint(tmp_path))


def test_validate_checkpoint_layout_rejects_missing_stats(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    (checkpoint / "assets" / serve_galaxea_r1_pro_bottle.ASSET_ID / "norm_stats.json").unlink()
    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        serve_galaxea_r1_pro_bottle._validate_checkpoint_layout(checkpoint)
