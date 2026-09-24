import numpy as np
import pytest

from openpi.serving import galaxea_r1_deployment
from openpi.training import config


class _FakePolicy:
    metadata = config.GALAXEA_POLICY_METADATA

    def infer(self, observation: dict) -> dict:
        assert observation["state"].shape == (14,)
        assert set(observation["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        return {"actions": np.zeros((15, 14), dtype=np.float32)}


def test_r1_metadata_and_warmup_observation() -> None:
    galaxea_r1_deployment.validate_r1_server_metadata(config.GALAXEA_POLICY_METADATA)
    observation = galaxea_r1_deployment.make_warmup_observation("test task")

    assert observation["state"].shape == (14,)
    assert observation["state"].dtype == np.float32
    for image in observation["images"].values():
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8


def test_warmup_policy_validates_action_chunk() -> None:
    elapsed = galaxea_r1_deployment.warmup_policy(_FakePolicy(), prompt="test task")
    assert elapsed >= 0


def test_invalid_metadata_fails_closed() -> None:
    metadata = dict(config.GALAXEA_POLICY_METADATA)
    metadata["state_order"] = list(reversed(metadata["state_order"]))
    with pytest.raises(ValueError, match="state_order"):
        galaxea_r1_deployment.validate_r1_server_metadata(metadata)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((10, 14), dtype=np.float32),
        np.zeros((15, 13), dtype=np.float32),
        np.full((15, 14), np.nan, dtype=np.float32),
    ],
)
def test_invalid_inference_result_fails_closed(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape|NaN/Inf"):
        galaxea_r1_deployment.validate_r1_inference_result({"actions": actions})
