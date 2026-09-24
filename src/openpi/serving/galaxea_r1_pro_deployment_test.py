import numpy as np
import pytest

from openpi.serving import galaxea_r1_pro_deployment
from openpi.training import config


def test_warmup_observation_and_metadata_use_r1_pro_contract() -> None:
    observation = galaxea_r1_pro_deployment.make_warmup_observation("stack the blocks")
    assert observation["state"].shape == (16,)
    assert all(image.shape == (224, 224, 3) for image in observation["images"].values())
    galaxea_r1_pro_deployment.validate_server_metadata(config.GALAXEA_R1_PRO_POLICY_METADATA)


def test_inference_result_requires_15_by_16() -> None:
    actions = galaxea_r1_pro_deployment.validate_inference_result(
        {"actions": np.zeros((15, 16), dtype=np.float32)}
    )
    assert actions.shape == (15, 16)
    with pytest.raises(ValueError, match="expected"):
        galaxea_r1_pro_deployment.validate_inference_result(
            {"actions": np.zeros((15, 14), dtype=np.float32)}
        )
