import numpy as np
import pytest

from openpi import transforms
from openpi.policies import galaxea_r1_pro_policy


def _raw_state() -> np.ndarray:
    return np.asarray(
        [
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.05,
            -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 0.0,
        ],
        dtype=np.float32,
    )


def _observation() -> dict:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    return {
        "images": dict.fromkeys(galaxea_r1_pro_policy.R1_PRO_IMAGE_NAMES, image),
        "state": _raw_state(),
        "prompt": "pick up the two bottles simultaneously",
    }


def test_inputs_use_16d_r1_pro_contract() -> None:
    result = galaxea_r1_pro_policy.GalaxeaR1ProInputs()(_observation())
    assert result["state"].shape == (16,)
    np.testing.assert_allclose(result["state"][[7, 15]], [0.0, 1.0])
    assert result["image"]["base_0_rgb"].shape == (32, 48, 3)


def test_input_output_and_delta_roundtrip() -> None:
    canonical = galaxea_r1_pro_policy.GalaxeaR1ProInputs()(_observation())
    state = canonical["state"]
    # DeltaActions 会原地写入 actions，因此测试必须传独立副本，避免把
    # 后续 AbsoluteActions 使用的 state 也通过 NumPy view 一并改写。
    absolute = state[None, :].copy()
    delta = transforms.DeltaActions(galaxea_r1_pro_policy.make_delta_action_mask())(
        {"state": state, "actions": absolute}
    )
    restored = transforms.AbsoluteActions(galaxea_r1_pro_policy.make_delta_action_mask())(
        {"state": state, "actions": delta["actions"]}
    )
    output = galaxea_r1_pro_policy.GalaxeaR1ProOutputs()(restored)
    np.testing.assert_allclose(output["actions"], _raw_state()[None, :], atol=1e-6)


def test_outputs_remove_32d_model_padding() -> None:
    canonical = galaxea_r1_pro_policy.GalaxeaR1ProInputs()(_observation())
    padded = np.concatenate(
        [canonical["state"][None, :], np.full((1, 16), 99.0, dtype=np.float32)], axis=-1
    )
    result = galaxea_r1_pro_policy.GalaxeaR1ProOutputs()({"actions": padded})
    assert result["actions"].shape == (1, 16)
    np.testing.assert_allclose(result["actions"], _raw_state()[None, :], atol=1e-6)


def test_metadata_is_distinct_from_standard_r1() -> None:
    metadata = galaxea_r1_pro_policy.make_norm_stats_metadata(
        action_horizon=15,
        use_delta_joint_actions=True,
    )
    assert metadata["robot"] == "r1_pro"
    assert metadata["action_dim"] == 16
    assert len(galaxea_r1_pro_policy.R1_PRO_STATE_ORDER) == 16

    wrong = dict(metadata, robot="r1")
    with pytest.raises(ValueError, match="robot"):
        galaxea_r1_pro_policy.validate_norm_stats_metadata(wrong, metadata)
