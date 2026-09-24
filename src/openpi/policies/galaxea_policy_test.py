import numpy as np
import pytest

from openpi import transforms
from openpi.policies import galaxea_policy


def _raw_state() -> np.ndarray:
    return np.asarray(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.05, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, 0.0],
        dtype=np.float32,
    )


def _raw_observation() -> dict:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    return {
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "state": _raw_state(),
        "prompt": "pick up the two bottles simultaneously",
    }


def test_galaxea_inputs_convert_images_and_grippers() -> None:
    result = galaxea_policy.GalaxeaInputs()(_raw_observation())

    assert result["image"]["base_0_rgb"].shape == (32, 48, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_allclose(result["state"][[6, 13]], [0.0, 1.0])
    assert result["state"].shape == (14,)


def test_galaxea_inputs_accept_chw_images() -> None:
    data = _raw_observation()
    chw = np.zeros((3, 32, 48), dtype=np.uint8)
    data["images"] = dict.fromkeys(galaxea_policy.GALAXEA_IMAGE_NAMES, chw)
    result = galaxea_policy.GalaxeaInputs()(data)
    assert result["image"]["base_0_rgb"].shape == (32, 48, 3)


def test_galaxea_inputs_reject_nonfinite_values() -> None:
    data = _raw_observation()
    data["state"] = _raw_state().copy()
    data["state"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        galaxea_policy.GalaxeaInputs()(data)


def test_gripper_action_roundtrip_without_stats() -> None:
    input_result = galaxea_policy.GalaxeaInputs()(_raw_observation())
    state = input_result["state"]
    absolute_model_action = state[None, ...].copy()
    delta = transforms.DeltaActions(galaxea_policy.make_delta_action_mask())(
        {"state": state.copy(), "actions": absolute_model_action.copy()}
    )
    restored = transforms.AbsoluteActions(galaxea_policy.make_delta_action_mask())(
        {"state": state.copy(), "actions": delta["actions"]}
    )
    output = galaxea_policy.GalaxeaOutputs()(restored)
    np.testing.assert_allclose(output["actions"], _raw_state()[None, ...], atol=1e-6)


def test_galaxea_outputs_removes_pi_padding() -> None:
    raw_state = _raw_state()
    model_action = galaxea_policy.GalaxeaInputs()(_raw_observation())["state"][None, :]
    padded = np.concatenate([model_action, np.full((1, 18), 99.0, dtype=np.float32)], axis=-1)
    result = galaxea_policy.GalaxeaOutputs()({"actions": padded})
    assert result["actions"].shape == (1, 14)
    np.testing.assert_allclose(result["actions"], raw_state[None, :], atol=1e-6)


def test_lerobot_repack_concatenates_component_fields() -> None:
    data = {
        "observation.images.head_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.left_wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.right_wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.state.left_arm_joints": np.arange(6, dtype=np.float32),
        "observation.state.left_gripper": np.asarray([0.05], dtype=np.float32),
        "observation.state.right_arm_joints": np.arange(6, 12, dtype=np.float32),
        "observation.state.right_gripper": np.asarray([0.0], dtype=np.float32),
        "action.left_arm_joints": np.arange(6, dtype=np.float32),
        "action.left_gripper": np.asarray([0.05], dtype=np.float32),
        "action.right_arm_joints": np.arange(6, 12, dtype=np.float32),
        "action.right_gripper": np.asarray([0.0], dtype=np.float32),
        "prompt": "test",
    }

    result = galaxea_policy.GalaxeaLeRobotRepack()(data)
    assert result["state"].shape == (14,)
    assert result["actions"].shape == (14,)
    assert result["prompt"] == "test"
    np.testing.assert_allclose(result["state"][[6, 13]], [0.05, 0.0])


def test_lerobot_repack_restores_scalar_and_sequence_gripper_axes() -> None:
    data = {
        "observation.images.head_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.left_wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.right_wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.state.left_arm_joints": np.arange(6, dtype=np.float32),
        "observation.state.left_gripper": np.float32(0.05),
        "observation.state.right_arm_joints": np.arange(6, 12, dtype=np.float32),
        "observation.state.right_gripper": np.float32(0.0),
        "action.left_arm_joints": np.zeros((3, 6), dtype=np.float32),
        "action.left_gripper": np.asarray([0.05, 0.04, 0.03], dtype=np.float32),
        "action.right_arm_joints": np.zeros((3, 6), dtype=np.float32),
        "action.right_gripper": np.asarray([0.0, 0.01, 0.02], dtype=np.float32),
    }

    result = galaxea_policy.GalaxeaLeRobotRepack()(data)
    assert result["state"].shape == (14,)
    assert result["actions"].shape == (3, 14)
    np.testing.assert_allclose(result["actions"][:, 6], [0.05, 0.04, 0.03])
    np.testing.assert_allclose(result["actions"][:, 13], [0.0, 0.01, 0.02])


def test_lerobot_repack_handles_one_step_action_sequences() -> None:
    data = {
        "observation.images.head_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.left_wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.right_wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.state.left_arm_joints": np.zeros(6, dtype=np.float32),
        "observation.state.left_gripper": np.float32(0.0),
        "observation.state.right_arm_joints": np.zeros(6, dtype=np.float32),
        "observation.state.right_gripper": np.float32(0.0),
        "action.left_arm_joints": np.zeros((1, 6), dtype=np.float32),
        "action.left_gripper": np.asarray([0.05], dtype=np.float32),
        "action.right_arm_joints": np.zeros((1, 6), dtype=np.float32),
        "action.right_gripper": np.asarray([0.0], dtype=np.float32),
    }

    result = galaxea_policy.GalaxeaLeRobotRepack()(data)
    assert result["actions"].shape == (1, 14)


def test_norm_stats_metadata_matches_horizon_and_delta_contract() -> None:
    metadata = galaxea_policy.make_norm_stats_metadata(
        action_horizon=15,
        use_delta_joint_actions=True,
    )
    galaxea_policy.validate_norm_stats_metadata(
        metadata,
        galaxea_policy.make_norm_stats_metadata(
            action_horizon=15,
            use_delta_joint_actions=True,
        ),
    )


def test_norm_stats_metadata_rejects_wrong_horizon() -> None:
    metadata = galaxea_policy.make_norm_stats_metadata(
        action_horizon=50,
        use_delta_joint_actions=True,
    )
    with pytest.raises(ValueError, match="action_horizon"):
        galaxea_policy.validate_norm_stats_metadata(
            metadata,
            galaxea_policy.make_norm_stats_metadata(
                action_horizon=15,
                use_delta_joint_actions=True,
            ),
        )


def test_dataset_metadata_contract_is_explicit() -> None:
    metadata = galaxea_policy.make_dataset_metadata()
    galaxea_policy.validate_dataset_metadata(metadata, galaxea_policy.make_dataset_metadata())


def test_dataset_metadata_rejects_missing_recording_contract() -> None:
    metadata = galaxea_policy.make_dataset_metadata()
    metadata.pop("galaxea_recording_contract")
    with pytest.raises(ValueError, match="recording_contract"):
        galaxea_policy.validate_dataset_metadata(metadata, galaxea_policy.make_dataset_metadata())
