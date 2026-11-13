import numpy as np
import pytest

from precision_landing import LanderConfig
from precision_landing.config import CameraMount


def test_from_dict_roundtrip():
	cfg = LanderConfig.from_dict({
		"tags": [{"tag_id": 3, "size": 0.2}, {"tag_id": 4, "size": 0.05, "offset": [0.1, 0.0, 0.0]}],
		"control": {"kp_xy": 1.0},
		"protocol": {"final_descent": "guided"},
	})
	assert [t.tag_id for t in cfg.tags] == [3, 4]
	assert cfg.control.kp_xy == 1.0
	assert cfg.control.ki_xy == LanderConfig().control.ki_xy  # untouched default
	again = LanderConfig.from_dict(cfg.to_dict())
	assert again.to_dict() == cfg.to_dict()


@pytest.mark.parametrize("bad", [
	{"contrl": {}},
	{"control": {"kpxy": 1.0}},
	{"tags": [{"tag_id": 0, "size": 0.1, "colour": "red"}]},
])
def test_typos_are_rejected(bad):
	with pytest.raises(ValueError, match = "unknown"):
		LanderConfig.from_dict(bad)


@pytest.mark.parametrize("bad", [
	{"tags": []},
	{"tags": [{"tag_id": 0, "size": 0.1}, {"tag_id": 0, "size": 0.2}]},
	{"tags": [{"tag_id": 0, "size": -1.0}]},
	{"protocol": {"final_descent": "yolo"}},
	{"protocol": {"handoff_height": 1.0, "commit_height": 0.5}},
	{"tracker": {"target_model": "teleporting"}},
	{"camera_mount": {"rotation_body_cam": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}},
])
def test_invalid_values_are_rejected(bad):
	with pytest.raises(ValueError, match = "invalid"):
		LanderConfig.from_dict(bad)


def test_camera_mount_heading():
	# default: image top toward the nose -> optical axis down, image right = body right
	R = CameraMount().rotation()
	assert np.allclose(R @ [0, 0, 1], [0, 0, 1])
	assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
	assert np.allclose(R @ [0, -1, 0], [1, 0, 0])  # image up -> nose
	# image top toward the right side of the vehicle
	R90 = CameraMount(image_up_heading_deg = 90.0).rotation()
	assert np.allclose(R90 @ [0, -1, 0], [0, 1, 0], atol = 1e-12)
