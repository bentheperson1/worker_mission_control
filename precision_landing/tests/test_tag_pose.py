import numpy as np

from precision_landing import CameraModel, estimate_tag_pose
from precision_landing.geometry import euler_to_rot, exp_so3
from precision_landing.tag_pose import homography_dlt, tag_corner_points

CAM = CameraModel.from_fov(640, 480, 62.2, dist = (0.06, -0.12, 0.0005, -0.0005, 0.02))
SIZE = 0.1524


def project_tag(R, t, cam = CAM, size = SIZE):
	return cam.project(tag_corner_points(size) @ R.T + t)


def test_homography_dlt_exact():
	rng = np.random.default_rng(0)
	H = np.array([[1.2, 0.1, 3.0], [-0.2, 0.9, -1.0], [0.01, 0.02, 1.0]])
	src = rng.uniform(-1, 1, (6, 2))
	dst_h = np.column_stack([src, np.ones(6)]) @ H.T
	dst = dst_h[:, :2] / dst_h[:, 2:]
	assert np.allclose(homography_dlt(src, dst), H, atol = 1e-9)


def test_noise_free_pose_is_exact():
	rng = np.random.default_rng(1)
	checked = 0
	while checked < 100:
		R = euler_to_rot(*rng.uniform(-0.5, 0.5, 3))
		t = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0.4, 4.0)])
		px = project_tag(R, t)
		if not np.all(CAM.contains(px)):
			continue  # the detector can't see tags outside the image
		checked += 1
		pose = estimate_tag_pose(px, CAM, SIZE)
		assert np.allclose(pose.translation, t, atol = 1e-7)
		assert np.allclose(pose.rotation, R, atol = 1e-6)
		assert pose.reprojection_rms_px < 1e-6


def test_corner_order_convention_does_not_move_the_center():
	"""Rotated or mirrored corner numbering must not change the tag position."""
	R = euler_to_rot(0.05, -0.03, 0.7)
	t = np.array([0.12, -0.08, 1.5])
	px = project_tag(R, t)
	for corners in (np.roll(px, 1, axis = 0), np.roll(px, 2, axis = 0), px[::-1]):
		pose = estimate_tag_pose(corners, CAM, SIZE)
		assert np.allclose(pose.translation, t, atol = 1e-6)


def test_level_prior_removes_depth_bias():
	rng = np.random.default_rng(2)
	free, level = [], []
	for _ in range(400):
		R = euler_to_rot(*rng.uniform(-0.1, 0.1, 2), rng.uniform(-3, 3))
		t = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 3.0])
		px = project_tag(R, t) + rng.normal(0.0, 1.0, (4, 2))
		normal = exp_so3(rng.normal(0.0, np.radians(1.0), 3)) @ R[:, 2]  # attitude known to ~1 deg
		free.append(estimate_tag_pose(px, CAM, SIZE).translation[2] - t[2])
		level.append(estimate_tag_pose(px, CAM, SIZE, level_normal_cam = normal).translation[2] - t[2])
	assert np.mean(free) < -0.03  # unconstrained fit is biased toward the camera
	assert abs(np.mean(level)) < 0.02
	assert np.std(level) < np.std(free)


def test_rejects_degenerate_input():
	assert estimate_tag_pose(np.zeros((4, 2)), CAM, SIZE) is None
	assert estimate_tag_pose(np.full((4, 2), np.nan), CAM, SIZE) is None
	assert estimate_tag_pose(np.zeros((3, 2)), CAM, SIZE) is None
