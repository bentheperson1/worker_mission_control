import math

import numpy as np
import pytest

from precision_landing import CameraModel
from precision_landing.geometry import (
	euler_to_rot,
	exp_so3,
	interp_clamped,
	log_so3,
	orthonormalize,
	quat_slerp,
	quat_to_rot,
	rot_to_euler,
	rot_to_quat,
	wrap_angle,
)

rng = np.random.default_rng(42)


def random_rotation():
	return exp_so3(rng.normal(0.0, 1.0, 3))


def test_quaternion_roundtrip():
	for _ in range(200):
		R = random_rotation()
		assert np.allclose(quat_to_rot(rot_to_quat(R)), R, atol = 1e-10)


def test_quaternion_matches_px4_convention():
	# 90 deg yaw: body x (nose) points east in NED
	q = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
	assert np.allclose(quat_to_rot(q) @ [1, 0, 0], [0, 1, 0], atol = 1e-12)


def test_euler_roundtrip():
	for _ in range(200):
		rpy = rng.uniform([-1.4, -1.4, -3.1], [1.4, 1.4, 3.1])
		assert np.allclose(rot_to_euler(euler_to_rot(*rpy)), rpy, atol = 1e-9)


def test_exp_log_roundtrip_including_near_pi():
	for _ in range(200):
		w = rng.normal(0.0, 1.0, 3)
		w *= rng.uniform(0.0, math.pi - 1e-3) / np.linalg.norm(w)
		assert np.allclose(log_so3(exp_so3(w)), w, atol = 1e-6)
	w = np.array([0.0, 0.0, math.pi])
	assert np.allclose(exp_so3(log_so3(exp_so3(w))), exp_so3(w), atol = 1e-6)


def test_slerp_endpoints_and_midpoint():
	q0 = rot_to_quat(np.eye(3))
	q1 = rot_to_quat(euler_to_rot(0.0, 0.0, 1.0))
	assert np.allclose(quat_slerp(q0, q1, 0.0), q0)
	assert np.allclose(np.abs(quat_slerp(q0, q1, 1.0)), np.abs(q1))
	assert np.allclose(quat_to_rot(quat_slerp(q0, q1, 0.5)), euler_to_rot(0.0, 0.0, 0.5), atol = 1e-12)


def test_orthonormalize_returns_proper_rotation():
	M = random_rotation() + rng.normal(0.0, 0.05, (3, 3))
	R = orthonormalize(M)
	assert np.allclose(R @ R.T, np.eye(3), atol = 1e-12)
	assert np.linalg.det(R) == pytest.approx(1.0)


def test_small_helpers():
	assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
	assert interp_clamped(5.0, 0.0, 1.0, 2.0, 4.0) == 4.0
	assert interp_clamped(0.5, 0.0, 1.0, 2.0, 4.0) == 3.0


def test_camera_project_undistort_roundtrip():
	cam = CameraModel.from_fov(640, 480, 62.2, dist = (0.08, -0.15, 0.001, -0.001, 0.03))
	pts = np.column_stack([rng.uniform(-0.5, 0.5, 100), rng.uniform(-0.4, 0.4, 100), rng.uniform(0.5, 5.0, 100)])
	px = cam.project(pts)
	norm = cam.undistort_points(px)
	assert np.allclose(norm, pts[:, :2] / pts[:, 2:3], atol = 1e-7)


def test_camera_from_camera_info():
	k = [500.0, 0, 320.0, 0, 510.0, 240.0, 0, 0, 1]
	p = [490.0, 0, 318.0, 0, 0, 495.0, 238.0, 0, 0, 0, 1, 0]
	raw = CameraModel.from_camera_info(k, [0.1, -0.2, 0.0, 0.0], 640, 480, p)
	assert (raw.fx, raw.fy, raw.cx, raw.cy) == (500.0, 510.0, 320.0, 240.0)
	assert raw.dist == (0.1, -0.2, 0.0, 0.0, 0.0)
	rect = CameraModel.from_camera_info(k, [0.1], 640, 480, p, rectified = True)
	assert (rect.fx, rect.fy, rect.cx, rect.cy) == (490.0, 495.0, 318.0, 238.0)
	assert not rect.has_distortion
