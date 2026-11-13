import math

import numpy as np
import pytest

from precision_landing import LanderConfig, VehicleState
from precision_landing.controller import LandingController, SlewRateLimiter, saturate
from precision_landing.estimator import TargetTracker, UpdateResult
from precision_landing.history import StateHistory
from precision_landing.geometry import rot_to_quat, rot_z
from precision_landing.search import square_spiral


def make_tracker(**kw):
	cfg = LanderConfig().tracker
	for k, v in kw.items():
		setattr(cfg, k, v)
	return TargetTracker(cfg)


def test_static_tracker_converges_and_shrinks_covariance():
	rng = np.random.default_rng(0)
	tr = make_tracker()
	truth = np.array([1.0, -2.0, 0.5])
	R = np.diag([0.02, 0.02, 0.05]) ** 2
	for k in range(60):
		tr.update(k / 15.0, truth + rng.multivariate_normal(np.zeros(3), R), R)
	assert np.linalg.norm(tr.position - truth) < 0.02
	assert np.all(tr.velocity == 0.0)
	assert np.sqrt(tr.P[0, 0]) < 0.01


def test_gate_rejects_outlier_then_reinitializes_on_persistent_jump():
	tr = make_tracker(max_consecutive_rejects = 4)
	R = np.eye(3) * 0.01 ** 2
	for k in range(20):
		tr.update(k * 0.1, np.zeros(3), R)
	assert tr.update(2.0, np.array([1.0, 0.0, 0.0]), R) == UpdateResult.REJECTED
	assert np.allclose(tr.position, 0.0, atol = 1e-3)
	results = [tr.update(2.1 + 0.1 * k, np.array([1.0, 0.0, 0.0]), R) for k in range(3)]
	assert results == [UpdateResult.REJECTED, UpdateResult.REJECTED, UpdateResult.REINIT]
	assert np.allclose(tr.position, [1.0, 0.0, 0.0])


def test_moving_tracker_follows_constant_velocity():
	tr = make_tracker(target_model = "moving", velocity_damping_tau = 0.0, process_accel_noise = 0.5)
	R = np.eye(3) * 0.01 ** 2
	v = np.array([0.5, -0.2, 0.0])
	for k in range(100):
		t = k * 0.05
		tr.update(t, v * t, R)
	assert np.allclose(tr.velocity, v, atol = 0.05)
	pos, _, _ = tr.predicted(5.0)
	assert np.allclose(pos, v * 5.0, atol = 0.05)


def test_prediction_does_not_mutate_and_is_capped():
	tr = make_tracker(target_model = "moving", velocity_damping_tau = 0.0, max_extrapolation_s = 1.0)
	tr.update(0.0, np.zeros(3), np.eye(3) * 1e-4)
	tr.x[3:] = [1.0, 0.0, 0.0]
	before = tr.x.copy()
	pos, _, _ = tr.predicted(10.0)
	assert np.allclose(tr.x, before)
	assert pos[0] == pytest.approx(1.0)  # capped at 1 s of extrapolation


def test_state_history_interpolates():
	h = StateHistory()
	q0 = rot_to_quat(rot_z(0.0))
	q1 = rot_to_quat(rot_z(0.2))
	h.add(VehicleState(0.0, [0, 0, -1], [1, 0, 0], q0))
	h.add(VehicleState(0.1, [0.1, 0, -1], [1, 0, 0], q1))
	s = h.at(0.05)
	assert np.allclose(s.position, [0.05, 0, -1])
	assert np.allclose(s.attitude, rot_to_quat(rot_z(0.1)), atol = 1e-9)
	assert h.at(-1.0) is None
	assert h.at(0.15) is h.latest()  # small forward gap -> latest
	assert h.at(1.0) is None


def test_controller_saturation_cone_and_schedule():
	c = LandingController(LanderConfig().control)
	v = c.horizontal(0.05, np.array([10.0, 0.0]), np.zeros(2), np.zeros(2), height = 5.0)
	assert np.linalg.norm(v) == pytest.approx(c.cfg.max_speed_xy_high)
	assert c.max_horizontal_speed(0.1) == c.cfg.max_speed_xy_low
	h = 2.0
	r = c.allowed_radius(h)
	assert c.descent_rate(h, 0.0) == pytest.approx(c.descent_speed(h))
	assert 0.0 < c.descent_rate(h, 0.75 * r) < c.descent_speed(h)
	assert c.descent_rate(h, 1.01 * r) == 0.0


def test_integrator_anti_windup():
	c = LandingController(LanderConfig().control)
	for _ in range(2000):
		c.horizontal(0.05, np.array([0.4, 0.0]), np.zeros(2), np.zeros(2), height = 0.3)
	assert np.linalg.norm(c.cfg.ki_xy * c.integral) <= c.cfg.integral_limit + 1e-9


def test_slew_rate_limiter():
	s = SlewRateLimiter(2.0)
	s.reset(np.zeros(3))
	out = s.step(np.array([1.0, 0.0, 0.0]), 0.05)
	assert np.linalg.norm(out) == pytest.approx(0.1)
	assert np.allclose(s.step(np.array([1.0, 0.0, 0.0]), 0.0), out)  # dt = 0 holds
	assert np.allclose(saturate(np.array([3.0, 4.0]), 1.0), [0.6, 0.8])


def test_square_spiral_covers_the_whole_square():
	pts = np.array(square_spiral([0.0, 0.0], 1.0, 2.5))
	assert np.max(np.abs(pts)) <= 2.5 + 1e-9
	# all four quadrants of the outer ring are visited
	for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
		assert np.any((np.sign(pts[:, 0]) == sx) & (np.sign(pts[:, 1]) == sy) & (np.max(np.abs(pts), axis = 1) >= 2.0))
