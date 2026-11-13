"""Statistical closed-loop checks: guard against regressions in overall accuracy."""
import numpy as np

from sim.run_sim import run_episode
from sim.world import make_params


def test_accuracy_with_sensor_errors_but_no_gusts():
	"""Noise, latency, drift, attitude bias, mount and calibration error - but calm air."""
	errs = []
	for seed in range(10):
		res = run_episode(seed, params = make_params("nominal", gust_std = 0.0))
		assert res.precision, res.transitions
		errs.append(res.touchdown_error)
	assert max(errs) < 0.04


def test_nominal_scenario_success_and_accuracy():
	"""Light gusts. Over 200 seeds: median ~7.5 cm, p95 ~17 cm, max ~24 cm (gust-limited
	with PX4-default velocity gains). Bounds are loose enough for a 12-seed sample."""
	errs = []
	for seed in range(12):
		res = run_episode(100 + seed, "nominal")
		assert res.precision, res.transitions
		errs.append(res.touchdown_error)
	assert np.median(errs) < 0.15
	assert max(errs) < 0.30


def test_strong_wind_never_ends_up_hovering_forever():
	"""In conditions too rough for precision, the vehicle must still land (fallback)."""
	for seed in range(4):
		res = run_episode(200 + seed, "windy")
		assert res.landed, res.transitions
