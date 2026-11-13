"""State machine behavior, exercised in closed loop against the simulator."""
import numpy as np
import pytest

from precision_landing import CommandMode, LanderConfig, PrecisionLander
from sim.run_sim import run_episode
from sim.world import make_params

# every error source off: behavior is deterministic and should be near-perfect
IDEAL = dict(
	wind_accel = 0.0, gust_std = 0.0, drift_std = 0.0, pos_noise_std = 0.0, vel_noise_std = 0.0,
	att_noise_std_deg = 0.0, att_bias_std_deg = 0.0, mount_error_std_deg = 0.0,
	calib_focal_error_std = 0.0, pixel_noise_px = 0.0, dropout_prob = 0.0,
	camera_latency_jitter_s = 0.0,
)


def ideal(**kw):
	return make_params("nominal", **{**IDEAL, **kw})


def states(res):
	return [b for _, _, b, _ in res.transitions]


def config(**sections):
	cfg = LanderConfig()
	for section, values in sections.items():
		for k, v in values.items():
			setattr(getattr(cfg, section), k, v)
	return cfg


def test_start_requires_vehicle_state():
	ok, msg = PrecisionLander().start(0.0)
	assert not ok and "vehicle state" in msg


def test_ideal_landing_follows_nominal_sequence():
	res = run_episode(0, params = ideal())
	assert states(res) == ["search", "acquire", "align", "descend", "touchdown", "landed"]
	assert res.touchdown_error < 0.03


def test_tag_never_found_falls_back_to_land_in_place():
	cfg = config(protocol = {"search_timeout_s": 5.0})
	res = run_episode(0, config = cfg, params = ideal(start_offset = (20.0, 20.0)))
	assert "fallback_land" in states(res)
	assert "tag not found" in res.transitions[states(res).index("fallback_land")][3]
	assert res.landed and not res.precision


def test_spiral_search_finds_offset_tag():
	res = run_episode(1, params = ideal(start_offset = (2.2, 2.6), start_height = (3.5, 3.5)))
	assert res.precision and res.touchdown_error < 0.03


def test_occlusion_during_descent_triggers_reacquire_then_lands():
	fired = []

	def hook(t, world, lander):
		if lander.state.value == "descend" and world.pos[2] > -1.5 and not fired:
			world.occluded_until = t + 3.5
			fired.append(t)

	res = run_episode(2, params = ideal(), hook = hook)
	assert fired
	assert "reacquire" in states(res)
	assert res.precision and res.touchdown_error < 0.03


def test_commit_when_tag_lost_low_and_centered():
	def hook(t, world, lander):
		if lander.state.value == "descend" and world.pos[2] > -0.55:
			world.occluded_until = 1e9  # tag gone for good (e.g. fills the image)

	res = run_episode(3, params = ideal(), hook = hook)
	i = states(res).index("touchdown")
	assert "committing" in res.transitions[i][3]
	assert res.precision and res.touchdown_error < 0.05


def test_stale_vehicle_state_aborts_to_hold():
	def hook(t, world, lander):
		if t > 3.0:
			world.state_feed_alive = False
		return t > 5.0

	res = run_episode(0, params = ideal(), hook = hook)
	assert res.final_state == "aborted"
	assert "stale" in res.transitions[-1][3]
	assert res.lander.last_command.mode == CommandMode.HOLD


def test_abort_then_restart():
	log = {}

	def hook(t, world, lander):
		if t >= 2.0 and "aborted" not in log:
			assert lander.abort(t, "test abort")
			log["aborted"] = lander.update(t).mode
			ok, _ = lander.start(t)
			log["restarted"] = ok
		return t > 3.0

	res = run_episode(0, params = ideal(), hook = hook)
	assert log == {"aborted": CommandMode.HOLD, "restarted": True}
	assert res.final_state in ("search", "acquire", "align")


def test_landed_report_ignored_while_high():
	results = []

	def hook(t, world, lander):
		if 1.0 <= t < 1.05:
			results.append(lander.notify_landed(t))
		return t > 1.5

	res = run_episode(0, params = ideal(), hook = hook)
	assert results == [False]
	assert res.lander.stats["ignored_landed"] == 1
	assert res.final_state != "landed"


def test_guided_final_descent_streams_velocity_until_contact():
	modes = []

	def hook(t, world, lander):
		if lander.state.value == "touchdown" and lander.last_command is not None:
			modes.append(lander.last_command.mode)

	cfg = config(protocol = {"final_descent": "guided"})
	res = run_episode(0, config = cfg, params = ideal(), hook = hook)
	assert modes and all(m == CommandMode.VELOCITY for m in modes[1:])
	assert res.precision and res.touchdown_error < 0.03


def test_stalled_descent_falls_back():
	"""Alignment radius impossible to meet -> ALIGN never finishes -> watchdog lands."""
	cfg = config(protocol = {"align_radius": 0.06, "handoff_radius": 0.05, "stall_timeout_s": 8.0})
	res = run_episode(0, config = cfg, params = ideal(gust_std = 1.0))
	assert "fallback_land" in states(res)
	assert "no descent progress" in res.transitions[states(res).index("fallback_land")][3]
	assert res.landed


def test_other_tag_ids_are_ignored():
	cfg = config(protocol = {"search_timeout_s": 4.0})
	res = run_episode(0, config = cfg, params = ideal(tag_id = 7))
	assert res.lander.stats["used"] == 0
	assert "fallback_land" in states(res)


def test_false_detections_are_rejected():
	res = run_episode(4, params = ideal(false_detection_prob = 0.3))
	s = res.lander.stats
	assert s["rejected_gate"] + s["rejected_pose"] > 0
	assert res.precision and res.touchdown_error < 0.05


def test_status_is_json_serializable():
	import json

	def hook(t, world, lander):
		json.dumps(lander.status())
		return t > 6.0

	run_episode(0, params = ideal(), hook = hook)
