"""Run the precision landing protocol in closed loop against the simulated world.

	python -m sim.run_sim                          # one nominal episode + plot
	python -m sim.run_sim --scenario windy --seed 4
	python -m sim.run_sim --monte-carlo 200        # accuracy statistics
	python -m sim.run_sim --monte-carlo 100 --scenario all
	python -m sim.run_sim --config config/precision_landing.yaml
	python -m sim.run_sim --monte-carlo 100 --set protocol.final_descent=guided --set control.kp_xy=1.2
"""
import argparse
from dataclasses import dataclass, field
import math
import os
import sys

import numpy as np

from precision_landing import LanderConfig, LandingState, PrecisionLander

from .world import SCENARIOS, SimWorld, make_params


@dataclass
class EpisodeResult:
	seed: int
	scenario: str
	landed: bool
	final_state: str
	touchdown_error: float
	duration: float
	transitions: list
	log: dict = field(default_factory = dict)

	@property
	def precision(self):
		"""True if the vehicle came down via the precision path (not a fallback)."""
		return self.landed and any(t[2] == LandingState.TOUCHDOWN.value for t in self.transitions)


def run_episode(seed = 0, scenario = "nominal", config = None, params = None, max_time = 150.0, record = False, hook = None, start = True):
	"""Fly one landing. `hook(t, world, lander)` runs every control tick (tests use it to
	inject events); returning True ends the episode early."""
	rng = np.random.default_rng(seed)
	params = params if params is not None else make_params(scenario)
	config = config if config is not None else LanderConfig()
	world = SimWorld(params, rng, mount = config.camera_mount)
	lander = PrecisionLander(config, world.lander_camera)
	dt = params.physics_dt

	def period(hz):
		return max(1, int(round(1.0 / (hz * dt))))

	ctrl_every, cam_every, state_every = period(params.control_hz), period(params.camera_hz), period(params.state_hz)
	pending = []
	log = {k: [] for k in ("t", "pos", "est", "tag_est", "state", "vcmd", "herr")}
	lander.add_vehicle_state(world.estimated_state(0.0))
	if start:
		ok, msg = lander.start(0.0)
		if not ok:
			raise RuntimeError(msg)
	world.state_feed_alive = True  # hooks can set False to simulate a dead autopilot link
	t = 0.0
	for k in range(int(max_time / dt)):
		t = k * dt
		if k % state_every == 0 and world.state_feed_alive:
			lander.add_vehicle_state(world.estimated_state(t))
		if k % cam_every == 0:
			latency = params.camera_latency_s + rng.uniform(-1.0, 1.0) * params.camera_latency_jitter_s
			pending.append((t + max(latency, 0.0), t, world.detect(t)))
		while pending and pending[0][0] <= t:
			_, stamp, dets = pending.pop(0)
			lander.add_detections(stamp, dets)
		if k % ctrl_every == 0:
			if hook is not None and hook(t, world, lander):
				break
			cmd = lander.update(t)
			world.apply(cmd, t)
			if record:
				log["t"].append(t)
				log["pos"].append(world.pos.copy())
				log["est"].append(world.pos + world.drift)
				log["tag_est"].append(lander.tracker.position if lander.tracker.initialized else np.full(3, np.nan))
				log["state"].append(lander.state.value)
				log["vcmd"].append(cmd.velocity.copy())
				log["herr"].append(np.linalg.norm(world.pos[:2] - world.tag_center[:2]))
		world.step(dt)
		if world.landed:
			lander.notify_landed(t)
			break
	err = float(np.linalg.norm(world.touchdown_xy - world.tag_center[:2])) if world.landed else math.nan
	res = EpisodeResult(seed, scenario, world.landed, lander.state.value, err, t, list(lander.transitions), log)
	res.lander = lander
	res.world = world
	return res


# ---------------------------------------------------------------- reporting
def print_episode(res):
	print(f"seed {res.seed} [{res.scenario}]  landed={res.landed}  final={res.final_state}  "
		f"touchdown error={res.touchdown_error * 100:.1f} cm  time={res.duration:.1f} s")
	for t, a, b, why in res.transitions:
		print(f"  {t:6.2f}s  {a:>13s} -> {b:<13s} {why}")


def _episode_job(args):
	seed, scenario, config = args
	res = run_episode(seed, scenario, config)
	res.log = {}
	res.lander = res.world = None  # keep inter-process traffic small
	return res


def monte_carlo(n, scenario, config, start_seed = 0, workers = None):
	jobs = [(start_seed + i, scenario, config) for i in range(n)]
	if workers == 1:
		results = [_episode_job(j) for j in jobs]
	else:
		from multiprocessing import Pool
		with Pool(workers) as pool:
			results = pool.map(_episode_job, jobs)
	errs = np.array([r.touchdown_error for r in results if r.precision])
	fallback = sum(1 for r in results if r.landed and not r.precision)
	failed = sum(1 for r in results if not r.landed)
	line = f"{scenario:>12s}  n={n:4d}  precision={len(errs):4d}  fallback={fallback:3d}  not-landed={failed:3d}"
	if len(errs):
		line += (f"  error cm: median {np.median(errs) * 100:5.1f}  mean {errs.mean() * 100:5.1f}  "
			f"p95 {np.percentile(errs, 95) * 100:5.1f}  max {errs.max() * 100:5.1f}  "
			f"<10cm {np.mean(errs < 0.10) * 100:5.1f}%")
	line += f"  mean time {np.mean([r.duration for r in results]):5.1f}s"
	print(line)
	worst = sorted((r for r in results if r.landed), key = lambda r: -r.touchdown_error)[:3]
	return results, worst


def plot_episode(res, path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	L = res.log
	t = np.array(L["t"])
	pos = np.array(L["pos"])
	tag = np.array(L["tag_est"])
	vcmd = np.array(L["vcmd"])
	states = L["state"]
	names = [s.value for s in LandingState]
	colors = dict(zip(names, plt.cm.tab10(np.linspace(0, 1, len(names)))))

	fig, ax = plt.subplots(2, 2, figsize = (13, 9))
	a = ax[0, 0]
	for i in range(len(t) - 1):
		a.plot(pos[i:i + 2, 1], pos[i:i + 2, 0], color = colors[states[i]], lw = 2)
	a.plot(0, 0, "ks", ms = 10, label = "pad (truth)")
	a.plot(tag[:, 1], tag[:, 0], "k:", lw = 1, label = "pad estimate")
	a.plot(pos[0, 1], pos[0, 0], "go", label = "start")
	a.set_xlabel("east (m)")
	a.set_ylabel("north (m)")
	a.set_title("top-down path (color = state)")
	a.axis("equal")
	a.legend(loc = "best", fontsize = 8)

	a = ax[0, 1]
	for i in range(len(t) - 1):
		a.plot(t[i:i + 2], -pos[i:i + 2, 2], color = colors[states[i]], lw = 2)
	for s in dict.fromkeys(states):
		a.plot([], [], color = colors[s], lw = 3, label = s)
	a.set_xlabel("time (s)")
	a.set_ylabel("height (m)")
	a.set_title("altitude")
	a.legend(fontsize = 8)

	a = ax[1, 0]
	a.semilogy(t, np.maximum(np.array(L["herr"]), 1e-3))
	a.axhline(0.10, color = "r", ls = "--", lw = 1, label = "10 cm")
	a.set_xlabel("time (s)")
	a.set_ylabel("horizontal error to pad (m)")
	a.set_title("true horizontal error")
	a.legend()

	a = ax[1, 1]
	a.plot(t, vcmd[:, 0], label = "v north")
	a.plot(t, vcmd[:, 1], label = "v east")
	a.plot(t, vcmd[:, 2], label = "v down")
	a.set_xlabel("time (s)")
	a.set_ylabel("m/s")
	a.set_title("velocity commands")
	a.legend()

	fig.suptitle(f"seed {res.seed} [{res.scenario}]: final {res.final_state}, touchdown error {res.touchdown_error * 100:.1f} cm")
	fig.tight_layout()
	fig.savefig(path, dpi = 110)
	plt.close(fig)


def load_config(path, overrides = ()):
	"""YAML config (optional) plus `section.key=value` overrides, e.g. control.kp_xy=1.2."""
	data = {}
	if path:
		import yaml
		with open(path) as f:
			data = yaml.safe_load(f) or {}
		data = data.get("precision_landing", data)
	for item in overrides:
		key, _, raw = item.partition("=")
		section, _, name = key.partition(".")
		if not name:
			raise SystemExit(f"--set expects section.key=value, got '{item}'")
		try:
			import yaml
			value = yaml.safe_load(raw)
		except ImportError:
			import ast
			try:
				value = ast.literal_eval(raw)
			except (ValueError, SyntaxError):
				value = raw
		data.setdefault(section, {})[name] = value
	return LanderConfig.from_dict(data)


def main(argv = None):
	ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--scenario", default = "nominal", choices = list(SCENARIOS) + ["all"])
	ap.add_argument("--seed", type = int, default = 0)
	ap.add_argument("--monte-carlo", type = int, default = 0, metavar = "N")
	ap.add_argument("--config", default = "", help = "YAML config (same format as the ROS node's)")
	ap.add_argument("--set", action = "append", default = [], metavar = "SECTION.KEY=VALUE",
		help = "override a config value, e.g. --set protocol.final_descent=guided (repeatable)")
	ap.add_argument("--plot", default = "sim_output/episode.png", help = "plot path for single episodes ('' to skip)")
	ap.add_argument("--workers", type = int, default = None, help = "Monte Carlo processes (default: all cores)")
	args = ap.parse_args(argv)
	config = load_config(args.config, args.set)

	if args.monte_carlo:
		scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
		for sc in scenarios:
			_, worst = monte_carlo(args.monte_carlo, sc, config, args.seed, args.workers)
			if worst and worst[0].touchdown_error > 0.10:
				print(f"{'':>12s}  worst seeds: " + ", ".join(f"{r.seed} ({r.touchdown_error * 100:.1f} cm)" for r in worst))
		return 0

	scenario = "nominal" if args.scenario == "all" else args.scenario
	res = run_episode(args.seed, scenario, config, record = bool(args.plot))
	print_episode(res)
	if args.plot:
		os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok = True)
		plot_episode(res, args.plot)
		print(f"plot saved to {args.plot}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
