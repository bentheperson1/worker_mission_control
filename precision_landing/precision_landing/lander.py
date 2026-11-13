"""The precision landing protocol: a state machine tying together tag pose
estimation, target tracking and guidance.

Framework-agnostic usage (ROS node, simulator and tests all drive it this way):

	lander = PrecisionLander(LanderConfig(), camera)
	lander.add_vehicle_state(state)           # every autopilot pose update (30-100 Hz)
	lander.add_detections(stamp, detections)  # every camera frame (stamp = capture time)
	ok, msg = lander.start(now)               # when the mission wants to land
	cmd = lander.update(now)                  # at a fixed rate (10-50 Hz) -> Command
	lander.notify_landed(now)                 # when the autopilot detects touchdown

States
	SEARCH         hover, or fly an expanding square spiral, until the tag is seen
	ACQUIRE        hold still until the tag is tracked consistently ("lock")
	ALIGN          center over the tag at constant altitude
	DESCEND        descend inside an alignment cone while tracking the tag
	REACQUIRE      tag lost: climb above its last known position to get it back
	TOUCHDOWN      low and centered: autopilot LAND (or guided descent) finishes the job
	LANDED         touchdown confirmed by the autopilot
	FALLBACK_LAND  precision landing could not be completed: land in place
	ABORTED        abort requested or vehicle data lost: autopilot position hold

	         sighted        locked         aligned          low + centered
	SEARCH --------> ACQUIRE ------> ALIGN --------> DESCEND --------------> TOUCHDOWN -> LANDED
	  ^  ^              |              ^  |              |
	  |  +--- lost -----+              |  +---- lost ----+--> REACQUIRE (climb)
	  |                                +------ locked ---------------+   |
	  +-------------------------- climb done, no lock -------------------+
	search timeout / no descent progress / too many reacquires / overall timeout -> FALLBACK_LAND
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .config import LanderConfig
from .controller import LandingController, SlewRateLimiter
from .datatypes import ACTIVE_STATES, LAND_STATES, Command, CommandMode, LandingState
from .estimator import TargetTracker, UpdateResult
from .geometry import quat_to_rot, yaw_from_quat
from .history import StateHistory
from .search import SpiralSearch
from .tag_pose import estimate_tag_pose

S = LandingState


@dataclass
class TargetView:
	"""The tracked landing point as seen from the vehicle at the current instant."""
	position: np.ndarray  # landing point, NED
	velocity: np.ndarray  # landing point velocity, NED
	error_xy: np.ndarray  # target - vehicle, horizontal
	horiz_error: float
	height: float  # height above the pad used for decisions (conservative)
	height_tag: float  # height above the pad from vision alone
	std_xy: float  # 1-sigma horizontal uncertainty of the estimate (worst axis)
	age: float  # seconds since the last accepted measurement
	fresh: bool  # age <= lost_timeout


class PrecisionLander:
	def __init__(self, config = None, camera = None):
		self.cfg = config if config is not None else LanderConfig()
		self.cfg.validate()
		self.camera = camera
		self.history = StateHistory(max_age = self.cfg.protocol.state_history_s)
		self.tracker = TargetTracker(self.cfg.tracker)
		self.controller = LandingController(self.cfg.control)
		self.slew = SlewRateLimiter(self.cfg.control.max_accel)
		self.tags = {t.tag_id: t for t in self.cfg.tags}
		self.R_body_cam = self.cfg.camera_mount.rotation()
		self.cam_offset_body = np.asarray(self.cfg.camera_mount.offset_body, dtype = float)
		self.on_transition = None  # optional callback(old_state, new_state, reason)
		self.state = S.IDLE
		self.reason = ""
		self._reset_run()

	def _reset_run(self):
		self.transitions = []  # (time, from, to, reason)
		self.t_start = None
		self.t_state = None
		self.t_last_update = None
		self.t_last_accept = None
		self.accept_times = deque()
		self.vehicle = None
		self.target = None
		self.yaw_hold = math.nan
		self.hold_point = None
		self.hold_z = None
		self.search = None
		self.search_center = None
		self.search_time = 0.0
		self.aligned_since = None
		self.handoff_wait_since = None
		self.last_fresh_error = None
		self.lost_handled = False
		self.guided_expired = False
		self.best_height = None
		self.t_progress = None
		self.reacquire_attempts = 0
		self.reacquire_goal = None
		self.reached_since = None
		self.last_tag_body = None
		self.last_command = None
		self.stats = {
			"frames": 0,
			"detections": 0,
			"used": 0,
			"rejected_quality": 0,
			"rejected_pose": 0,
			"rejected_gate": 0,
			"no_vehicle_state": 0,
			"tracker_resets": 0,
			"ignored_landed": 0,
		}

	# ================================================================== inputs
	def set_camera(self, camera):
		self.camera = camera

	def add_vehicle_state(self, state):
		self.history.add(state)

	def add_detections(self, stamp, detections):
		"""Feed one camera frame. `stamp` is the image capture time. Returns measurements used."""
		if self.state not in ACTIVE_STATES or self.camera is None:
			return 0
		self.stats["frames"] += 1
		dc = self.cfg.detection
		stamp = stamp - dc.extra_latency_s
		relevant = [d for d in detections if d.tag_id in self.tags]
		if not relevant:
			return 0
		vehicle = self.history.at(stamp)
		if vehicle is None:
			self.stats["no_vehicle_state"] += 1
			return 0
		R_ned_body = quat_to_rot(vehicle.attitude)
		R_ned_cam = R_ned_body @ self.R_body_cam
		cam_pos = vehicle.position + R_ned_body @ self.cam_offset_body
		min_level = math.cos(math.radians(dc.max_tag_tilt_deg))
		down_cam = R_ned_cam.T @ np.array([0.0, 0.0, 1.0])  # world "down" seen from the camera
		used = 0
		for det in relevant:
			self.stats["detections"] += 1
			if det.hamming > dc.max_hamming or det.decision_margin < dc.min_decision_margin:
				self.stats["rejected_quality"] += 1
				continue
			spec = self.tags[det.tag_id]
			pose = estimate_tag_pose(
				det.corners, self.camera, spec.size,
				level_normal_cam = down_cam if dc.level_prior_sigma_deg > 0.0 else None,
				tilt_sigma_rad = math.radians(dc.level_prior_sigma_deg),
				pixel_sigma = dc.pixel_noise_px,
			)
			if pose is None or pose.reprojection_rms_px > dc.max_reprojection_error_px:
				self.stats["rejected_pose"] += 1
				continue
			# tag normal vs vertical; abs() tolerates either corner-winding convention
			if abs((R_ned_cam @ pose.rotation[:, 2])[2]) < min_level:
				self.stats["rejected_pose"] += 1
				continue
			p_cam = pose.translation + pose.rotation @ np.asarray(spec.offset, dtype = float)
			z = cam_pos + R_ned_cam @ p_cam
			result = self.tracker.update(stamp, z, self._measurement_covariance(p_cam, spec.size, R_ned_cam))
			if result == UpdateResult.REJECTED:
				self.stats["rejected_gate"] += 1
				continue
			if result == UpdateResult.REINIT:
				self.stats["tracker_resets"] += 1
				self.accept_times.clear()
			self.stats["used"] += 1
			used += 1
			self.accept_times.append(stamp)
			self.t_last_accept = stamp if self.t_last_accept is None else max(self.t_last_accept, stamp)
			self.last_tag_body = self.R_body_cam @ p_cam + self.cam_offset_body
		return used

	def _measurement_covariance(self, p_cam, tag_size, R_ned_cam):
		"""Range-dependent noise of one tag position measurement, rotated into NED.

		Lateral: pixel noise and attitude error both scale with distance d.
		Depth: d = f*s/size_px, so sigma_d = d^2 / (f*s) * sigma_size_px - it grows
		with d^2 and dominates far away, which is why it gets its own axis.
		"""
		dc = self.cfg.detection
		d = max(float(p_cam[2]), 0.05)
		f = self.camera.focal
		floor = dc.noise_floor_m ** 2
		lateral = (dc.pixel_noise_px * d / f) ** 2 + (dc.attitude_noise_rad * d) ** 2 + floor
		depth = (math.sqrt(2.0) * dc.pixel_noise_px * d * d / (f * tag_size)) ** 2 + floor
		C = np.diag([lateral, lateral, depth]) * dc.noise_inflation
		return R_ned_cam @ C @ R_ned_cam.T

	# ================================================================ commands
	def start(self, now):
		"""Begin the protocol from the current position. Returns (ok, message)."""
		if self.state not in (S.IDLE, S.LANDED, S.ABORTED):
			return False, f"already running (state {self.state.value})"
		vehicle = self.history.latest()
		if vehicle is None or now - vehicle.stamp > self.cfg.protocol.state_timeout_s:
			return False, "no recent vehicle state"
		self._reset_run()
		self.tracker.reset()
		self.controller.reset()
		self.vehicle = vehicle
		self.t_start = now
		self.t_last_update = now
		self.yaw_hold = yaw_from_quat(vehicle.attitude)
		self.slew.reset(vehicle.velocity)
		self._enter(S.SEARCH, now, "started")
		return True, "precision landing started"

	def abort(self, now, reason = "abort requested"):
		if self.state in (S.IDLE, S.LANDED, S.ABORTED):
			return False
		self._enter(S.ABORTED, now, reason)
		return True

	def reset(self):
		"""Return to IDLE (only meaningful after the run is over)."""
		self.state = S.IDLE
		self.reason = ""
		self._reset_run()

	def notify_landed(self, now):
		"""Autopilot reports touchdown. Ignored if we believe we're still well above the pad."""
		if self.state in LAND_STATES:
			self._enter(S.LANDED, now, "touchdown detected")
			return True
		if self.state in ACTIVE_STATES and self.target is not None and self.target.height <= self.cfg.protocol.commit_height:
			self._enter(S.LANDED, now, "touchdown detected during descent")
			return True
		if self.state in ACTIVE_STATES:
			self.stats["ignored_landed"] += 1
		return False

	# ================================================================== update
	def update(self, now):
		"""Run one control step and return the Command to execute."""
		if self.state in (S.IDLE, S.LANDED):
			return self._command(CommandMode.NONE)
		if self.state == S.ABORTED:
			return self._command(CommandMode.HOLD)
		dt = min(max(now - self.t_last_update, 0.0), 0.2)
		self.t_last_update = now
		vehicle = self.history.latest()
		if vehicle is None or now - vehicle.stamp > self.cfg.protocol.state_timeout_s:
			self._enter(S.ABORTED, now, "vehicle state stale")
			return self._command(CommandMode.HOLD)
		self.vehicle = vehicle
		self.target = self._target_view(now)
		if self.state not in LAND_STATES and now - self.t_start > self.cfg.protocol.overall_timeout_s:
			self._enter(S.FALLBACK_LAND, now, "overall timeout")
		self._check_progress(now)
		if self._autopilot_landing():
			return self._command(CommandMode.LAND)

		step = {
			S.SEARCH: self._step_search,
			S.ACQUIRE: self._step_acquire,
			S.ALIGN: self._step_align,
			S.DESCEND: self._step_descend,
			S.REACQUIRE: self._step_reacquire,
			S.TOUCHDOWN: self._step_guided_touchdown,
		}[self.state]
		v = step(now, dt)
		if self._autopilot_landing():
			return self._command(CommandMode.LAND)
		return self._command(CommandMode.VELOCITY, self.slew.step(v, dt))

	def _check_progress(self, now):
		"""Watchdog: once tracking the tag, we must keep reaching new lowest heights.

		Catches wind too strong to hold the alignment cone (endless ALIGN, or DESCEND
		stalled at one height) and slow reacquire loops, which per-state timeouts miss.
		"""
		if self.state not in (S.ALIGN, S.DESCEND, S.REACQUIRE) or self.target is None:
			return
		if self.best_height is None or self.target.height < self.best_height - 0.1:
			self.best_height = self.target.height
			self.t_progress = now
		elif now - self.t_progress > self.cfg.protocol.stall_timeout_s:
			self._enter(S.FALLBACK_LAND, now, f"no descent progress for {self.cfg.protocol.stall_timeout_s:.0f} s (stuck near {self.best_height:.1f} m)")

	def _autopilot_landing(self):
		"""True when the autopilot's LAND mode should be flying the vehicle."""
		if self.state == S.FALLBACK_LAND:
			return True
		if self.state == S.TOUCHDOWN:
			return self.cfg.protocol.final_descent == "land" or self.guided_expired
		return False

	def _command(self, mode, velocity = None):
		cmd = Command(
			mode = mode,
			state = self.state,
			velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype = float),
			yaw = self.yaw_hold,
		)
		self.last_command = cmd
		return cmd

	def _target_view(self, now):
		if not self.tracker.initialized:
			return None
		p, v, P = self.tracker.predicted(now)
		rel = p - self.vehicle.position
		height_tag = float(rel[2])
		height = height_tag
		rng = self.vehicle.range_bottom
		if math.isfinite(rng) and rng > 0.0:
			height = min(height, rng)  # trust whichever says we're lower
		age = math.inf if self.t_last_accept is None else now - self.t_last_accept
		return TargetView(
			position = p,
			velocity = v,
			error_xy = rel[:2],
			horiz_error = float(np.linalg.norm(rel[:2])),
			height = height,
			height_tag = height_tag,
			std_xy = float(math.sqrt(max(np.linalg.eigvalsh(P[:2, :2])))),
			age = age,
			fresh = age <= self.cfg.protocol.lost_timeout_s,
		)

	def _has_lock(self, now, tgt):
		p = self.cfg.protocol
		while self.accept_times and self.accept_times[0] < now - p.lock_window_s:
			self.accept_times.popleft()
		return (
			tgt is not None and tgt.fresh
			and len(self.accept_times) >= p.lock_min_detections
			and tgt.std_xy <= p.lock_max_std
		)

	# ------------------------------------------------------------ state steps
	def _step_search(self, now, dt):
		p = self.cfg.protocol
		self.search_time += dt
		if self.target is not None and self.target.fresh:
			self._enter(S.ACQUIRE, now, "tag sighted")
			return self._hold(self.hold_point)
		if self.search_time > p.search_timeout_s:
			self._enter(S.FALLBACK_LAND, now, "tag not found (search timeout)")
			return np.zeros(3)
		v = self._hold(self.hold_point)
		if p.search_pattern == "spiral":
			if self.search is None:
				center = self.search_center if self.search_center is not None else self.hold_point[:2]
				self.search = SpiralSearch(center, self._search_step(), p.search_max_radius, p.search_speed, self.cfg.control.hold_kp)
			v[:2] = self.search.velocity(self.vehicle.position[:2])
		return v

	def _step_acquire(self, now, dt):
		p = self.cfg.protocol
		tgt = self.target
		if self._has_lock(now, tgt):
			self._enter(S.ALIGN, now, "tag locked")
		elif tgt is None or tgt.age > p.lost_hold_s:
			self._enter(S.SEARCH, now, "tag lost before lock")
		elif now - self.t_state > p.acquire_timeout_s:
			self._enter(S.SEARCH, now, "could not confirm lock")
		return self._hold(self.hold_point)

	def _step_align(self, now, dt):
		p = self.cfg.protocol
		tgt = self.target
		if tgt is None or tgt.age > p.lost_hold_s:
			self._enter(S.REACQUIRE, now, "tag lost while aligning")
			return np.zeros(3)
		v = np.zeros(3)
		v[:2] = self._track_xy(dt, tgt)
		v[2] = self._hold_altitude(self.hold_z)
		if tgt.fresh and tgt.horiz_error <= p.align_radius:
			if self.aligned_since is None:
				self.aligned_since = now
			if now - self.aligned_since >= p.align_hold_s and self._has_lock(now, tgt):
				self._enter(S.DESCEND, now, f"aligned within {tgt.horiz_error * 100:.1f} cm")
		else:
			self.aligned_since = None
		return v

	def _step_descend(self, now, dt):
		p = self.cfg.protocol
		tgt = self.target
		if tgt is None:
			self._enter(S.REACQUIRE, now, "no target estimate")
			return np.zeros(3)
		v = np.zeros(3)
		v[:2] = self._track_xy(dt, tgt)
		if tgt.fresh:
			self.last_fresh_error = tgt.horiz_error
			self.lost_handled = False
			if tgt.height <= p.handoff_height:
				if self.handoff_wait_since is None:
					self.handoff_wait_since = now
				waited = now - self.handoff_wait_since
				speed = float(np.linalg.norm(self.vehicle.velocity[:2] - tgt.velocity[:2]))
				centered = tgt.horiz_error <= p.handoff_radius and speed <= p.handoff_max_speed
				if centered or (waited >= p.handoff_max_wait_s and tgt.horiz_error <= p.commit_radius):
					self._enter(S.TOUCHDOWN, now, f"handoff at {tgt.height:.2f} m, error {tgt.horiz_error * 100:.1f} cm")
					return v
				v[2] = 0.0  # wait at handoff height until centered
			else:
				self.handoff_wait_since = None
				v[2] = self.controller.descent_rate(tgt.height, tgt.horiz_error)
				# no vertical feedforward: the pad's vertical velocity is estimated
				# from noisy depth and would make the vehicle bob up and down
		else:
			# decide once, at the moment the tag disappears, using the last error we
			# actually measured (not a prediction that ages while we wait)
			if not self.lost_handled:
				self.lost_handled = True
				last = self.last_fresh_error
				if tgt.height <= p.commit_height and last is not None and last <= p.commit_radius:
					self._enter(S.TOUCHDOWN, now, f"tag lost at {tgt.height:.2f} m while centered ({last * 100:.1f} cm): committing")
					return v
			# never descend blind; climb gently so the footprint grows and the tag comes back
			v[2] = -p.lost_climb_speed
			if tgt.age > p.lost_hold_s:
				self._enter(S.REACQUIRE, now, "tag lost during descent")
		return v

	def _step_reacquire(self, now, dt):
		p = self.cfg.protocol
		if self._has_lock(now, self.target):
			self._enter(S.ALIGN, now, "tag reacquired")
			return np.zeros(3)
		goal = self.reacquire_goal
		c = self.cfg.control
		v = self.controller.hold(self.vehicle.position, goal, c.max_speed_xy_low, c.climb_speed)
		if abs(self.vehicle.position[2] - goal[2]) < 0.15:
			if self.reached_since is None:
				self.reached_since = now
			if now - self.reached_since >= p.reacquire_wait_s:
				self.search_center = goal[:2].copy()
				self.search = None
				self._enter(S.SEARCH, now, "climbed but tag not reacquired")
		return v

	def _step_guided_touchdown(self, now, dt):
		"""final_descent == "guided": keep steering onto the pad estimate down to contact.

		The tag usually fills or leaves the view in the last few decimeters; the
		estimate lives in the world frame, so odometry carries us the rest of the
		way (drift over ~2 s is millimeters).
		"""
		p = self.cfg.protocol
		if now - self.t_state > p.guided_timeout_s or self.target is None:
			self.guided_expired = True
			self.reason = "guided final descent timed out: autopilot LAND"
			return np.zeros(3)
		v = np.zeros(3)
		v[:2] = self._track_xy(dt, self.target)
		v[2] = self.cfg.control.descent_speed_low
		return v

	# ------------------------------------------------------------ helpers
	def _track_xy(self, dt, tgt):
		error_rate = tgt.velocity[:2] - self.vehicle.velocity[:2]
		return self.controller.horizontal(dt, tgt.error_xy, error_rate, tgt.velocity[:2], tgt.height, integrate = tgt.fresh)

	def _hold(self, point):
		c = self.cfg.control
		return self.controller.hold(self.vehicle.position, point, c.max_speed_xy_high, c.climb_speed)

	def _hold_altitude(self, z):
		c = self.cfg.control
		return float(np.clip(c.hold_kp * (z - self.vehicle.position[2]), -c.climb_speed, c.climb_speed))

	def _search_step(self):
		"""Spiral leg spacing from the camera footprint at the current height."""
		if self.camera is None:
			return 1.0
		height = max(-float(self.vehicle.position[2]), 0.5)  # local origin ~ ground at arming
		footprint = 2.0 * height * math.tan(min(self.camera.hfov, self.camera.vfov) / 2.0)
		return float(np.clip(footprint * (1.0 - self.cfg.protocol.search_overlap), 0.3, 3.0))

	def _enter(self, new_state, now, reason):
		p = self.cfg.protocol
		if new_state == S.REACQUIRE:
			self.reacquire_attempts += 1
			if self.reacquire_attempts > p.max_reacquire_attempts:
				new_state = S.FALLBACK_LAND
				reason = f"{reason}; reacquire attempts exhausted"
		old = self.state
		self.state = new_state
		self.reason = reason
		self.t_state = now
		self.transitions.append((now, old.value, new_state.value, reason))
		v = self.vehicle
		if new_state in (S.SEARCH, S.ACQUIRE):
			self.hold_point = v.position.copy()
		elif new_state == S.ALIGN:
			self.hold_z = float(v.position[2])
			self.aligned_since = None
			self.controller.reset()
		elif new_state == S.DESCEND:
			self.handoff_wait_since = None
			self.last_fresh_error = None
			self.lost_handled = False
		elif new_state == S.REACQUIRE:
			if self.tracker.initialized:
				xy = self.tracker.position[:2]
				ground_z = float(self.tracker.position[2])
			else:
				xy = v.position[:2]
				ground_z = 0.0
			z_goal = max(float(v.position[2]) - p.reacquire_climb, ground_z - p.reacquire_max_height)
			self.reacquire_goal = np.array([xy[0], xy[1], z_goal])
			self.reached_since = None
		if self.on_transition is not None:
			self.on_transition(old, new_state, reason)

	# ================================================================= status
	def status(self):
		"""JSON-serializable snapshot for logging / telemetry / the dashboard."""
		t = self.target
		now = self.t_last_update
		out = {
			"state": self.state.value,
			"reason": self.reason,
			"time_in_state": None if (now is None or self.t_state is None) else round(now - self.t_state, 2),
			"elapsed": None if (now is None or self.t_start is None) else round(now - self.t_start, 2),
			"camera_ready": self.camera is not None,
			"reacquire_attempts": self.reacquire_attempts,
			"stats": dict(self.stats),
		}
		if t is not None:
			out.update({
				"target_ned": [round(float(x), 3) for x in t.position],
				"horizontal_error_m": round(t.horiz_error, 3),
				"height_m": round(t.height, 3),
				"height_vision_m": round(t.height_tag, 3),
				"target_std_m": round(t.std_xy, 3),
				"tag_age_s": None if math.isinf(t.age) else round(t.age, 2),
				"tag_fresh": t.fresh,
			})
		if self.vehicle is not None and math.isfinite(self.vehicle.range_bottom):
			out["range_bottom_m"] = round(float(self.vehicle.range_bottom), 3)
		if self.last_tag_body is not None:
			# where the camera last saw the pad relative to the airframe (FRD) - handy
			# for verifying the camera mount: a tag ahead of the nose must read +x
			out["last_tag_body_frd"] = [round(float(x), 3) for x in self.last_tag_body]
		if self.last_command is not None:
			out["command"] = {
				"mode": self.last_command.mode.value,
				"velocity_ned": [round(float(x), 3) for x in self.last_command.velocity],
			}
		return out
