"""Every tunable number of the landing protocol, in one place.

Load from a nested dict (YAML file / ROS params) with LanderConfig.from_dict().
Unknown keys raise an error on purpose: a typo in a flight config must not
silently fall back to a default.

Units are SI (m, s, m/s) unless the name ends in _deg.
"""
from dataclasses import asdict, dataclass, field, fields
import math
from typing import List, Optional, Sequence

import numpy as np

from .geometry import rot_z


@dataclass
class TagSpec:
	"""One AprilTag that marks the landing point."""
	tag_id: int
	size: float  # edge length of the black square (the detector's corners), m
	# vector from this tag's center to the landing point, in the tag frame
	# (x right, y down, z into the tag). Leave zero for a single centered tag;
	# use it for multi-tag pads (e.g. a small tag nested in a big one).
	offset: Sequence[float] = (0.0, 0.0, 0.0)


@dataclass
class CameraMount:
	"""Where the downward camera sits on the airframe."""
	# direction the TOP of the image points, measured clockwise from the nose
	# when looking down from above (0 = image top toward the nose, 90 = toward
	# the right side). Ignored if rotation_body_cam is given.
	image_up_heading_deg: float = 0.0
	# camera optical center relative to the vehicle origin, body FRD, m
	offset_body: Sequence[float] = (0.0, 0.0, 0.0)
	# optional full 3x3 R_body_cam (rows), for tilted / unusual mounts
	rotation_body_cam: Optional[Sequence[Sequence[float]]] = None

	def rotation(self):
		if self.rotation_body_cam is not None:
			return np.asarray(self.rotation_body_cam, dtype = float).reshape(3, 3)
		# camera x (image right) -> body right, camera y (image down) -> body back,
		# camera z (optical axis) -> body down
		base = np.array([
			[0.0, -1.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0],
		])
		return rot_z(math.radians(self.image_up_heading_deg)) @ base


@dataclass
class DetectionConfig:
	max_hamming: int = 0  # bit errors allowed in the decoded tag
	min_decision_margin: float = 0.0  # detector confidence; 0 disables the check
	max_reprojection_error_px: float = 3.0  # pose fit quality gate
	max_tag_tilt_deg: float = 60.0  # reject poses whose tag plane is far from level
	# soft prior that the pad is level, within this many degrees (0 disables).
	# removes the depth bias of small, nearly head-on tags; raise it (or set 0)
	# if the pad itself tilts, e.g. on a moving deck
	level_prior_sigma_deg: float = 3.0
	# measurement noise model (see PrecisionLander._measurement_covariance)
	pixel_noise_px: float = 1.0
	attitude_noise_rad: float = 0.01
	noise_floor_m: float = 0.01
	noise_inflation: float = 1.0
	# added to the image timestamp's age if the driver stamps late (s)
	extra_latency_s: float = 0.0


@dataclass
class TrackerConfig:
	# "static": pad doesn't move; its estimate only follows odometry drift (random walk)
	# "moving": near-constant-velocity model, e.g. a pad on a moving vehicle / deck
	target_model: str = "static"
	position_random_walk: float = 0.02  # static model: allowed drift, m/sqrt(s)
	process_accel_noise: float = 0.05  # moving model: acceleration spectral density, m^2/s^3
	velocity_damping_tau: float = 2.0  # moving model: s; 0 = pure constant velocity
	initial_velocity_std: float = 0.1  # m/s
	gate_chi2: float = 16.27  # 3 dof chi-square, 99.9 %
	max_consecutive_rejects: int = 8  # then re-initialize from the measurements
	max_target_speed: float = 1.5  # m/s
	max_extrapolation_s: float = 1.0  # cap on prediction during dropouts


@dataclass
class ControlConfig:
	# tuned in sim against a PX4-default velocity loop (MPC_XY_VEL_P_ACC 1.8,
	# MPC_XY_VEL_I_ACC 0.4); lower kp_xy if the real vehicle oscillates over the pad
	kp_xy: float = 1.5
	ki_xy: float = 0.5
	kd_xy: float = 0.2
	integral_limit: float = 0.3  # max velocity contributed by the integrator, m/s
	integral_zone: float = 0.5  # only integrate within this horizontal error, m
	velocity_feedforward: float = 1.0  # fraction of estimated target velocity fed forward (moving model)
	max_speed_xy_high: float = 1.0  # horizontal speed limit at/above schedule_height_high
	max_speed_xy_low: float = 0.3  # horizontal speed limit at/below schedule_height_low
	descent_speed_high: float = 0.6
	descent_speed_low: float = 0.15
	schedule_height_low: float = 0.6
	schedule_height_high: float = 3.0
	cone_half_angle_deg: float = 12.0  # descend only while error < height * tan(angle)
	cone_min_radius: float = 0.08
	max_accel: float = 2.0  # slew limit on the velocity command, m/s^2
	hold_kp: float = 1.0  # position-hold gain while searching / waiting
	climb_speed: float = 0.5


@dataclass
class ProtocolConfig:
	# SEARCH
	search_pattern: str = "spiral"  # "spiral" or "hover"
	search_timeout_s: float = 45.0  # total time allowed in SEARCH before fallback landing
	search_speed: float = 0.5
	search_max_radius: float = 3.0
	search_overlap: float = 0.3  # fraction of camera footprint shared by adjacent legs
	# ACQUIRE (tag lock confirmation)
	acquire_timeout_s: float = 5.0
	lock_min_detections: int = 6  # accepted measurements needed ...
	lock_window_s: float = 1.0  # ... within this window
	lock_max_std: float = 0.10  # and horizontal estimate std below this, m
	# ALIGN
	align_radius: float = 0.15
	align_hold_s: float = 1.0
	# DESCEND / TOUCHDOWN
	lost_timeout_s: float = 0.5  # no accepted measurement for this long = tag lost
	lost_hold_s: float = 2.0  # wait this long for the tag before reacquiring ...
	lost_climb_speed: float = 0.2  # ... climbing gently meanwhile (widens the camera footprint)
	handoff_height: float = 0.4  # start the final descent below this height ...
	handoff_radius: float = 0.05  # ... if centered within this radius ...
	handoff_max_speed: float = 0.08  # ... and nearly stopped horizontally (m/s)
	handoff_max_wait_s: float = 5.0  # after waiting this long at handoff height, accept commit_radius
	# how the last handoff_height meters are flown:
	#   "land"   - autopilot LAND mode (holds its own position estimate, vision unused)
	#   "guided" - keep steering onto the tracked pad estimate at descent_speed_low
	#              until the autopilot reports touchdown (tag may leave the view)
	final_descent: str = "land"
	guided_timeout_s: float = 10.0  # guided final descent fallback: switch to LAND after this
	commit_height: float = 0.6  # tag lost below this height while centered ...
	commit_radius: float = 0.12  # ... within this radius: land anyway (tag fills the view)
	# REACQUIRE
	reacquire_climb: float = 1.0
	reacquire_max_height: float = 5.0
	reacquire_wait_s: float = 2.0
	max_reacquire_attempts: int = 3
	# SAFETY
	stall_timeout_s: float = 30.0  # no new lowest height (by 10 cm) for this long once tracking = fallback
	state_timeout_s: float = 0.5  # vehicle state older than this = abort
	overall_timeout_s: float = 180.0
	state_history_s: float = 2.0


def _build(cls, data, path):
	data = dict(data or {})
	names = {f.name for f in fields(cls)}
	unknown = sorted(set(data) - names)
	if unknown:
		raise ValueError(f"unknown key(s) in '{path}': {', '.join(unknown)}")
	return cls(**data)


@dataclass
class LanderConfig:
	tags: List[TagSpec] = field(default_factory = lambda: [TagSpec(tag_id = 0, size = 0.1524)])
	camera_mount: CameraMount = field(default_factory = CameraMount)
	detection: DetectionConfig = field(default_factory = DetectionConfig)
	tracker: TrackerConfig = field(default_factory = TrackerConfig)
	control: ControlConfig = field(default_factory = ControlConfig)
	protocol: ProtocolConfig = field(default_factory = ProtocolConfig)

	@classmethod
	def from_dict(cls, data):
		data = dict(data or {})
		sections = {
			"camera_mount": CameraMount,
			"detection": DetectionConfig,
			"tracker": TrackerConfig,
			"control": ControlConfig,
			"protocol": ProtocolConfig,
		}
		unknown = sorted(set(data) - set(sections) - {"tags"})
		if unknown:
			raise ValueError(f"unknown top-level config key(s): {', '.join(unknown)}")
		kwargs = {name: _build(sub, data.get(name), name) for name, sub in sections.items()}
		if "tags" in data:
			kwargs["tags"] = [_build(TagSpec, t, "tags[]") for t in data["tags"]]
		cfg = cls(**kwargs)
		cfg.validate()
		return cfg

	def to_dict(self):
		return asdict(self)

	def validate(self):
		problems = []
		if not self.tags:
			problems.append("at least one tag is required")
		ids = [t.tag_id for t in self.tags]
		if len(ids) != len(set(ids)):
			problems.append("tag ids must be unique")
		for t in self.tags:
			if t.size <= 0.0:
				problems.append(f"tag {t.tag_id}: size must be > 0")
			if len(t.offset) != 3:
				problems.append(f"tag {t.tag_id}: offset must have 3 elements")
		R = self.camera_mount.rotation()
		if not np.allclose(R @ R.T, np.eye(3), atol = 1e-6) or np.linalg.det(R) < 0.0:
			problems.append("camera_mount rotation is not a proper rotation matrix")
		p = self.protocol
		if p.search_pattern not in ("spiral", "hover"):
			problems.append("protocol.search_pattern must be 'spiral' or 'hover'")
		if p.final_descent not in ("land", "guided"):
			problems.append("protocol.final_descent must be 'land' or 'guided'")
		if self.tracker.target_model not in ("static", "moving"):
			problems.append("tracker.target_model must be 'static' or 'moving'")
		if p.handoff_height <= 0.0 or p.commit_height < p.handoff_height:
			problems.append("need 0 < handoff_height <= commit_height")
		if p.handoff_radius <= 0.0 or p.align_radius < p.handoff_radius:
			problems.append("need 0 < handoff_radius <= align_radius")
		if p.lost_timeout_s <= 0.0 or p.lost_hold_s < p.lost_timeout_s:
			problems.append("need 0 < lost_timeout_s <= lost_hold_s")
		c = self.control
		if c.max_accel <= 0.0 or c.kp_xy <= 0.0:
			problems.append("control.max_accel and control.kp_xy must be > 0")
		if problems:
			raise ValueError("invalid precision landing config: " + "; ".join(problems))
