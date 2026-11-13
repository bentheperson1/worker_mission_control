"""Closed-loop simulation world for the precision landing protocol (no ROS).

Models the things that actually break precision landings:
	- PX4 velocity loop lag (first order) + acceleration limit
	- wind: constant push + gusts (Ornstein-Uhlenbeck acceleration disturbance)
	- vehicle tilt while accelerating (moves the camera's view of the tag)
	- odometry drift + noise in the autopilot's position estimate
	- attitude estimate bias/noise, camera mount misalignment, focal-length calibration error
	- lens distortion, pixel noise, minimum detectable tag size, field of view
	- camera latency + jitter, random dropouts, multi-second occlusions, false detections
	- PX4 LAND mode holding position on its own (drifting) estimate after handoff
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from precision_landing import CameraModel, CommandMode, TagDetection, VehicleState
from precision_landing.config import CameraMount
from precision_landing.geometry import euler_to_rot, exp_so3, rot_to_quat, rot_z
from precision_landing.tag_pose import tag_corner_points

G = 9.81


@dataclass
class SimParams:
	physics_dt: float = 0.005
	control_hz: float = 20.0
	camera_hz: float = 15.0
	state_hz: float = 50.0
	camera_latency_s: float = 0.08
	camera_latency_jitter_s: float = 0.02
	# vehicle: PX4 velocity PI loop (defaults MPC_XY_VEL_P_ACC / _I_ACC, MPC_Z_VEL_P_ACC / _I_ACC)
	vel_p_xy: float = 1.8
	vel_i_xy: float = 0.4
	vel_p_z: float = 4.0
	vel_i_z: float = 2.0
	att_time_constant: float = 0.08
	max_accel: float = 4.0
	land_speed: float = 0.6
	land_hold_gain: float = 0.8
	# wind
	wind_accel: float = 0.3  # steady disturbance magnitude (random direction), m/s^2
	gust_std: float = 0.2  # m/s^2 (0.5 is strong turbulence for a small quad)
	gust_tau: float = 1.5  # s
	# camera (DEXI: IMX219 downward, 640x480, ~62 deg HFOV)
	image_width: int = 640
	image_height: int = 480
	hfov_deg: float = 62.2
	lens_dist: tuple = (0.06, -0.12, 0.0005, -0.0005, 0.02)
	calib_focal_error_std: float = 0.01  # fractional
	mount_error_std_deg: float = 0.5
	camera_offset_body: tuple = (0.05, 0.0, 0.03)
	# detection
	tag_size: float = 0.1524
	pixel_noise_px: float = 0.8
	min_tag_px: float = 14.0
	edge_margin_px: float = 3.0
	dropout_prob: float = 0.05
	occlusion_rate_hz: float = 0.0  # chance per second of a multi-second occlusion starting
	occlusion_duration_s: tuple = (0.5, 2.5)
	false_detection_prob: float = 0.0
	# autopilot state estimate
	pos_noise_std: float = 0.01
	drift_std: float = 0.01  # random-walk position drift, m/sqrt(s)
	vel_noise_std: float = 0.02
	att_noise_std_deg: float = 0.3
	att_bias_std_deg: float = 0.5
	rangefinder: bool = False
	range_noise_std: float = 0.01
	# scenario
	start_height: tuple = (2.5, 4.0)
	start_offset: tuple = (0.0, 1.2)  # horizontal distance from the pad, m
	tag_id: int = 0


SCENARIOS = {
	"nominal": {},
	"windy": {"wind_accel": 0.6, "gust_std": 0.5},
	"far": {"start_offset": (2.0, 3.0), "start_height": (3.0, 4.0)},
	"occlusions": {"occlusion_rate_hz": 0.08, "dropout_prob": 0.15},
	"noisy": {"pixel_noise_px": 2.0, "false_detection_prob": 0.02, "drift_std": 0.03, "att_bias_std_deg": 1.0},
	"rangefinder": {"rangefinder": True},
}


def make_params(scenario = "nominal", **overrides):
	return replace(SimParams(), **{**SCENARIOS[scenario], **overrides})


class SimWorld:
	def __init__(self, params, rng, mount = None):
		self.p = params
		self.rng = rng
		pr = params
		# ---- scenario
		self.tag_center = np.array([0.0, 0.0, 0.0])
		self.tag_rotation = rot_z(rng.uniform(-math.pi, math.pi))  # tag lies flat, face up
		heading = rng.uniform(-math.pi, math.pi)
		dist = rng.uniform(*pr.start_offset)
		bearing = rng.uniform(-math.pi, math.pi)
		self.pos = np.array([dist * math.cos(bearing), dist * math.sin(bearing), -rng.uniform(*pr.start_height)])
		self.vel = np.zeros(3)
		self.yaw = heading
		self.vel_err_int = np.zeros(3)
		self.tilt_accel = np.zeros(3)
		ang = rng.uniform(-math.pi, math.pi)
		self.wind = pr.wind_accel * np.array([math.cos(ang), math.sin(ang), 0.0])
		self.gust = np.zeros(3)
		# ---- camera: truth vs what the lander is told
		self.mount = mount if mount is not None else CameraMount(offset_body = pr.camera_offset_body)
		mount_err = exp_so3(rng.normal(0.0, math.radians(pr.mount_error_std_deg), 3))
		self.R_body_cam_true = mount_err @ self.mount.rotation()
		self.cam_offset = np.asarray(self.mount.offset_body, dtype = float)
		self.camera = CameraModel.from_fov(pr.image_width, pr.image_height, pr.hfov_deg, pr.lens_dist)
		focal_scale = 1.0 + rng.normal(0.0, pr.calib_focal_error_std)
		self.lander_camera = CameraModel(
			pr.image_width, pr.image_height,
			self.camera.fx * focal_scale, self.camera.fy * focal_scale,
			self.camera.cx, self.camera.cy, pr.lens_dist,
		)
		# ---- estimator errors
		self.drift = np.zeros(3)
		self.att_bias = exp_so3(rng.normal(0.0, math.radians(pr.att_bias_std_deg), 3) * np.array([1.0, 1.0, 0.3]))
		self.occluded_until = -1.0
		# ---- autopilot mode
		self.mode = "offboard"
		self.vel_cmd = np.zeros(3)
		self.hold_point = None
		self.landed = False
		self.touchdown_xy = None

	# ------------------------------------------------------------ truth
	def attitude(self):
		"""True R_ned_body: thrust vector tilted to produce the commanded acceleration."""
		thrust = self.tilt_accel - np.array([0.0, 0.0, G])
		z_b = -thrust / np.linalg.norm(thrust)
		x_c = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
		y_b = np.cross(z_b, x_c)
		y_b /= np.linalg.norm(y_b)
		x_b = np.cross(y_b, z_b)
		return np.column_stack([x_b, y_b, z_b])

	def apply(self, cmd, now):
		if cmd.mode == CommandMode.VELOCITY:
			self.mode = "offboard"
			self.vel_cmd = np.asarray(cmd.velocity, dtype = float)
			if math.isfinite(cmd.yaw):
				self.yaw = cmd.yaw
		elif cmd.mode == CommandMode.LAND and self.mode != "land":
			self.mode = "land"
			self.hold_point = self._estimated_position()
		elif cmd.mode in (CommandMode.HOLD, CommandMode.NONE) and self.mode not in ("hold", "land"):
			self.mode = "hold"
			self.hold_point = self._estimated_position()

	def step(self, dt):
		if self.landed:
			return
		pr = self.p
		est = self._estimated_position()
		if self.mode == "land":
			v_cmd = np.zeros(3)
			v_cmd[:2] = pr.land_hold_gain * (self.hold_point[:2] - est[:2])
			v_cmd[2] = pr.land_speed
		elif self.mode == "hold":
			v_cmd = pr.land_hold_gain * (self.hold_point - est)
		else:
			v_cmd = self.vel_cmd
		# PX4-like velocity PI controller -> acceleration (tilt) setpoint
		err = v_cmd - self.vel
		kp = np.array([pr.vel_p_xy, pr.vel_p_xy, pr.vel_p_z])
		ki = np.array([pr.vel_i_xy, pr.vel_i_xy, pr.vel_i_z])
		accel = kp * err + ki * self.vel_err_int
		n = np.linalg.norm(accel)
		if n > pr.max_accel:
			accel *= pr.max_accel / n
		else:
			self.vel_err_int += err * dt  # integrate only when not saturated (anti-windup)
		# the airframe reaches the commanded tilt after a short attitude lag
		self.tilt_accel += (accel - self.tilt_accel) * min(1.0, dt / pr.att_time_constant)
		# gusts: Ornstein-Uhlenbeck process
		self.gust += -self.gust * dt / pr.gust_tau + pr.gust_std * math.sqrt(2.0 * dt / pr.gust_tau) * self.rng.normal(0.0, 1.0, 3) * np.array([1.0, 1.0, 0.3])
		self.vel += (self.tilt_accel + self.wind + self.gust) * dt
		self.pos += self.vel * dt
		self.drift += pr.drift_std * math.sqrt(dt) * self.rng.normal(0.0, 1.0, 3) * np.array([1.0, 1.0, 0.5])
		if self.pos[2] >= self.tag_center[2]:
			self.pos[2] = self.tag_center[2]
			self.landed = True
			self.touchdown_xy = self.pos[:2].copy()

	# ------------------------------------------------------------ sensors
	def _estimated_position(self):
		return self.pos + self.drift

	def estimated_state(self, now):
		pr = self.p
		R = self.attitude()
		R_est = R @ self.att_bias @ exp_so3(self.rng.normal(0.0, math.radians(pr.att_noise_std_deg), 3))
		rng_bottom = math.nan
		if pr.rangefinder:
			cos_tilt = max(R[2, 2], 0.5)
			rng_bottom = (self.tag_center[2] - self.pos[2]) / cos_tilt + self.rng.normal(0.0, pr.range_noise_std)
		return VehicleState(
			stamp = now,
			position = self._estimated_position() + self.rng.normal(0.0, pr.pos_noise_std, 3),
			velocity = self.vel + self.rng.normal(0.0, pr.vel_noise_std, 3),
			attitude = rot_to_quat(R_est),
			range_bottom = rng_bottom,
		)

	def detect(self, now):
		"""Simulated apriltag detector output for a frame captured at `now`."""
		pr = self.p
		rng = self.rng
		dets = []
		if pr.false_detection_prob > 0.0 and rng.random() < pr.false_detection_prob:
			c = np.array([rng.uniform(80, pr.image_width - 80), rng.uniform(80, pr.image_height - 80)])
			s = rng.uniform(15, 60)
			a = rng.uniform(0, math.pi)
			sq = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * s / 2
			Rs = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
			dets.append(TagDetection(pr.tag_id, c + sq @ Rs.T, decision_margin = 30.0))
		if pr.occlusion_rate_hz > 0.0 and rng.random() < pr.occlusion_rate_hz / pr.camera_hz:
			self.occluded_until = now + rng.uniform(*pr.occlusion_duration_s)
		if now < self.occluded_until or rng.random() < pr.dropout_prob:
			return dets
		R_ned_body = self.attitude()
		R_ned_cam = R_ned_body @ self.R_body_cam_true
		cam_pos = self.pos + R_ned_body @ self.cam_offset
		corners_ned = self.tag_center + tag_corner_points(pr.tag_size) @ self.tag_rotation.T
		p_cam = (corners_ned - cam_pos) @ R_ned_cam
		if np.any(p_cam[:, 2] < 0.05):
			return dets
		px = self.camera.project(p_cam)
		if not np.all(self.camera.contains(px, pr.edge_margin_px)):
			return dets
		edges = np.linalg.norm(px - np.roll(px, -1, axis = 0), axis = 1)
		if edges.min() < pr.min_tag_px:
			return dets
		px = px + rng.normal(0.0, pr.pixel_noise_px, px.shape)
		dets.append(TagDetection(pr.tag_id, px, decision_margin = 60.0))
		return dets
