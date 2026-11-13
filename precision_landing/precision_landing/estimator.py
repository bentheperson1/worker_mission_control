"""Kalman filter that tracks the landing point in the local NED frame.

State x = [px, py, pz, vx, vy, vz] (landing-point position and velocity).

Why track the target in the world frame instead of steering on raw camera
offsets?
	- the vehicle's own motion is already removed (we use its pose at capture time)
	- dropped frames are bridged by prediction instead of producing a zero error
	- measurements are weighted by how trustworthy they are (range-dependent noise)
	- outliers are rejected with a chi-square gate on the innovation
	- a slowly drifting estimate (odometry drift) or a slowly moving pad is
	  followed through the velocity states

Process models
	"static"  position random walk, velocity pinned to zero. The pad doesn't move,
	          but its estimate must follow the slow drift of the autopilot's local
	          frame. Velocity states would only pick up noise and then extrapolate
	          it during dropouts, so they are switched off.
	"moving"  near-constant velocity with velocity decay (time constant tau), for
	          a pad that moves (deck of another vehicle). tau = 0 -> pure CV.
"""
import math

import numpy as np

_H = np.hstack([np.eye(3), np.zeros((3, 3))])


class UpdateResult:
	INIT = "init"
	ACCEPTED = "accepted"
	REJECTED = "rejected"
	REINIT = "reinit"


class TargetTracker:
	def __init__(self, cfg):
		self.cfg = cfg
		self.reset()

	def reset(self):
		self.x = np.zeros(6)
		self.P = np.eye(6)
		self.t = None
		self.initialized = False
		self.consecutive_rejects = 0
		self.num_updates = 0
		self.num_rejects = 0
		self.last_nis = math.nan  # normalized innovation squared of the last measurement

	# ------------------------------------------------------------------ model
	@property
	def static(self):
		return self.cfg.target_model == "static"

	def _transition(self, dt):
		if self.static:
			F = np.zeros((6, 6))
			F[:3, :3] = np.eye(3)
			Q = np.zeros((6, 6))
			Q[:3, :3] = (self.cfg.position_random_walk ** 2) * dt * np.eye(3)
			return F, Q
		tau = self.cfg.velocity_damping_tau
		if tau and tau > 0.0 and math.isfinite(tau):
			decay = math.exp(-dt / tau)
			gain = tau * (1.0 - decay)
		else:
			decay = 1.0
			gain = dt
		F = np.eye(6)
		F[:3, 3:] = gain * np.eye(3)
		F[3:, 3:] = decay * np.eye(3)
		q = self.cfg.process_accel_noise
		Q = np.zeros((6, 6))
		Q[:3, :3] = (dt ** 3 / 3.0) * q * np.eye(3)
		Q[:3, 3:] = (dt ** 2 / 2.0) * q * np.eye(3)
		Q[3:, :3] = Q[:3, 3:]
		Q[3:, 3:] = dt * q * np.eye(3)
		return F, Q

	def _initialize(self, t, z, R):
		self.x = np.concatenate([z, np.zeros(3)])
		self.P = np.zeros((6, 6))
		self.P[:3, :3] = R
		vel_std = 0.0 if self.static else self.cfg.initial_velocity_std
		self.P[3:, 3:] = (vel_std ** 2) * np.eye(3)
		self.t = t
		self.initialized = True
		self.consecutive_rejects = 0
		self.num_updates += 1
		self.last_nis = 0.0

	# ------------------------------------------------------------------ public
	def predict(self, t):
		"""Advance the filter to time t (no-op for t in the past)."""
		if not self.initialized or t <= self.t:
			return
		F, Q = self._transition(t - self.t)
		self.x = F @ self.x
		self.P = F @ self.P @ F.T + Q
		self.t = t

	def predicted(self, t):
		"""(position, velocity, position covariance) at time t without changing the filter."""
		if not self.initialized:
			return None, None, None
		dt = min(max(t - self.t, 0.0), self.cfg.max_extrapolation_s)
		F, Q = self._transition(dt)
		x = F @ self.x
		P = F @ self.P @ F.T + Q
		return x[:3].copy(), x[3:].copy(), P[:3, :3].copy()

	def update(self, t, z, R):
		"""Fuse a landing-point position measurement z (NED) with covariance R."""
		z = np.asarray(z, dtype = float)
		R = np.asarray(R, dtype = float)
		if not self.initialized:
			self._initialize(t, z, R)
			return UpdateResult.INIT
		self.predict(t)  # late measurements (t < self.t) update in place - fine for a slow target
		y = z - self.x[:3]
		S = self.P[:3, :3] + R
		S_inv = np.linalg.inv(S)
		nis = float(y @ S_inv @ y)
		self.last_nis = nis
		if nis > self.cfg.gate_chi2:
			self.consecutive_rejects += 1
			self.num_rejects += 1
			if self.consecutive_rejects >= self.cfg.max_consecutive_rejects:
				# measurements consistently disagree with the estimate: the estimate
				# is what's wrong (bad first lock, long dropout, odometry jump)
				self._initialize(t, z, R)
				return UpdateResult.REINIT
			return UpdateResult.REJECTED
		K = self.P[:, :3] @ S_inv
		self.x = self.x + K @ y
		I_KH = np.eye(6) - K @ _H
		self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form: stays symmetric PSD
		speed = np.linalg.norm(self.x[3:])
		if speed > self.cfg.max_target_speed:
			self.x[3:] *= self.cfg.max_target_speed / speed
		self.consecutive_rejects = 0
		self.num_updates += 1
		return UpdateResult.ACCEPTED

	@property
	def position(self):
		return self.x[:3].copy()

	@property
	def velocity(self):
		return self.x[3:].copy()
