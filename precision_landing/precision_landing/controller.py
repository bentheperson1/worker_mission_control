"""Guidance laws: horizontal tracking, descent scheduling and command smoothing.

The autopilot (PX4) already closes the velocity loop, so we act as the outer
position loop and output NED velocity setpoints:

	v_xy = ff * v_target + Kp * e + Ki * integral(e) + Kd * (v_target - v_vehicle)

With PX4's velocity response approximated as first order (time constant tau),
the closed loop is  tau*e'' + (1 + Kd)*e' + Kp*e = 0, i.e. damping
zeta = (1 + Kd) / (2*sqrt(Kp*tau)). Kp = 0.8, Kd = 0.1, tau ~ 0.3 s gives
zeta ~ 1.1 (no overshoot) - leave margin because camera latency eats phase.

Descent is only allowed inside an alignment cone around the tag, and its speed
is scheduled by height so the vehicle is fast up high and gentle near the pad.
"""
import math

import numpy as np

from .geometry import interp_clamped


def saturate(v, limit):
	"""Scale vector v down so its norm is at most `limit` (keeps direction)."""
	n = float(np.linalg.norm(v))
	if n > limit > 0.0:
		return v * (limit / n)
	return v


class SlewRateLimiter:
	"""Limits how fast a vector command can change (an acceleration limit for velocity commands)."""

	def __init__(self, max_rate):
		self.max_rate = max_rate
		self.value = None

	def reset(self, value):
		self.value = np.array(value, dtype = float)

	def step(self, target, dt):
		target = np.asarray(target, dtype = float)
		if self.value is None:
			self.value = target.copy()
			return self.value.copy()
		if dt <= 0.0:
			return self.value.copy()
		self.value = self.value + saturate(target - self.value, self.max_rate * dt)
		return self.value.copy()


class LandingController:
	def __init__(self, cfg):
		self.cfg = cfg
		self.integral = np.zeros(2)

	def reset(self):
		self.integral = np.zeros(2)

	# ---------------------------------------------------------- schedules
	def max_horizontal_speed(self, height):
		c = self.cfg
		return interp_clamped(height, c.schedule_height_low, c.schedule_height_high, c.max_speed_xy_low, c.max_speed_xy_high)

	def descent_speed(self, height):
		c = self.cfg
		return interp_clamped(height, c.schedule_height_low, c.schedule_height_high, c.descent_speed_low, c.descent_speed_high)

	def allowed_radius(self, height):
		"""Horizontal error allowed while descending at this height (the alignment cone)."""
		c = self.cfg
		return max(c.cone_min_radius, max(height, 0.0) * math.tan(math.radians(c.cone_half_angle_deg)))

	# ---------------------------------------------------------- laws
	def horizontal(self, dt, error_xy, error_rate_xy, target_vel_xy, height, integrate = True):
		"""Horizontal NED velocity command (2,) that drives error_xy (target - vehicle) to zero."""
		c = self.cfg
		e = np.asarray(error_xy, dtype = float)
		de = np.asarray(error_rate_xy, dtype = float)
		ff = c.velocity_feedforward * np.asarray(target_vel_xy, dtype = float)
		vmax = self.max_horizontal_speed(height)
		base = ff + c.kp_xy * e + c.kd_xy * de
		if integrate and c.ki_xy > 0.0 and dt > 0.0 and np.linalg.norm(e) < c.integral_zone:
			unsat = base + c.ki_xy * self.integral
			# conditional integration (anti-windup): don't integrate further into saturation
			if np.linalg.norm(unsat) < vmax or np.dot(e, unsat) < 0.0:
				self.integral = self.integral + e * dt
			self.integral = saturate(self.integral, c.integral_limit / c.ki_xy)
		return saturate(base + c.ki_xy * self.integral, vmax)

	def descent_rate(self, height, horizontal_error):
		"""Downward speed (>= 0): full speed well inside the cone, tapering to 0 at its edge."""
		ratio = horizontal_error / self.allowed_radius(height)
		if ratio >= 1.0:
			return 0.0
		scale = 1.0 if ratio <= 0.5 else (1.0 - ratio) / 0.5
		return self.descent_speed(height) * scale

	def hold(self, position, setpoint, max_xy, max_z):
		"""P controller that holds a 3D position (used while searching/waiting)."""
		e = np.asarray(setpoint, dtype = float) - np.asarray(position, dtype = float)
		v = self.cfg.hold_kp * e
		v[:2] = saturate(v[:2], max_xy)
		v[2] = float(np.clip(v[2], -max_z, max_z))
		return v
