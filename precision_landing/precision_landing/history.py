"""Short buffer of vehicle states so a camera frame can be paired with the pose
the vehicle had when the frame was *captured*, not when it was processed.

Detection latency on a Pi-class computer is easily 50-150 ms. At 1 m/s that is
5-15 cm of error, and a few degrees of tilt during that window moves the tag
estimate further still, so we interpolate the pose at the image timestamp.
"""
from collections import deque

import numpy as np

from .datatypes import VehicleState
from .geometry import quat_slerp


class StateHistory:
	def __init__(self, max_age = 2.0, max_forward_gap = 0.1):
		self.max_age = max_age
		self.max_forward_gap = max_forward_gap
		self._buf = deque()

	def __len__(self):
		return len(self._buf)

	def clear(self):
		self._buf.clear()

	def add(self, state):
		if self._buf and state.stamp <= self._buf[-1].stamp:
			return  # out-of-order or duplicate sample
		self._buf.append(state)
		while self._buf and self._buf[-1].stamp - self._buf[0].stamp > self.max_age:
			self._buf.popleft()

	def latest(self):
		return self._buf[-1] if self._buf else None

	def at(self, stamp):
		"""Interpolated state at `stamp`, or None if `stamp` is outside the buffer."""
		if not self._buf:
			return None
		newest = self._buf[-1]
		if stamp >= newest.stamp:
			return newest if stamp - newest.stamp <= self.max_forward_gap else None
		if stamp < self._buf[0].stamp:
			return None
		# binary search for the bracketing pair
		lo, hi = 0, len(self._buf) - 1
		while hi - lo > 1:
			mid = (lo + hi) // 2
			if self._buf[mid].stamp <= stamp:
				lo = mid
			else:
				hi = mid
		a, b = self._buf[lo], self._buf[hi]
		u = (stamp - a.stamp) / (b.stamp - a.stamp)
		if np.isfinite(a.range_bottom) and np.isfinite(b.range_bottom):
			rng = a.range_bottom + u * (b.range_bottom - a.range_bottom)
		else:
			rng = b.range_bottom
		return VehicleState(
			stamp = stamp,
			position = a.position + u * (b.position - a.position),
			velocity = a.velocity + u * (b.velocity - a.velocity),
			attitude = quat_slerp(a.attitude, b.attitude, u),
			range_bottom = rng,
		)
