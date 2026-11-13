"""Expanding square-spiral search used when the tag is not in view.

GPS/odometry error at the end of transit can easily put the pad a couple of
meters outside the camera footprint. The spiral's leg spacing is derived from
the footprint so consecutive legs overlap.
"""
import numpy as np

from .controller import saturate

# north, east, south, west in NED
_DIRECTIONS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])


def square_spiral(center, step, max_radius):
	"""Waypoints of a square spiral (legs 1,1,2,2,3,3,... * step) inside max_radius."""
	center = np.asarray(center, dtype = float)
	points = [center.copy()]
	pos = center.copy()
	leg = 1
	d = 0
	while True:
		for _ in range(2):
			nxt = pos + _DIRECTIONS[d % 4] * leg * step
			if np.max(np.abs(nxt - center)) > max_radius + 1e-9:
				# finish the outer ring by running this leg out to the boundary
				clipped = np.clip(nxt, center - max_radius, center + max_radius)
				if np.linalg.norm(clipped - pos) > 1e-6:
					points.append(clipped)
				return points
			points.append(nxt)
			pos = nxt
			d += 1
		leg += 1


class SpiralSearch:
	def __init__(self, center, step, max_radius, speed, gain = 1.0):
		self.center = np.asarray(center, dtype = float)
		self.step = step
		self.speed = speed
		self.gain = gain
		self.waypoints = square_spiral(self.center, step, max_radius)
		self.arrive_radius = max(0.15, 0.2 * step)
		self.index = 0
		self.loops = 0

	@property
	def current_waypoint(self):
		return self.waypoints[self.index]

	def velocity(self, position_xy):
		"""Horizontal velocity command (2,) toward the current waypoint."""
		delta = self.current_waypoint - np.asarray(position_xy, dtype = float)
		if np.linalg.norm(delta) < self.arrive_radius:
			self.index += 1
			if self.index >= len(self.waypoints):
				self.index = 0
				self.loops += 1
			delta = self.current_waypoint - np.asarray(position_xy, dtype = float)
		return saturate(self.gain * delta, self.speed)
