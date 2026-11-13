"""Pinhole camera with Brown-Conrady (OpenCV "plumb_bob") lens distortion."""
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass
class CameraModel:
	width: int
	height: int
	fx: float
	fy: float
	cx: float
	cy: float
	# (k1, k2, p1, p2, k3) - same order as sensor_msgs/CameraInfo.d for plumb_bob
	dist: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0)

	def __post_init__(self):
		d = [float(v) for v in self.dist][:5]
		self.dist = tuple(d + [0.0] * (5 - len(d)))

	@classmethod
	def from_fov(cls, width, height, hfov_deg, dist = (0.0, 0.0, 0.0, 0.0, 0.0)):
		"""Ideal camera from a horizontal field of view (square pixels, centered)."""
		fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
		return cls(width, height, fx, fx, width / 2.0, height / 2.0, dist)

	@classmethod
	def from_camera_info(cls, k, d, width, height, p = None, rectified = False):
		"""Build from sensor_msgs/CameraInfo fields.

		If detections are computed on the *rectified* image, pass rectified=True:
		the projection matrix P is used and distortion is zero. Otherwise K and D
		describe the raw (distorted) image the detector saw.
		"""
		if rectified:
			if p is None or len(p) < 12:
				raise ValueError("rectified camera needs the 3x4 projection matrix p")
			return cls(int(width), int(height), float(p[0]), float(p[5]), float(p[2]), float(p[6]))
		return cls(int(width), int(height), float(k[0]), float(k[4]), float(k[2]), float(k[5]), tuple(d))

	@property
	def has_distortion(self):
		return any(abs(v) > 0.0 for v in self.dist)

	@property
	def hfov(self):
		return 2.0 * math.atan(self.width / (2.0 * self.fx))

	@property
	def vfov(self):
		return 2.0 * math.atan(self.height / (2.0 * self.fy))

	@property
	def focal(self):
		return 0.5 * (self.fx + self.fy)

	def distort(self, xy):
		"""Undistorted normalized coords (N, 2) -> distorted normalized coords."""
		k1, k2, p1, p2, k3 = self.dist
		x = xy[:, 0]
		y = xy[:, 1]
		r2 = x * x + y * y
		radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
		xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
		yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
		return np.column_stack([xd, yd])

	def project(self, points_cam):
		"""3D points in the camera frame (N, 3) -> pixel coordinates (N, 2)."""
		p = np.atleast_2d(np.asarray(points_cam, dtype = float))
		xy = p[:, :2] / p[:, 2:3]
		if self.has_distortion:
			xy = self.distort(xy)
		return np.column_stack([self.fx * xy[:, 0] + self.cx, self.fy * xy[:, 1] + self.cy])

	def undistort_points(self, pixels, iterations = 20):
		"""Pixel coordinates (N, 2) -> undistorted normalized image coords (N, 2).

		Inverts the distortion model by fixed-point iteration (the same scheme
		OpenCV's undistortPoints uses); converges in a few steps for typical lenses.
		"""
		px = np.atleast_2d(np.asarray(pixels, dtype = float))
		xd = (px[:, 0] - self.cx) / self.fx
		yd = (px[:, 1] - self.cy) / self.fy
		if not self.has_distortion:
			return np.column_stack([xd, yd])
		k1, k2, p1, p2, k3 = self.dist
		x = xd.copy()
		y = yd.copy()
		for _ in range(iterations):
			r2 = x * x + y * y
			radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
			dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
			dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
			x = (xd - dx) / radial
			y = (yd - dy) / radial
		return np.column_stack([x, y])

	def contains(self, pixels, margin = 0.0):
		"""True for each pixel that lies inside the image with the given margin."""
		px = np.atleast_2d(pixels)
		return (
			(px[:, 0] >= margin) & (px[:, 0] <= self.width - 1 - margin)
			& (px[:, 1] >= margin) & (px[:, 1] <= self.height - 1 - margin)
		)
