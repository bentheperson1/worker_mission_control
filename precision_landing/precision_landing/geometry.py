"""Rotation and frame utilities, written from scratch on top of numpy.

Conventions used everywhere in this package
	- World frame: PX4 local NED (x north, y east, z down), meters.
	- Body frame: PX4 FRD (x forward, y right, z down).
	- Camera frame: OpenCV optical (x right in image, y down in image, z out of the lens).
	- Tag frame: AprilTag convention (x right, y down, z into the tag), origin at the tag center.
	- Quaternions are (w, x, y, z), Hamilton, and rotate vectors from the child frame
	  into the parent frame (PX4 VehicleAttitude.q is body FRD -> NED).
	- R_a_b means "rotation taking vectors expressed in frame b into frame a".
"""
import math

import numpy as np


def skew(v):
	"""Cross-product matrix: skew(a) @ b == np.cross(a, b)."""
	x, y, z = v
	return np.array([
		[0.0, -z, y],
		[z, 0.0, -x],
		[-y, x, 0.0],
	])


def rot_x(a):
	c, s = math.cos(a), math.sin(a)
	return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
	c, s = math.cos(a), math.sin(a)
	return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
	c, s = math.cos(a), math.sin(a)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_rot(roll, pitch, yaw):
	"""Aerospace ZYX Euler angles -> R_world_body."""
	return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_to_euler(R):
	"""R_world_body -> (roll, pitch, yaw), ZYX convention."""
	pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
	roll = math.atan2(R[2, 1], R[2, 2])
	yaw = math.atan2(R[1, 0], R[0, 0])
	return roll, pitch, yaw


def yaw_from_rot(R):
	return math.atan2(R[1, 0], R[0, 0])


def quat_normalize(q):
	q = np.asarray(q, dtype = float)
	n = np.linalg.norm(q)
	if n < 1e-12:
		return np.array([1.0, 0.0, 0.0, 0.0])
	return q / n


def quat_to_rot(q):
	"""Unit quaternion (w, x, y, z) -> rotation matrix."""
	w, x, y, z = quat_normalize(q)
	return np.array([
		[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
		[2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
		[2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
	])


def rot_to_quat(R):
	"""Rotation matrix -> unit quaternion (w, x, y, z) with w >= 0 (Shepperd's method)."""
	tr = R[0, 0] + R[1, 1] + R[2, 2]
	if tr > 0.0:
		s = math.sqrt(tr + 1.0) * 2.0
		q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
	elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
		s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
		q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
	elif R[1, 1] > R[2, 2]:
		s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
		q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
	else:
		s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
		q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
	q = quat_normalize(q)
	return q if q[0] >= 0.0 else -q


def yaw_from_quat(q):
	return yaw_from_rot(quat_to_rot(q))


def quat_slerp(q0, q1, t):
	"""Spherical interpolation between unit quaternions, t in [0, 1]."""
	q0 = quat_normalize(q0)
	q1 = quat_normalize(q1)
	dot = float(np.dot(q0, q1))
	if dot < 0.0:  # take the short way around
		q1 = -q1
		dot = -dot
	if dot > 0.9995:  # nearly identical: normalized lerp is accurate and stable
		return quat_normalize(q0 + t * (q1 - q0))
	theta = math.acos(dot)
	s = math.sin(theta)
	return (math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1) / s


def exp_so3(w):
	"""Rodrigues' formula: axis-angle vector -> rotation matrix."""
	w = np.asarray(w, dtype = float)
	theta = float(np.linalg.norm(w))
	K = skew(w)
	if theta < 1e-9:
		return np.eye(3) + K
	return np.eye(3) + (math.sin(theta) / theta) * K + ((1.0 - math.cos(theta)) / (theta * theta)) * (K @ K)


def log_so3(R):
	"""Rotation matrix -> axis-angle vector (inverse of exp_so3)."""
	cos_theta = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
	theta = math.acos(cos_theta)
	if theta < 1e-9:
		return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
	if math.pi - theta < 1e-6:
		# near 180 deg the antisymmetric part vanishes; recover the axis from R + I
		M = (R + np.eye(3)) / 2.0
		axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
		i = int(np.argmax(axis))
		for j in range(3):
			if j != i and M[i, j] < 0.0:
				axis[j] = -axis[j]
		return theta * axis / np.linalg.norm(axis)
	return theta / (2.0 * math.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def orthonormalize(M):
	"""Closest proper rotation matrix to M in the Frobenius sense (via SVD)."""
	U, _, Vt = np.linalg.svd(M)
	R = U @ Vt
	if np.linalg.det(R) < 0.0:
		U[:, -1] = -U[:, -1]
		R = U @ Vt
	return R


def wrap_angle(a):
	"""Wrap an angle to [-pi, pi)."""
	return (a + math.pi) % (2.0 * math.pi) - math.pi


def interp_clamped(x, x0, x1, y0, y1):
	"""Linear map of x from [x0, x1] onto [y0, y1], clamped at the ends."""
	if x1 <= x0:
		return y1 if x >= x1 else y0
	u = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
	return y0 + u * (y1 - y0)
