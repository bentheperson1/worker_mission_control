"""6-DOF tag pose from the four detected corners (a from-scratch planar PnP).

Pipeline
	1. undistort the corner pixels into normalized image coordinates
	2. fit the plane-to-image homography with the normalized DLT
	3. decompose H = lambda * [r1 r2 t] into an initial rotation + translation
	4. refine (R, t) with Levenberg-Marquardt on the reprojection error

Only the detector (corner finding + ID decoding) comes from a library; everything
here is plain numpy.

Corner order follows the AprilRobotics detector (which apriltag_ros forwards
unchanged): counter-clockwise in the image, starting bottom-left of the tag:
	p0 = (-s/2, +s/2), p1 = (+s/2, +s/2), p2 = (+s/2, -s/2), p3 = (-s/2, -s/2)
in the tag frame (x right, y down, z into the tag). If a detector numbers the
corners differently the recovered *position* is unchanged (see tests); only the
tag's in-plane orientation would be rotated or mirrored.
"""
from dataclasses import dataclass
import math

import numpy as np

from .geometry import exp_so3, orthonormalize, skew


@dataclass
class TagPose:
	rotation: np.ndarray  # R_cam_tag
	translation: np.ndarray  # tag center expressed in the camera frame (m)
	reprojection_rms_px: float


def tag_corner_points(size):
	"""Tag corners in the tag frame, matching the detector's corner order."""
	h = size / 2.0
	return np.array([
		[-h, h, 0.0],
		[h, h, 0.0],
		[h, -h, 0.0],
		[-h, -h, 0.0],
	])


def _hartley_normalize(pts):
	"""Translate to the centroid and scale so the mean distance is sqrt(2)."""
	centroid = pts.mean(axis = 0)
	mean_dist = np.sqrt(((pts - centroid) ** 2).sum(axis = 1)).mean()
	s = math.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
	T = np.array([
		[s, 0.0, -s * centroid[0]],
		[0.0, s, -s * centroid[1]],
		[0.0, 0.0, 1.0],
	])
	homog = np.column_stack([pts, np.ones(len(pts))]) @ T.T
	return homog[:, :2], T


def homography_dlt(src, dst):
	"""Homography H with dst ~ H @ src for N >= 4 point pairs (normalized DLT)."""
	src = np.asarray(src, dtype = float)
	dst = np.asarray(dst, dtype = float)
	src_n, T_src = _hartley_normalize(src)
	dst_n, T_dst = _hartley_normalize(dst)
	A = np.zeros((2 * len(src), 9))
	for i, ((x, y), (u, v)) in enumerate(zip(src_n, dst_n)):
		A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
		A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
	_, _, Vt = np.linalg.svd(A)
	H_n = Vt[-1].reshape(3, 3)
	H = np.linalg.inv(T_dst) @ H_n @ T_src
	if abs(H[2, 2]) > 1e-12:
		H = H / H[2, 2]
	return H


def pose_from_homography(H):
	"""Decompose a tag-plane -> normalized-image homography into (R_cam_tag, t).

	For points on the tag plane (Z = 0): x ~ [r1 r2 t] [X Y 1]^T, so the columns
	of H are r1, r2 and t up to one common scale.
	"""
	h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
	lam = 2.0 / (np.linalg.norm(h1) + np.linalg.norm(h2))
	if h3[2] * lam < 0.0:  # the tag must be in front of the camera (t_z > 0)
		lam = -lam
	r1 = lam * h1
	r2 = lam * h2
	t = lam * h3
	R = orthonormalize(np.column_stack([r1, r2, np.cross(r1, r2)]))
	return R, t


def _project_normalized(R, t, obj):
	p = obj @ R.T + t
	return p[:, :2] / p[:, 2:3], p


def _residuals(R, t, obj, img, normal_prior, prior_weight):
	proj, p = _project_normalized(R, t, obj)
	r = (proj - img).ravel()
	if normal_prior is not None:
		r = np.concatenate([r, prior_weight * np.cross(R[:, 2], normal_prior)])
	return r, p


def refine_pose(R, t, obj, img, iterations = 15, normal_prior = None, prior_weight = 0.0):
	"""Levenberg-Marquardt on the normalized-image reprojection error.

	Rotation is updated multiplicatively (R <- exp(dw) R) so it always stays a
	valid rotation; translation additively.

	normal_prior (optional): expected direction of the tag's z axis in the camera
	frame. Adds the residual prior_weight * (z_tag x normal_prior), i.e. a soft
	"the pad is level" constraint (MAP instead of ML estimate).
	"""
	if normal_prior is not None:
		normal_prior = np.asarray(normal_prior, dtype = float)
	r, p = _residuals(R, t, obj, img, normal_prior, prior_weight)
	cost = float(r @ r)
	damping = 1e-3
	n = len(obj)
	rows = 2 * n + (3 if normal_prior is not None else 0)
	for _ in range(iterations):
		J = np.zeros((rows, 6))
		for i in range(n):
			x, y, z = p[i]
			Jp = np.array([
				[1.0 / z, 0.0, -x / (z * z)],
				[0.0, 1.0 / z, -y / (z * z)],
			])
			J[2 * i:2 * i + 2, :3] = Jp @ (-skew(p[i] - t))  # d(exp(dw) R X)/d(dw) = -[R X]x
			J[2 * i:2 * i + 2, 3:] = Jp
		if normal_prior is not None:
			# d(n x n0)/d(dw) with n <- n + dw x n  ==>  [n0]x [n]x
			J[2 * n:, :3] = prior_weight * (skew(normal_prior) @ skew(R[:, 2]))
		A = J.T @ J
		g = J.T @ r
		improved = False
		while damping < 1e8:
			try:
				delta = -np.linalg.solve(A + damping * (np.diag(np.diag(A)) + 1e-12 * np.eye(6)), g)
			except np.linalg.LinAlgError:
				damping *= 10.0
				continue
			R_new = exp_so3(delta[:3]) @ R
			t_new = t + delta[3:]
			r_new, p_new = _residuals(R_new, t_new, obj, img, normal_prior, prior_weight)
			if np.all(p_new[:, 2] > 0.0):
				cost_new = float(r_new @ r_new)
				if cost_new < cost:
					R, t, p, r, cost = R_new, t_new, p_new, r_new, cost_new
					damping = max(damping * 0.3, 1e-9)
					improved = True
					break
			damping *= 10.0
		if not improved or np.linalg.norm(delta) < 1e-10:
			break
	return R, t


def _quad_area(c):
	x, y = c[:, 0], c[:, 1]
	return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def estimate_tag_pose(corners_px, camera, tag_size, refine = True, min_area_px = 25.0, level_normal_cam = None, tilt_sigma_rad = 0.05, pixel_sigma = 1.0):
	"""Pose of one tag from its 4 corner pixels. Returns TagPose or None.

	level_normal_cam (optional): the world "down" direction expressed in the
	camera frame (known from vehicle attitude). If given, the pad is assumed
	level to within tilt_sigma_rad. Small tags seen nearly head-on have a weakly
	observable tilt; image noise then produces a spurious tilt, and because a
	tilted square looks smaller, the unconstrained fit is biased *closer*. The
	prior removes that bias and tightens the depth estimate considerably.
	"""
	corners = np.asarray(corners_px, dtype = float).reshape(-1, 2)
	if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
		return None
	if _quad_area(corners) < min_area_px:
		return None
	obj = tag_corner_points(tag_size)
	try:
		img = camera.undistort_points(corners)
		H = homography_dlt(obj[:, :2], img)
		R, t = pose_from_homography(H)
		if refine:
			R, t = refine_pose(R, t, obj, img)
			if level_normal_cam is not None and tilt_sigma_rad > 0.0:
				n0 = np.asarray(level_normal_cam, dtype = float)
				n0 = n0 / np.linalg.norm(n0)
				# take the sign from the free fit so either corner-winding convention works
				if np.dot(R[:, 2], n0) < 0.0:
					n0 = -n0
				weight = (pixel_sigma / camera.focal) / tilt_sigma_rad  # same units as image residuals
				R, t = refine_pose(R, t, obj, img, normal_prior = n0, prior_weight = weight)
	except np.linalg.LinAlgError:
		return None
	if not (np.all(np.isfinite(t)) and t[2] > 0.0):
		return None
	reproj = camera.project(obj @ R.T + t)
	rms = float(np.sqrt(np.mean(np.sum((reproj - corners) ** 2, axis = 1))))
	return TagPose(R, t, rms)
