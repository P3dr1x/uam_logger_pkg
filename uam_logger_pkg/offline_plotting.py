"""Offline plotting utilities for uam_logger_pkg CSV exports.

This script loads the CSV produced by `uam_logger_node` and generates plots for:
- Desired vs real end-effector 3D trajectory
- Pose tracking error norm over time (nearest-timestamp matching)
- Drone position and attitude (3x2 grid) (odometry in sim, mocap pose in real)
- Drone RMS attitude disturbance (RMS of roll/pitch/yaw)
- Drone displacement norms w.r.t. initial pose
- PX4 SensorCombined accelerometer + gyroscope data (if present)
- Real drone twist components from /real_t960a_twist (if present)

The CSV format is the one produced by `UamLoggerNode._save_experiment()`:
columns: `t`, `topic`, plus topic-specific fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PoseSeries:
	"""Time series for poses."""

	t: List[float]
	px: List[float]
	py: List[float]
	pz: List[float]
	qx: List[float]
	qy: List[float]
	qz: List[float]
	qw: List[float]


@dataclass(frozen=True)
class OdomSeries:
	"""Time series for odometry pose (position + RPY)."""

	t: List[float]
	x: List[float]
	y: List[float]
	z: List[float]
	roll: List[float]
	pitch: List[float]
	yaw: List[float]


@dataclass(frozen=True)
class AngularVelSeries:
	"""Time series for angular velocity (from odometry twist.angular)."""

	t: List[float]
	wx: List[float]
	wy: List[float]
	wz: List[float]


@dataclass(frozen=True)
class TwistSeries:
	"""Time series for Twist components."""

	t: List[float]
	vx: List[float]
	vy: List[float]
	vz: List[float]
	wx: List[float]
	wy: List[float]
	wz: List[float]


@dataclass(frozen=True)
class AccelSeries:
	"""Time series for linear accelerations."""

	t: List[float]
	ax: List[float]
	ay: List[float]
	az: List[float]


@dataclass(frozen=True)
class GyroSeries:
	"""Time series for angular rates (body gyro)."""

	t: List[float]
	gx: List[float]
	gy: List[float]
	gz: List[float]


@dataclass(frozen=True)
class ExperimentData:
	"""All time-series extracted from a single CSV file."""

	label: str
	csv_path: Path

	desired_pose: Optional[PoseSeries]
	real_pose: Optional[PoseSeries]

	# Base drone pose used for plotting: odometry in sim, mocap pose in real.
	base_pose_topic: str
	odom: Optional[OdomSeries]

	accel: Optional[AccelSeries]
	gyro: Optional[GyroSeries]

	# Controller parameters (metadata) if available in the CSV.
	controller_params: Optional[Dict[str, object]]


def _clamp(x: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, x))


def _wrap_to_pi(a: float) -> float:
	"""Wrap angle to [-pi, pi]."""
	while a > math.pi:
		a -= 2.0 * math.pi
	while a < -math.pi:
		a += 2.0 * math.pi
	return a


def _quat_normalize(
	qx: float, qy: float, qz: float, qw: float
) -> Tuple[float, float, float, float]:
	n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
	if n <= 0.0:
		return 0.0, 0.0, 0.0, 1.0
	return qx / n, qy / n, qz / n, qw / n


def _quat_conj(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float, float]:
	return -qx, -qy, -qz, qw


def _quat_mul(
	ax: float, ay: float, az: float, aw: float, bx: float, by: float, bz: float, bw: float
) -> Tuple[float, float, float, float]:
	"""Hamilton product q = a ⊗ b."""
	x = aw * bx + ax * bw + ay * bz - az * by
	y = aw * by - ax * bz + ay * bw + az * bx
	z = aw * bz + ax * by - ay * bx + az * bw
	w = aw * bw - ax * bx - ay * by - az * bz
	return x, y, z, w


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
	cr = math.cos(roll * 0.5)
	sr = math.sin(roll * 0.5)
	cp = math.cos(pitch * 0.5)
	sp = math.sin(pitch * 0.5)
	cy = math.cos(yaw * 0.5)
	sy = math.sin(yaw * 0.5)

	qw = cr * cp * cy + sr * sp * sy
	qx = sr * cp * cy - cr * sp * sy
	qy = cr * sp * cy + sr * cp * sy
	qz = cr * cp * sy - sr * sp * cy
	return _quat_normalize(qx, qy, qz, qw)


def _quat_angle(qx: float, qy: float, qz: float, qw: float) -> float:
	"""Return rotation angle in [0, pi] for a unit quaternion."""
	qx, qy, qz, qw = _quat_normalize(qx, qy, qz, qw)
	qw = abs(qw)  # double cover
	qw = _clamp(qw, -1.0, 1.0)
	return 2.0 * math.acos(qw)


def _quat_to_axis_angle_vec(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
	"""Convert quaternion to axis-angle error vector e = axis * angle.

	The returned vector is the logarithm map representation on SO(3): e in R^3.
	Its norm is the rotation angle in [0, pi].
	"""
	qx, qy, qz, qw = _quat_normalize(qx, qy, qz, qw)
	# Use the shortest rotation (double cover).
	if qw < 0.0:
		qx, qy, qz, qw = -qx, -qy, -qz, -qw

	vn = math.sqrt(qx * qx + qy * qy + qz * qz)
	if vn < 1e-12:
		return 0.0, 0.0, 0.0

	# Stable angle computation.
	angle = 2.0 * math.atan2(vn, qw)
	ax = qx / vn
	ay = qy / vn
	az = qz / vn
	return ax * angle, ay * angle, az * angle


def _rad_to_deg(a: float) -> float:
	return a * (180.0 / math.pi)


def _relative_rotation_angle(
	qx_a: float,
	qy_a: float,
	qz_a: float,
	qw_a: float,
	qx_b: float,
	qy_b: float,
	qz_b: float,
	qw_b: float,
) -> float:
	"""Angle of q_err = q_a^{-1} ⊗ q_b."""
	qx_a, qy_a, qz_a, qw_a = _quat_normalize(qx_a, qy_a, qz_a, qw_a)
	qx_b, qy_b, qz_b, qw_b = _quat_normalize(qx_b, qy_b, qz_b, qw_b)
	ix, iy, iz, iw = _quat_conj(qx_a, qy_a, qz_a, qw_a)
	ex, ey, ez, ew = _quat_mul(ix, iy, iz, iw, qx_b, qy_b, qz_b, qw_b)
	return _quat_angle(ex, ey, ez, ew)


def _relative_rotation_axis_angle_vec(
	qx_a: float,
	qy_a: float,
	qz_a: float,
	qw_a: float,
	qx_b: float,
	qy_b: float,
	qz_b: float,
	qw_b: float,
) -> Tuple[float, float, float]:
	"""Axis-angle error vector of q_err = q_a^{-1} ⊗ q_b."""
	qx_a, qy_a, qz_a, qw_a = _quat_normalize(qx_a, qy_a, qz_a, qw_a)
	qx_b, qy_b, qz_b, qw_b = _quat_normalize(qx_b, qy_b, qz_b, qw_b)
	ix, iy, iz, iw = _quat_conj(qx_a, qy_a, qz_a, qw_a)
	ex, ey, ez, ew = _quat_mul(ix, iy, iz, iw, qx_b, qy_b, qz_b, qw_b)
	return _quat_to_axis_angle_vec(ex, ey, ez, ew)


def _nearest_indices(t_ref: List[float], t_query: List[float]) -> List[int]:
	"""For each t in t_query, return index of nearest t_ref sample.

	Assumes both lists are sorted.
	"""
	if not t_ref:
		return []

	indices: List[int] = []
	j = 0
	for tq in t_query:
		while j + 1 < len(t_ref) and t_ref[j + 1] < tq:
			j += 1
		# Candidate j and j+1
		if j + 1 < len(t_ref):
			if abs(t_ref[j + 1] - tq) < abs(t_ref[j] - tq):
				indices.append(j + 1)
			else:
				indices.append(j)
		else:
			indices.append(j)
	return indices


def _compute_pose_tracking_errors(
	desired: PoseSeries, real: PoseSeries
) -> Tuple[List[float], List[float], List[float]]:
	"""Compute position/orientation tracking errors using nearest-timestamp matching.

	The returned timestamps are the desired trajectory timestamps.

	Returns:
		err_t: timestamps (seconds)
		pos_err_norm: ||e_p|| [m]
		ori_err_norm_deg: ||e_R|| [deg], using axis-angle log map norm
	"""
	idx = _nearest_indices(real.t, desired.t)
	if not idx:
		return [], [], []

	err_t: List[float] = []
	pos_err_norm: List[float] = []
	ori_err_norm_deg: List[float] = []
	for k, j in enumerate(idx):
		dx = desired.px[k] - real.px[j]
		dy = desired.py[k] - real.py[j]
		dz = desired.pz[k] - real.pz[j]
		pos_err = math.sqrt(dx * dx + dy * dy + dz * dz)
		ex, ey, ez = _relative_rotation_axis_angle_vec(
			desired.qx[k],
			desired.qy[k],
			desired.qz[k],
			desired.qw[k],
			real.qx[j],
			real.qy[j],
			real.qz[j],
			real.qw[j],
		)
		ori_err = math.sqrt(ex * ex + ey * ey + ez * ez)
		ori_err_deg = _rad_to_deg(ori_err)
		err_t.append(desired.t[k])
		pos_err_norm.append(pos_err)
		ori_err_norm_deg.append(ori_err_deg)

	return err_t, pos_err_norm, ori_err_norm_deg


def _controller_param_order() -> List[str]:
	"""Preferred row order for controller param comparison tables."""
	return [
		"k_com_vel",
		"k_damp",
		"k_err_pos_",
		"k_err_vel_",
		"redundant",
		"w_com",
		"w_damp",
		"w_dyn",
		"w_kin",
		"w_mom",
		"k_err",
	]


def _extract_controller_params(groups: Dict[str, List[Dict[str, str]]]) -> Optional[Dict[str, object]]:
	rows = groups.get("__metadata__/controller_params", [])
	if not rows:
		return None

	r = rows[0]
	raw = (r.get("controller_params_json") or "").strip()
	if not raw:
		return None
	try:
		obj = json.loads(raw)
		if isinstance(obj, dict):
			return obj  # type: ignore[return-value]
	except Exception:
		return None
	return None


def _format_controller_params_markdown(experiments: Sequence[ExperimentData]) -> Optional[str]:
	"""Return a Markdown table (rows=params, cols=experiments) or None if empty."""
	all_params = set()
	for e in experiments:
		if e.controller_params is not None:
			all_params.update(e.controller_params.keys())

	if not all_params:
		return None

	ordered = []
	for p in _controller_param_order():
		if p in all_params:
			ordered.append(p)
	for p in sorted(all_params):
		if p not in ordered:
			ordered.append(p)

	col_labels = [e.label for e in experiments]
	# Header
	lines: List[str] = []
	lines.append("| param | " + " | ".join(col_labels) + " |")
	lines.append("|---|" + "|".join(["---"] * len(col_labels)) + "|")

	for p in ordered:
		row_vals: List[str] = []
		for e in experiments:
			val = None
			if e.controller_params is not None:
				val = e.controller_params.get(p)
			row_vals.append("" if val is None else str(val))
		lines.append("| " + p + " | " + " | ".join(row_vals) + " |")

	return "\n".join(lines)


def _plot_controller_params_table(experiments: Sequence[ExperimentData]) -> None:
	"""Show a separate window with controller parameters table."""
	import matplotlib.pyplot as plt

	all_params = set()
	for e in experiments:
		if e.controller_params is not None:
			all_params.update(e.controller_params.keys())
	if not all_params:
		return

	ordered: List[str] = []
	for p in _controller_param_order():
		if p in all_params:
			ordered.append(p)
	for p in sorted(all_params):
		if p not in ordered:
			ordered.append(p)

	col_labels = [e.label for e in experiments]
	cell_text: List[List[str]] = []
	for p in ordered:
		row: List[str] = []
		for e in experiments:
			val = None
			if e.controller_params is not None:
				val = e.controller_params.get(p)
			row.append("" if val is None else str(val))
		cell_text.append(row)

	# Heuristic sizing for readability.
	width = max(8.0, 1.8 * len(experiments))
	height = max(3.0, 0.35 * len(ordered) + 1.5)
	fig, ax = plt.subplots(figsize=(width, height))
	fig.suptitle("Controller parameters (from CSV metadata)")
	ax.axis("off")

	table = ax.table(
		cellText=cell_text,
		rowLabels=ordered,
		colLabels=col_labels,
		loc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(9)
	table.scale(1.0, 1.2)


def _load_experiment(csv_path: Path, label: str) -> ExperimentData:
	groups = _load_csv_grouped(csv_path)

	topic_desired_pose = "/desired_ee_global_pose"
	topic_real_pose = "/ee_world_pose"
	topic_odom = "/model/t960a_0/odometry"
	topic_mocap_pose = "/t960a/pose"
	topic_sensor = "/fmu/out/sensor_combined"

	desired_pose = _extract_pose_series(groups.get(topic_desired_pose, []))
	real_pose = _extract_pose_series(groups.get(topic_real_pose, []))

	base_pose_topic = topic_mocap_pose if topic_mocap_pose in groups else topic_odom
	odom = _extract_odom_series(groups.get(base_pose_topic, []))

	accel = _extract_accel_series(groups.get(topic_sensor, []))
	gyro = _extract_gyro_series(groups.get(topic_sensor, []))
	controller_params = _extract_controller_params(groups)

	return ExperimentData(
		label=label,
		csv_path=csv_path,
		desired_pose=desired_pose,
		real_pose=real_pose,
		base_pose_topic=base_pose_topic,
		odom=odom,
		accel=accel,
		gyro=gyro,
		controller_params=controller_params,
	)


def _load_csv_grouped(csv_path: Path) -> Dict[str, List[Dict[str, str]]]:
	"""Load CSV and group rows by `topic`."""
	groups: Dict[str, List[Dict[str, str]]] = {}
	with csv_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			topic = (row.get("topic") or "").strip()
			if not topic:
				continue
			groups.setdefault(topic, []).append(row)
	return groups


def _get_float(row: Dict[str, str], key: str) -> Optional[float]:
	v = row.get(key)
	if v is None:
		return None
	v = v.strip()
	if v == "":
		return None
	try:
		return float(v)
	except ValueError:
		return None


def _extract_pose_series(rows: List[Dict[str, str]]) -> Optional[PoseSeries]:
	t: List[float] = []
	px: List[float] = []
	py: List[float] = []
	pz: List[float] = []
	qx: List[float] = []
	qy: List[float] = []
	qz: List[float] = []
	qw: List[float] = []

	for r in rows:
		t_i = _get_float(r, "t")
		px_i = _get_float(r, "px")
		py_i = _get_float(r, "py")
		pz_i = _get_float(r, "pz")
		qx_i = _get_float(r, "qx")
		qy_i = _get_float(r, "qy")
		qz_i = _get_float(r, "qz")
		qw_i = _get_float(r, "qw")
		if None in (t_i, px_i, py_i, pz_i, qx_i, qy_i, qz_i, qw_i):
			continue
		t.append(float(t_i))
		px.append(float(px_i))
		py.append(float(py_i))
		pz.append(float(pz_i))
		qx.append(float(qx_i))
		qy.append(float(qy_i))
		qz.append(float(qz_i))
		qw.append(float(qw_i))

	if not t:
		return None
	return PoseSeries(t=t, px=px, py=py, pz=pz, qx=qx, qy=qy, qz=qz, qw=qw)


def _extract_odom_series(rows: List[Dict[str, str]]) -> Optional[OdomSeries]:
	t: List[float] = []
	x: List[float] = []
	y: List[float] = []
	z: List[float] = []
	roll: List[float] = []
	pitch: List[float] = []
	yaw: List[float] = []

	for r in rows:
		t_i = _get_float(r, "t")
		x_i = _get_float(r, "x")
		y_i = _get_float(r, "y")
		z_i = _get_float(r, "z")
		r_i = _get_float(r, "roll")
		p_i = _get_float(r, "pitch")
		y_iw = _get_float(r, "yaw")
		if None in (t_i, x_i, y_i, z_i, r_i, p_i, y_iw):
			continue
		t.append(float(t_i))
		x.append(float(x_i))
		y.append(float(y_i))
		z.append(float(z_i))
		roll.append(float(r_i))
		pitch.append(float(p_i))
		yaw.append(float(y_iw))

	if not t:
		return None
	return OdomSeries(t=t, x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)


def _extract_odom_angular_velocity_series(rows: List[Dict[str, str]]) -> Optional[AngularVelSeries]:
	"""Extract angular velocity (wx/wy/wz) time series from Odometry rows.

	Expected CSV columns are wx/wy/wz (rad/s), as saved by uam_logger_node.
	"""
	t: List[float] = []
	wx: List[float] = []
	wy: List[float] = []
	wz: List[float] = []

	for r in rows:
		t_i = _get_float(r, "t")
		wx_i = _get_float(r, "wx")
		wy_i = _get_float(r, "wy")
		wz_i = _get_float(r, "wz")
		if None in (t_i, wx_i, wy_i, wz_i):
			continue
		t.append(float(t_i))
		wx.append(float(wx_i))
		wy.append(float(wy_i))
		wz.append(float(wz_i))

	if not t:
		return None
	return AngularVelSeries(t=t, wx=wx, wy=wy, wz=wz)


def _extract_accel_series(rows: List[Dict[str, str]]) -> Optional[AccelSeries]:
	t: List[float] = []
	ax: List[float] = []
	ay: List[float] = []
	az: List[float] = []

	for r in rows:
		t_i = _get_float(r, "t")
		ax_i = _get_float(r, "ax")
		ay_i = _get_float(r, "ay")
		az_i = _get_float(r, "az")
		if None in (t_i, ax_i, ay_i, az_i):
			continue
		t.append(float(t_i))
		ax.append(float(ax_i))
		ay.append(float(ay_i))
		az.append(float(az_i))

	if not t:
		return None
	return AccelSeries(t=t, ax=ax, ay=ay, az=az)


def _extract_gyro_series(rows: List[Dict[str, str]]) -> Optional[GyroSeries]:
	"""Extract gyro (angular velocity) time series from SensorCombined rows.

	Expected CSV columns are gx/gy/gz (rad/s), as saved by uam_logger_node.
	"""
	t: List[float] = []
	gx: List[float] = []
	gy: List[float] = []
	gz: List[float] = []

	for r in rows:
		t_i = _get_float(r, "t")
		gx_i = _get_float(r, "gx")
		gy_i = _get_float(r, "gy")
		gz_i = _get_float(r, "gz")
		# Backward/alternate naming fallback (if user has older CSVs)
		if gx_i is None and gy_i is None and gz_i is None:
			gx_i = _get_float(r, "wx")
			gy_i = _get_float(r, "wy")
			gz_i = _get_float(r, "wz")
		if None in (t_i, gx_i, gy_i, gz_i):
			continue
		t.append(float(t_i))
		gx.append(float(gx_i))
		gy.append(float(gy_i))
		gz.append(float(gz_i))

	if not t:
		return None
	return GyroSeries(t=t, gx=gx, gy=gy, gz=gz)


def _extract_twist_series(rows: List[Dict[str, str]]) -> Optional[TwistSeries]:
	"""Extract vx/vy/vz + wx/wy/wz time series from Twist-like rows."""
	t: List[float] = []
	vx: List[float] = []
	vy: List[float] = []
	vz: List[float] = []
	wx: List[float] = []
	wy: List[float] = []
	wz: List[float] = []

	for r in rows:
		t_i = _get_float(r, "t")
		vx_i = _get_float(r, "vx")
		vy_i = _get_float(r, "vy")
		vz_i = _get_float(r, "vz")
		wx_i = _get_float(r, "wx")
		wy_i = _get_float(r, "wy")
		wz_i = _get_float(r, "wz")
		if None in (t_i, vx_i, vy_i, vz_i, wx_i, wy_i, wz_i):
			continue
		t.append(float(t_i))
		vx.append(float(vx_i))
		vy.append(float(vy_i))
		vz.append(float(vz_i))
		wx.append(float(wx_i))
		wy.append(float(wy_i))
		wz.append(float(wz_i))

	if not t:
		return None
	return TwistSeries(t=t, vx=vx, vy=vy, vz=vz, wx=wx, wy=wy, wz=wz)


def _plot_ee_trajectories(desired: PoseSeries, real: PoseSeries, title_prefix: str) -> None:
	import matplotlib.pyplot as plt

	def _compute_alignment_yaw(px: List[float], py: List[float]) -> float:
		"""Return yaw angle (rad) to align the main XY direction with +X.

		The rotation is about +Z and is chosen so that the desired trajectory lies
		as parallel as possible to the X-Z plane (i.e., minimizes Y variance).
		"""
		if len(px) < 2:
			return 0.0

		mx = sum(px) / len(px)
		my = sum(py) / len(py)
		sxx = 0.0
		syy = 0.0
		sxy = 0.0
		for x, y in zip(px, py):
			dx = x - mx
			dy = y - my
			sxx += dx * dx
			syy += dy * dy
			sxy += dx * dy

		if sxx == 0.0 and syy == 0.0:
			return 0.0

		alpha = 0.5 * math.atan2(2.0 * sxy, (sxx - syy))
		# Rotate by -alpha so the principal direction aligns with +X.
		return -alpha

	def _rotate_about_z(x: float, y: float, yaw: float) -> Tuple[float, float]:
		c = math.cos(yaw)
		s = math.sin(yaw)
		xr = c * x - s * y
		yr = s * x + c * y
		return xr, yr

	yaw_align = _compute_alignment_yaw(desired.px, desired.py)

	dx_r: List[float] = []
	dy_r: List[float] = []
	dz_r: List[float] = list(desired.pz)

	rx_r: List[float] = []
	ry_r: List[float] = []
	rz_r: List[float] = list(real.pz)

	for x, y in zip(desired.px, desired.py):
		xr, yr = _rotate_about_z(x, y, yaw_align)
		dx_r.append(xr)
		dy_r.append(yr)
	for x, y in zip(real.px, real.py):
		xr, yr = _rotate_about_z(x, y, yaw_align)
		rx_r.append(xr)
		ry_r.append(yr)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.plot(
		dx_r,
		dy_r,
		dz_r,
		label="desired",
	)
	ax.plot(rx_r, ry_r, rz_r, label="real")

	# Mark start/end of the effective trajectory
	if rx_r and ry_r and rz_r:
		ax.scatter(
			rx_r[0],
			ry_r[0],
			rz_r[0],
			color="green",
			marker="o",
			s=40,
			label="real start",
		)
		ax.scatter(
			rx_r[-1],
			ry_r[-1],
			rz_r[-1],
			color="black",
			marker="o",
			s=40,
			label="real end",
		)
	ax.set_title(f"{title_prefix} - End-effector trajectory (yaw-aligned)")
	ax.set_xlabel("x [m]")
	ax.set_ylabel("y [m]")
	ax.set_zlabel("z [m]")
	ax.legend()

	# Widen the perpendicular axis (Y) range so off-plane deviations look smaller.
	perp_scale = 10.0
	y_all = dy_r + ry_r
	if y_all:
		y0 = sum(y_all) / len(y_all)
		max_dev = max(abs(y - y0) for y in y_all)
		if max_dev <= 1e-9:
			max_dev = 0.01
		ax.set_ylim(y0 - perp_scale * max_dev, y0 + perp_scale * max_dev)



def _plot_pose_error_norm(desired: PoseSeries, real: PoseSeries, title_prefix: str) -> None:
	import matplotlib.pyplot as plt

	err_t, pos_err_norm, ori_err_norm_deg = _compute_pose_tracking_errors(desired, real)
	if not err_t:
		return

	fig, axs = plt.subplots(2, 1, sharex=True)
	fig.suptitle(f"{title_prefix} - EE Pose tracking errors")

	axs[0].plot(err_t, pos_err_norm, color="red")
	axs[0].grid(True)
	axs[0].set_ylabel(r"$\|\|e_p\|\|\;[m]$")

	axs[1].plot(err_t, ori_err_norm_deg, color="red")
	axs[1].grid(True)
	axs[1].set_xlabel("t [s] (desired timestamps)")
	axs[1].set_ylabel(r"$\|\|e_R\|\|\;[°]$")


def _plot_ee_trajectories_comparison(experiments: Sequence[ExperimentData]) -> None:
	"""Plot desired vs real EE trajectory for multiple experiments.

	For each experiment, desired is translated so that it starts at (0,0,0), and
	the same translation is applied to the real EE trajectory. This matches the
	comparison strategy described in logger_instructions.md.
	"""
	import matplotlib.pyplot as plt

	def _vec_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
		ax, ay, az = a
		bx, by, bz = b
		return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

	def _vec_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

	def _vec_norm(a: Tuple[float, float, float]) -> float:
		return math.sqrt(_vec_dot(a, a))

	def _vec_scale(a: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
		return (a[0] * s, a[1] * s, a[2] * s)

	def _vec_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
		return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

	def _mat_mul_vec(m: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
		return (
			m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
			m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
			m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
		)

	def _rotation_matrix_from_axis_angle(axis: Tuple[float, float, float], angle: float) -> List[List[float]]:
		"""Rodrigues' rotation formula."""
		ax, ay, az = axis
		c = math.cos(angle)
		s = math.sin(angle)
		v = 1.0 - c
		return [
			[c + ax * ax * v, ax * ay * v - az * s, ax * az * v + ay * s],
			[ay * ax * v + az * s, c + ay * ay * v, ay * az * v - ax * s],
			[az * ax * v - ay * s, az * ay * v + ax * s, c + az * az * v],
		]

	def _estimate_plane_normal(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
		"""Estimate trajectory plane normal via accumulated triangle areas.

		This is robust for planar loops (circles/rectangles) and avoids heavy deps.
		"""
		if len(points) < 3:
			return (0.0, 1.0, 0.0)
		p0 = points[0]
		nx = ny = nz = 0.0
		for i in range(1, len(points) - 1):
			v1 = _vec_sub(points[i], p0)
			v2 = _vec_sub(points[i + 1], p0)
			cx, cy, cz = _vec_cross(v1, v2)
			nx += cx
			ny += cy
			nz += cz
		n = (nx, ny, nz)
		nn = _vec_norm(n)
		if nn < 1e-12:
			return (0.0, 1.0, 0.0)
		return _vec_scale(n, 1.0 / nn)

	def _rotation_to_align_vector(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> List[List[float]]:
		"""Return rotation matrix R such that R*a is aligned with b."""
		an = _vec_norm(a)
		bn = _vec_norm(b)
		if an < 1e-12 or bn < 1e-12:
			return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
		a_u = _vec_scale(a, 1.0 / an)
		b_u = _vec_scale(b, 1.0 / bn)
		# Force deterministic sign when both are almost collinear.
		d = _clamp(_vec_dot(a_u, b_u), -1.0, 1.0)
		if d > 1.0 - 1e-10:
			return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
		if d < -1.0 + 1e-10:
			# 180deg rotation around X maps -Y to +Y, and is stable.
			return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]

		axis = _vec_cross(a_u, b_u)
		axis_n = _vec_norm(axis)
		if axis_n < 1e-12:
			return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
		axis_u = _vec_scale(axis, 1.0 / axis_n)
		angle = math.atan2(axis_n, d)
		return _rotation_matrix_from_axis_angle(axis_u, angle)

	def _rotate_about_y(x: float, y: float, z: float, theta: float) -> Tuple[float, float, float]:
		c = math.cos(theta)
		s = math.sin(theta)
		xr = c * x + s * z
		zr = -s * x + c * z
		return xr, y, zr

	def _estimate_rot_y_to_match_reference(
		ref_x: List[float],
		ref_z: List[float],
		x: List[float],
		z: List[float],
	) -> float:
		"""Return theta (rad) to best align (x,z) to (ref_x,ref_z) by rot about +Y.

		All trajectories are assumed translated so they start at the origin.
		"""
		m = min(len(ref_x), len(ref_z), len(x), len(z))
		if m < 2:
			return 0.0

		max_points = 300
		step = max(1, m // max_points)

		a = 0.0
		b = 0.0
		energy_ref = 0.0
		energy_cur = 0.0
		for i in range(0, m, step):
			cx = x[i]
			cz = z[i]
			rx = ref_x[i]
			rz = ref_z[i]
			# maximize c*a + s*b where dot after rotY is:
			# v · (R_y(theta) u) = c*(rx*cx + rz*cz) + s*(rx*cz - rz*cx)
			a += rx * cx + rz * cz
			b += rx * cz - rz * cx
			energy_ref += rx * rx + rz * rz
			energy_cur += cx * cx + cz * cz
		if energy_ref < 1e-12 or energy_cur < 1e-12:
			return 0.0
		return math.atan2(b, a)

	usable = [e for e in experiments if e.desired_pose is not None and e.real_pose is not None]
	if not usable:
		return

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	fig.suptitle(
		"End-effector trajectories (comparison, translated + plane-aligned desired)"
	)

	# Build reference desired trajectory in the plotting frame:
	# - translated so start at origin
	# - rotated so that its best-fit plane becomes the X-Z plane (y=0)
	# - projected on X-Z plane for visualization
	ref_exp = usable[0]
	ref_desired = ref_exp.desired_pose
	assert ref_desired is not None
	ref_x0, ref_y0, ref_z0 = ref_desired.px[0], ref_desired.py[0], ref_desired.pz[0]
	ref_pts = [(x - ref_x0, y - ref_y0, z - ref_z0) for x, y, z in zip(ref_desired.px, ref_desired.py, ref_desired.pz)]
	# Ensure consistent normal sign (prefer +Y)
	ref_n = _estimate_plane_normal(ref_pts)
	if ref_n[1] < 0.0:
		ref_n = _vec_scale(ref_n, -1.0)
	ref_R = _rotation_to_align_vector(ref_n, (0.0, 1.0, 0.0))
	ref_rot = [_mat_mul_vec(ref_R, p) for p in ref_pts]
	ref_dx = [p[0] for p in ref_rot]
	ref_dz = [p[2] for p in ref_rot]

	colors = plt.get_cmap("tab10")
	x_all: List[float] = []
	y_all: List[float] = []
	z_all: List[float] = []
	for i, exp in enumerate(usable):
		desired = exp.desired_pose
		real = exp.real_pose
		assert desired is not None
		assert real is not None

		x0, y0, z0 = desired.px[0], desired.py[0], desired.pz[0]
		pts_d = [(x - x0, y - y0, z - z0) for x, y, z in zip(desired.px, desired.py, desired.pz)]
		pts_r = [(x - x0, y - y0, z - z0) for x, y, z in zip(real.px, real.py, real.pz)]

		# Rotate so the desired trajectory plane becomes X-Z (y=0).
		n = _estimate_plane_normal(pts_d)
		if n[1] < 0.0:
			n = _vec_scale(n, -1.0)
		R = _rotation_to_align_vector(n, (0.0, 1.0, 0.0))
		pts_d = [_mat_mul_vec(R, p) for p in pts_d]
		pts_r = [_mat_mul_vec(R, p) for p in pts_r]

		dx = [p[0] for p in pts_d]
		dy = [p[1] for p in pts_d]
		dz = [p[2] for p in pts_d]
		rx = [p[0] for p in pts_r]
		ry_ = [p[1] for p in pts_r]
		rz = [p[2] for p in pts_r]

		# Further align in-plane (rotation about +Y) to overlap identical desired trajectories.
		theta = 0.0 if exp is ref_exp else _estimate_rot_y_to_match_reference(ref_dx, ref_dz, dx, dz)
		if theta != 0.0:
			dx_r: List[float] = []
			dz_r: List[float] = []
			rx_r: List[float] = []
			ry_r: List[float] = []
			rz_r: List[float] = []
			for xx, yy, zz in zip(dx, dy, dz):
				xr, yr, zr = _rotate_about_y(xx, yy, zz, theta)
				dx_r.append(xr)
				dz_r.append(zr)
			for xx, yy, zz in zip(rx, ry_, rz):
				xr, yr, zr = _rotate_about_y(xx, yy, zz, theta)
				rx_r.append(xr)
				ry_r.append(yr)
				rz_r.append(zr)
			dx, dz = dx_r, dz_r
			rx, ry_, rz = rx_r, ry_r, rz_r
			# dy is already close to 0; keep it for completeness

		# Force desired to lie exactly on X-Z plane for visualization.
		dy = [0.0 for _ in dx]

		color = colors(i % 10)
		ax.plot(dx, dy, dz, linestyle="--", color="lime", alpha=0.9, label=f"{exp.label} desired")
		ax.plot(rx, ry_, rz, linestyle="-", color=color, alpha=1.0, label=f"{exp.label} real")

		y_all.extend(dy)
		y_all.extend(ry_)
		x_all.extend(dx)
		x_all.extend(rx)
		z_all.extend(dz)
		z_all.extend(rz)

		if rx and ry_ and rz:
			ax.scatter(rx[0], ry_[0], rz[0], color=color, marker="o", s=25)
			ax.scatter(rx[-1], ry_[-1], rz[-1], color=color, marker="x", s=35)

	ax.set_xlabel("x [m]")
	ax.set_ylabel("y [m]")
	ax.set_zlabel("z [m]")
	ax.legend(loc="best")

	# Widen the perpendicular axis (Y) range so off-plane deviations look smaller.
	perp_scale = 10.0
	if y_all:
		y0 = sum(y_all) / len(y_all)
		max_dev = max(abs(y - y0) for y in y_all)
		if max_dev <= 1e-9:
			max_dev = 0.01
		ax.set_ylim(y0 - perp_scale * max_dev, y0 + perp_scale * max_dev)

	# In 3D, Matplotlib does not use an equal aspect ratio by default.
	# Enforce equal scaling on X and Z so planar circles/rectangles don't look squashed.
	if x_all and z_all:
		x_min, x_max = min(x_all), max(x_all)
		z_min, z_max = min(z_all), max(z_all)
		x_rng = x_max - x_min
		z_rng = z_max - z_min
		max_rng = max(x_rng, z_rng, 1e-6)
		x_c = 0.5 * (x_min + x_max)
		z_c = 0.5 * (z_min + z_max)
		ax.set_xlim(x_c - 0.5 * max_rng, x_c + 0.5 * max_rng)
		ax.set_zlim(z_c - 0.5 * max_rng, z_c + 0.5 * max_rng)
		try:
			# Make the Y axis visually shorter (about half) than X/Z,
			# but not so short that tick labels become unreadable.
			# This changes only the rendering aspect, not the data limits or tick values.
			ax.set_box_aspect((1.0, 0.5, 1.0))
		except Exception:
			pass


def _plot_pose_error_norm_comparison(experiments: Sequence[ExperimentData]) -> None:
	import matplotlib.pyplot as plt

	usable = [e for e in experiments if e.desired_pose is not None and e.real_pose is not None]
	if not usable:
		return

	fig, axs = plt.subplots(2, 1, sharex=True)
	fig.suptitle("EE Pose tracking errors (comparison)")

	for exp in usable:
		desired = exp.desired_pose
		real = exp.real_pose
		assert desired is not None
		assert real is not None
		err_t, pos_err, ori_err_deg = _compute_pose_tracking_errors(desired, real)
		if not err_t:
			continue
		axs[0].plot(err_t, pos_err, label=exp.label)
		axs[1].plot(err_t, ori_err_deg, label=exp.label)

	for ax in axs:
		ax.grid(True)
		ax.legend(loc="best")
	axs[0].set_ylabel(r"$\|\|e_p\|\|\;[m]$")
	axs[1].set_xlabel("t [s] (desired timestamps)")
	axs[1].set_ylabel(r"$\|\|e_R\|\|\;[°]$")


def _plot_odometry_comparison(experiments: Sequence[ExperimentData]) -> None:
	import matplotlib.pyplot as plt

	usable = [e for e in experiments if e.odom is not None]
	if not usable:
		return

	fig, axs = plt.subplots(3, 2, sharex=True)
	fig.suptitle("Drone odometry (comparison, deviations from initial)")

	for exp in usable:
		odom = exp.odom
		assert odom is not None
		x0, y0, z0 = odom.x[0], odom.y[0], odom.z[0]
		r0, p0, yw0 = odom.roll[0], odom.pitch[0], odom.yaw[0]

		dx = [x - x0 for x in odom.x]
		dy = [y - y0 for y in odom.y]
		dz = [z - z0 for z in odom.z]
		droll = [_rad_to_deg(_wrap_to_pi(r - r0)) for r in odom.roll]
		dpitch = [_rad_to_deg(_wrap_to_pi(p - p0)) for p in odom.pitch]
		dyaw = [_rad_to_deg(_wrap_to_pi(yw - yw0)) for yw in odom.yaw]

		axs[0, 0].plot(odom.t, dx, label=exp.label)
		axs[1, 0].plot(odom.t, dy, label=exp.label)
		axs[2, 0].plot(odom.t, dz, label=exp.label)
		axs[0, 1].plot(odom.t, droll, label=exp.label)
		axs[1, 1].plot(odom.t, dpitch, label=exp.label)
		axs[2, 1].plot(odom.t, dyaw, label=exp.label)

	axs[0, 0].set_ylabel("Δx [m]")
	axs[1, 0].set_ylabel("Δy [m]")
	axs[2, 0].set_ylabel("Δz [m]")
	axs[2, 0].set_xlabel("t [s]")
	axs[0, 1].set_ylabel("Δroll [°]")
	axs[1, 1].set_ylabel("Δpitch [°]")
	axs[2, 1].set_ylabel("Δyaw [°]")
	axs[2, 1].set_xlabel("t [s]")

	for i in range(3):
		for j in range(2):
			axs[i, j].grid(True)
			axs[i, j].legend(loc="best")


def _plot_odometry_displacement_norms_comparison(experiments: Sequence[ExperimentData]) -> None:
	import matplotlib.pyplot as plt

	usable = [e for e in experiments if e.odom is not None and e.odom.t]
	if not usable:
		return

	fig, axs = plt.subplots(2, 1, sharex=True)
	fig.suptitle("Drone displacement norms (comparison)")

	for exp in usable:
		odom = exp.odom
		assert odom is not None
		x0, y0, z0 = odom.x[0], odom.y[0], odom.z[0]
		q0x, q0y, q0z, q0w = _rpy_to_quat(odom.roll[0], odom.pitch[0], odom.yaw[0])

		pos_norm: List[float] = []
		ang_disp_norm_deg: List[float] = []
		for k in range(len(odom.t)):
			dx = odom.x[k] - x0
			dy = odom.y[k] - y0
			dz = odom.z[k] - z0
			pos_norm.append(math.sqrt(dx * dx + dy * dy + dz * dz))

			qkx, qky, qkz, qkw = _rpy_to_quat(odom.roll[k], odom.pitch[k], odom.yaw[k])
			ex, ey, ez = _relative_rotation_axis_angle_vec(
				q0x, q0y, q0z, q0w, qkx, qky, qkz, qkw
			)
			ang_disp_norm_deg.append(_rad_to_deg(math.sqrt(ex * ex + ey * ey + ez * ez)))

		axs[0].plot(odom.t, pos_norm, label=exp.label)
		axs[1].plot(odom.t, ang_disp_norm_deg, label=exp.label)

	for ax in axs:
		ax.grid(True)
		ax.legend(loc="best")
	axs[0].set_ylabel("||Δp|| [m]")
	axs[1].set_ylabel("||Δθ|| [°]")
	axs[1].set_xlabel("t [s]")


def _plot_odometry_rms_disturbance_comparison(experiments: Sequence[ExperimentData]) -> None:
	"""Overlay RMS attitude disturbance across experiments.

	RMS is computed from roll/pitch/yaw deviations (deg) w.r.t. initial attitude:
	RMS = sqrt((droll^2 + dpitch^2 + dyaw^2) / 3).
	"""
	import matplotlib.pyplot as plt

	usable = [e for e in experiments if e.odom is not None and e.odom.t]
	if not usable:
		return

	fig, ax = plt.subplots()
	fig.suptitle("Drone attitude disturbance (RMS, comparison)")

	for exp in usable:
		odom = exp.odom
		assert odom is not None
		r0, p0, yw0 = odom.roll[0], odom.pitch[0], odom.yaw[0]
		droll = [_rad_to_deg(_wrap_to_pi(r - r0)) for r in odom.roll]
		dpitch = [_rad_to_deg(_wrap_to_pi(p - p0)) for p in odom.pitch]
		dyaw = [_rad_to_deg(_wrap_to_pi(yw - yw0)) for yw in odom.yaw]

		rms: List[float] = []
		for dr, dp, dy in zip(droll, dpitch, dyaw):
			rms.append(math.sqrt((dr * dr + dp * dp + dy * dy) / 3.0))

		ax.plot(odom.t, rms, label=exp.label)

	ax.grid(True)
	ax.set_xlabel("t [s]")
	ax.set_ylabel("RMS [°]")
	ax.legend(loc="best")


def _plot_sensor_combined_accel_comparison(experiments: Sequence[ExperimentData]) -> None:
	import matplotlib.pyplot as plt

	usable = [e for e in experiments if e.accel is not None and e.accel.t]
	if not usable:
		return

	fig, axs = plt.subplots(3, 1, sharex=True)
	fig.suptitle("SensorCombined accelerometer (comparison)")

	for exp in usable:
		accel = exp.accel
		assert accel is not None
		axs[0].plot(accel.t, accel.ax, label=exp.label)
		axs[1].plot(accel.t, accel.ay, label=exp.label)
		axs[2].plot(accel.t, accel.az, label=exp.label)

	axs[0].set_ylabel("ax [m/s^2]")
	axs[1].set_ylabel("ay [m/s^2]")
	axs[2].set_ylabel("az [m/s^2]")
	axs[2].set_xlabel("t [s]")
	for ax in axs:
		ax.grid(True)
		ax.legend(loc="best")


def _plot_odometry(odom: OdomSeries, title_prefix: str) -> None:
	import matplotlib.pyplot as plt

	# Plot deviations from initial values
	x0, y0, z0 = odom.x[0], odom.y[0], odom.z[0]
	r0, p0, yw0 = odom.roll[0], odom.pitch[0], odom.yaw[0]

	dx = [x - x0 for x in odom.x]
	dy = [y - y0 for y in odom.y]
	dz = [z - z0 for z in odom.z]

	droll = [_rad_to_deg(_wrap_to_pi(r - r0)) for r in odom.roll]
	dpitch = [_rad_to_deg(_wrap_to_pi(p - p0)) for p in odom.pitch]
	dyaw = [_rad_to_deg(_wrap_to_pi(yw - yw0)) for yw in odom.yaw]

	fig, axs = plt.subplots(3, 2, sharex=True)
	fig.suptitle(f"{title_prefix} - Drone odometry")

	axs[0, 0].plot(odom.t, dx, color="red")
	axs[1, 0].plot(odom.t, dy, color=(31 / 255, 230 / 255, 51 / 255))
	axs[2, 0].plot(odom.t, dz, color="blue")
	axs[0, 0].set_ylabel("Δx [m]")
	axs[1, 0].set_ylabel("Δy [m]")
	axs[2, 0].set_ylabel("Δz [m]")
	axs[2, 0].set_xlabel("t [s]")
	for i in range(3):
		axs[i, 0].grid(True)

	axs[0, 1].plot(odom.t, droll, color="red")
	axs[1, 1].plot(odom.t, dpitch, color=(31 / 255, 230 / 255, 51 / 255))
	axs[2, 1].plot(odom.t, dyaw, color="blue")
	axs[0, 1].set_ylabel("Δroll [°]")
	axs[1, 1].set_ylabel("Δpitch [°]")
	axs[2, 1].set_ylabel("Δyaw [°]")
	axs[2, 1].set_xlabel("t [s]")
	for i in range(3):
		axs[i, 1].grid(True)


def _plot_odometry_displacement_norms(odom: OdomSeries, title_prefix: str) -> None:
	import matplotlib.pyplot as plt

	if not odom.t:
		return

	x0, y0, z0 = odom.x[0], odom.y[0], odom.z[0]
	q0x, q0y, q0z, q0w = _rpy_to_quat(odom.roll[0], odom.pitch[0], odom.yaw[0])

	pos_norm: List[float] = []
	ang_disp_norm_deg: List[float] = []
	for k in range(len(odom.t)):
		dx = odom.x[k] - x0
		dy = odom.y[k] - y0
		dz = odom.z[k] - z0
		pos_norm.append(math.sqrt(dx * dx + dy * dy + dz * dz))

		qkx, qky, qkz, qkw = _rpy_to_quat(odom.roll[k], odom.pitch[k], odom.yaw[k])
		ex, ey, ez = _relative_rotation_axis_angle_vec(q0x, q0y, q0z, q0w, qkx, qky, qkz, qkw)
		ang_disp_norm_deg.append(_rad_to_deg(math.sqrt(ex * ex + ey * ey + ez * ez)))

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ln1 = ax1.plot(odom.t, pos_norm, label="||Δp||", color="tab:blue")
	ln2 = ax2.plot(odom.t, ang_disp_norm_deg, label="||Δθ||", color="tab:orange")

	ax1.grid(True)
	ax1.set_title(f"{title_prefix} - Drone displacement norms")
	ax1.set_xlabel("t [s]")
	ax1.set_ylabel("||Δp|| [m]")
	ax2.set_ylabel("||Δθ|| [°]")

	lines = ln1 + ln2
	labels = [l.get_label() for l in lines]
	ax1.legend(lines, labels, loc="best")


def _plot_odometry_angular_velocity(angvel: AngularVelSeries, title_prefix: str) -> None:
	import matplotlib.pyplot as plt

	if not angvel.t:
		return

	wx_deg_s = [_rad_to_deg(w) for w in angvel.wx]
	wy_deg_s = [_rad_to_deg(w) for w in angvel.wy]
	wz_deg_s = [_rad_to_deg(w) for w in angvel.wz]

	fig, axs = plt.subplots(3, 1, sharex=True)
	fig.suptitle(f"{title_prefix} - Drone angular velocity (odometry)")

	axs[0].plot(angvel.t, wx_deg_s, color="red")
	axs[1].plot(angvel.t, wy_deg_s, color=(31 / 255, 230 / 255, 51 / 255))
	axs[2].plot(angvel.t, wz_deg_s, color="blue")
	axs[0].set_ylabel("ωx [°/s]")
	axs[1].set_ylabel("ωy [°/s]")
	axs[2].set_ylabel("ωz [°/s]")
	axs[2].set_xlabel("t [s]")
	for ax in axs:
		ax.grid(True)


def _plot_odometry_rms_disturbance(odom: OdomSeries, title_prefix: str) -> None:
	"""Plot RMS disturbance of base attitude.

	Computed as RMS of the three attitude angle deviations (roll, pitch, yaw)
	with respect to the initial attitude.
	"""
	import matplotlib.pyplot as plt

	if not odom.t:
		return

	r0, p0, yw0 = odom.roll[0], odom.pitch[0], odom.yaw[0]
	# Disturbance angles (deg)
	droll = [_rad_to_deg(_wrap_to_pi(r - r0)) for r in odom.roll]
	dpitch = [_rad_to_deg(_wrap_to_pi(p - p0)) for p in odom.pitch]
	dyaw = [_rad_to_deg(_wrap_to_pi(yw - yw0)) for yw in odom.yaw]

	rms: List[float] = []
	for dr, dp, dy in zip(droll, dpitch, dyaw):
		rms.append(math.sqrt((dr * dr + dp * dp + dy * dy) / 3.0))

	plt.figure()
	plt.plot(odom.t, rms, color="purple")
	plt.grid(True)
	plt.title(f"{title_prefix} - $\\delta_{{RMS}}$ disturbance (attitude)")
	plt.xlabel("t [s]")
	plt.ylabel(r"$\delta_{RMS}$ [°]")


def _plot_sensor_combined_imu(
	accel: Optional[AccelSeries],
	gyro: Optional[GyroSeries],
	title_prefix: str,
) -> None:
	import matplotlib.pyplot as plt

	if accel is None and gyro is None:
		return

	if accel is not None and gyro is not None:
		fig, axs = plt.subplots(3, 2, sharex=True)
		fig.suptitle(f"{title_prefix} - SensorCombined IMU")

		axs[0, 0].plot(accel.t, accel.ax)
		axs[1, 0].plot(accel.t, accel.ay)
		axs[2, 0].plot(accel.t, accel.az)
		axs[0, 0].set_ylabel("ax [m/s^2]")
		axs[1, 0].set_ylabel("ay [m/s^2]")
		axs[2, 0].set_ylabel("az [m/s^2]")
		axs[2, 0].set_xlabel("t [s]")
		for i in range(3):
			axs[i, 0].grid(True)

		gx_deg_s = [_rad_to_deg(w) for w in gyro.gx]
		gy_deg_s = [_rad_to_deg(w) for w in gyro.gy]
		gz_deg_s = [_rad_to_deg(w) for w in gyro.gz]
		axs[0, 1].plot(gyro.t, gx_deg_s)
		axs[1, 1].plot(gyro.t, gy_deg_s)
		axs[2, 1].plot(gyro.t, gz_deg_s)
		axs[0, 1].set_ylabel("ωx [°/s]")
		axs[1, 1].set_ylabel("ωy [°/s]")
		axs[2, 1].set_ylabel("ωz [°/s]")
		axs[2, 1].set_xlabel("t [s]")
		for i in range(3):
			axs[i, 1].grid(True)
		return

	# Fallback: keep backward-compatible behavior if only one of the two exists.
	if accel is not None:
		fig, axs = plt.subplots(3, 1, sharex=True)
		fig.suptitle(f"{title_prefix} - SensorCombined accelerometer")
		axs[0].plot(accel.t, accel.ax)
		axs[1].plot(accel.t, accel.ay)
		axs[2].plot(accel.t, accel.az)
		axs[0].set_ylabel("ax [m/s^2]")
		axs[1].set_ylabel("ay [m/s^2]")
		axs[2].set_ylabel("az [m/s^2]")
		axs[2].set_xlabel("t [s]")
		for ax in axs:
			ax.grid(True)


def _plot_real_t960a_twist(twist: TwistSeries, title_prefix: str) -> None:
	import matplotlib.pyplot as plt

	if not twist.t:
		return

	wx_deg_s = [_rad_to_deg(w) for w in twist.wx]
	wy_deg_s = [_rad_to_deg(w) for w in twist.wy]
	wz_deg_s = [_rad_to_deg(w) for w in twist.wz]

	fig, axs = plt.subplots(3, 2, sharex=True)
	fig.suptitle(f"{title_prefix} - Real drone twist (/real_t960a_twist)")

	axs[0, 0].plot(twist.t, twist.vx, color="red")
	axs[1, 0].plot(twist.t, twist.vy, color=(31 / 255, 230 / 255, 51 / 255))
	axs[2, 0].plot(twist.t, twist.vz, color="blue")
	axs[0, 0].set_ylabel("vx [m/s]")
	axs[1, 0].set_ylabel("vy [m/s]")
	axs[2, 0].set_ylabel("vz [m/s]")
	axs[2, 0].set_xlabel("t [s]")
	for i in range(3):
		axs[i, 0].grid(True)

	axs[0, 1].plot(twist.t, wx_deg_s, color="red")
	axs[1, 1].plot(twist.t, wy_deg_s, color=(31 / 255, 230 / 255, 51 / 255))
	axs[2, 1].plot(twist.t, wz_deg_s, color="blue")
	axs[0, 1].set_ylabel("ωx [°/s]")
	axs[1, 1].set_ylabel("ωy [°/s]")
	axs[2, 1].set_ylabel("ωz [°/s]")
	axs[2, 1].set_xlabel("t [s]")
	for i in range(3):
		axs[i, 1].grid(True)
	return


def run(csv_path: Path, show: bool, save_dir: Optional[Path]) -> None:
	"""Generate plots from a single uam_logger_pkg CSV file."""
	try:
		import matplotlib.pyplot as plt
	except Exception as e:
		raise RuntimeError(
			"matplotlib non disponibile. Installa ad esempio: "
			"sudo apt install python3-matplotlib"
		) from e

	groups = _load_csv_grouped(csv_path)

	topic_desired_pose = "/desired_ee_global_pose"
	topic_real_pose = "/ee_world_pose"
	topic_odom = "/model/t960a_0/odometry"
	topic_mocap_pose = "/t960a/pose"
	topic_sensor = "/fmu/out/sensor_combined"
	topic_real_twist = "/real_t960a_twist"

	desired_pose = _extract_pose_series(groups.get(topic_desired_pose, []))
	real_pose = _extract_pose_series(groups.get(topic_real_pose, []))
	# Base-drone pose source:
	# - Real: motion capture PoseStamped on /t960a/pose (logged by uam_logger_node)
	# - Sim: Gazebo odometry on /model/t960a_0/odometry
	base_pose_topic = topic_mocap_pose if topic_mocap_pose in groups else topic_odom
	odom = _extract_odom_series(groups.get(base_pose_topic, []))
	odom_angvel = _extract_odom_angular_velocity_series(groups.get(topic_odom, []))
	accel = _extract_accel_series(groups.get(topic_sensor, []))
	gyro = _extract_gyro_series(groups.get(topic_sensor, []))
	real_twist = _extract_twist_series(groups.get(topic_real_twist, []))
	controller_params = _extract_controller_params(groups)

	title_prefix = csv_path.stem
	base_title_prefix = title_prefix if base_pose_topic == topic_odom else f"{title_prefix} (mocap)"

	if desired_pose is not None and real_pose is not None:
		_plot_ee_trajectories(desired_pose, real_pose, title_prefix)
		_plot_pose_error_norm(desired_pose, real_pose, title_prefix)

	if odom is not None:
		_plot_odometry(odom, base_title_prefix)
		_plot_odometry_rms_disturbance(odom, base_title_prefix)
		_plot_odometry_displacement_norms(odom, base_title_prefix)

	if odom_angvel is not None:
		_plot_odometry_angular_velocity(odom_angvel, title_prefix)

	if accel is not None or gyro is not None:
		_plot_sensor_combined_imu(accel, gyro, title_prefix)

	if real_twist is not None:
		_plot_real_t960a_twist(real_twist, title_prefix)

	# Show controller parameters (if present) also for single-file runs.
	_plot_controller_params_table(
		[
			ExperimentData(
				label=title_prefix,
				csv_path=csv_path,
				desired_pose=desired_pose,
				real_pose=real_pose,
				base_pose_topic=base_pose_topic,
				odom=odom,
				accel=accel,
				gyro=gyro,
				controller_params=controller_params,
			)
		]
	)

	if save_dir is not None:
		save_dir.mkdir(parents=True, exist_ok=True)
		for fig_num in plt.get_fignums():
			fig = plt.figure(fig_num)
			out = save_dir / f"{csv_path.stem}_fig{fig_num}.png"
			fig.savefig(out, dpi=200, bbox_inches="tight")

	if show:
		plt.show()
	else:
		plt.close("all")


def run_comparison(
	csv_paths: Sequence[Path],
	show: bool,
	save_dir: Optional[Path],
	labels: Optional[Sequence[str]] = None,
	print_params_markdown: bool = False,
) -> None:
	"""Generate comparison plots from multiple CSV files.

	The script overlays multiple experiments on the same figures to make
	performance comparisons easier.
	"""
	try:
		import matplotlib.pyplot as plt
	except Exception as e:
		raise RuntimeError(
			"matplotlib non disponibile. Installa ad esempio: "
			"sudo apt install python3-matplotlib"
		) from e

	if not csv_paths:
		return

	normalized_paths = [p.expanduser() for p in csv_paths]
	for p in normalized_paths:
		if not p.exists():
			raise FileNotFoundError(f"CSV non trovato: {p}")

	if labels is not None and len(labels) != len(normalized_paths):
		raise ValueError("--labels deve avere la stessa lunghezza di --csv")

	experiments: List[ExperimentData] = []
	for i, p in enumerate(normalized_paths):
		label = labels[i] if labels is not None else p.stem
		experiments.append(_load_experiment(p, label=label))

	md = _format_controller_params_markdown(experiments)
	if print_params_markdown and md is not None:
		print(md)

	_plot_ee_trajectories_comparison(experiments)
	_plot_pose_error_norm_comparison(experiments)
	_plot_odometry_comparison(experiments)
	_plot_odometry_rms_disturbance_comparison(experiments)
	_plot_odometry_displacement_norms_comparison(experiments)
	_plot_sensor_combined_accel_comparison(experiments)
	_plot_controller_params_table(experiments)

	if save_dir is not None:
		save_dir.mkdir(parents=True, exist_ok=True)
		for fig_num in plt.get_fignums():
			fig = plt.figure(fig_num)
			out = save_dir / f"comparison_fig{fig_num}.png"
			fig.savefig(out, dpi=200, bbox_inches="tight")

	if show:
		plt.show()
	else:
		plt.close("all")


def main(argv: Optional[List[str]] = None) -> None:
	"""CLI entry point for offline plotting."""
	parser = argparse.ArgumentParser(
		description="Offline plotting for uam_logger_pkg CSV logs"
	)
	parser.add_argument(
		"--csv",
		required=True,
		nargs="+",
		help="Path del/dei file CSV generato/i dal logger (uno o piu')",
	)
	parser.add_argument(
		"--labels",
		default=None,
		nargs="+",
		help="Etichette opzionali (una per CSV) usate nelle legende",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Non mostrare le figure (utile se usi solo --save-dir)",
	)
	parser.add_argument(
		"--save-dir",
		default=None,
		help="Directory dove salvare i PNG (opzionale). Se non specificato, non salva su disco.",
	)
	parser.add_argument(
		"--print-params-markdown",
		action="store_true",
		help="Stampa su terminale una tabella Markdown dei parametri controller trovati nel CSV (default: false)",
	)
	args = parser.parse_args(argv)

	csv_paths = [Path(p) for p in args.csv]
	save_dir = Path(args.save_dir).expanduser() if args.save_dir else None

	if len(csv_paths) == 1:
		csv_path = csv_paths[0].expanduser()
		if not csv_path.exists():
			raise FileNotFoundError(f"CSV non trovato: {csv_path}")
		run(csv_path=csv_path, show=not args.no_show, save_dir=save_dir)
		if args.print_params_markdown:
			exp = _load_experiment(csv_path, label=csv_path.stem)
			md = _format_controller_params_markdown([exp])
			if md is not None:
				print(md)
		return

	run_comparison(
		csv_paths=[p for p in csv_paths],
		show=not args.no_show,
		save_dir=save_dir,
		labels=args.labels,
		print_params_markdown=args.print_params_markdown,
	)


if __name__ == "__main__":
	main()
