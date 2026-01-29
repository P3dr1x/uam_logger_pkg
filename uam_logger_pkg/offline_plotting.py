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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

	idx = _nearest_indices(real.t, desired.t)
	if not idx:
		return

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

	fig, axs = plt.subplots(2, 1, sharex=True)
	fig.suptitle(f"{title_prefix} - EE Pose tracking errors")

	axs[0].plot(err_t, pos_err_norm, color="red")
	axs[0].grid(True)
	axs[0].set_ylabel(r"$\|\|e_p\|\|\;[m]$")

	axs[1].plot(err_t, ori_err_norm_deg, color="red")
	axs[1].grid(True)
	axs[1].set_xlabel("t [s] (desired timestamps)")
	axs[1].set_ylabel(r"$\|\|e_R\|\|\;[°]$")


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
	"""Generate plots from a uam_logger_pkg CSV file.

	Args:
		csv_path: Path to the CSV produced by the logger.
		show: If True, opens matplotlib GUI windows.
		save_dir: If set, saves figures as PNG in this directory.
	"""
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


def main(argv: Optional[List[str]] = None) -> None:
	"""CLI entry point for offline plotting."""
	parser = argparse.ArgumentParser(
		description="Offline plotting for uam_logger_pkg CSV logs"
	)
	parser.add_argument("--csv", required=True, help="Path del file CSV generato dal logger")
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
	args = parser.parse_args(argv)

	csv_path = Path(args.csv).expanduser()
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV non trovato: {csv_path}")

	save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
	run(
		csv_path=csv_path,
		show=not args.no_show,
		save_dir=save_dir,
	)


if __name__ == "__main__":
	main()
