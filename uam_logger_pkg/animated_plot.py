"""Animated plotting for uam_logger_pkg CSV exports.

This script loads the CSV produced by `uam_logger_node` and generates *animated*
Matplotlib plots whose final appearance matches the static ones from
`offline_plotting.py`:

- 3D desired vs real end-effector trajectory (yaw-aligned)
- End-effector pose tracking errors (position/orientation norm) over time

The animation evolves with the experiment timestamps (simulation time stored in
the CSV as `t`).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _median(values: List[float]) -> Optional[float]:
	if not values:
		return None
	s = sorted(values)
	n = len(s)
	m = n // 2
	if n % 2 == 1:
		return s[m]
	return 0.5 * (s[m - 1] + s[m])


def _infer_fps_from_timestamps(t: List[float], speed: float, default_fps: int = 30) -> int:
	"""Infer a reasonable FPS from timestamps and speed.

	The animation is saved with a fixed FPS, so we approximate real-time playback
	using the median sample period.
	"""
	if len(t) < 2:
		return max(1, int(round(default_fps * speed)))
	dts = [t[i + 1] - t[i] for i in range(len(t) - 1)]
	dts = [dt for dt in dts if dt > 1e-9]
	med = _median(dts)
	if med is None or med <= 1e-9:
		return max(1, int(round(default_fps * speed)))
	fps = int(round((1.0 / med) * speed))
	return max(1, min(240, fps))


def _resolve_save_paths(save: Path, stem: str) -> Tuple[Path, Optional[Path]]:
	"""Return (trajectory_path, error_path).

	If `save` is a directory, two files are created in it.
	If `save` is a file path, the trajectory is saved there, and the error plot
	(if any) is saved next to it with a `_errors` suffix.
	"""
	save = save.expanduser()
	if save.suffix == "":
		# Treat as directory.
		ext = ".mp4"
		traj_path = save / f"{stem}_ee_traj{ext}"
		err_path = save / f"{stem}_ee_errors{ext}"
		return traj_path, err_path

	traj_path = save
	err_path = save.with_name(save.stem + "_errors" + save.suffix)
	return traj_path, err_path


def _ensure_parent_dir(p: Path) -> None:
	parent = p.parent
	parent.mkdir(parents=True, exist_ok=True)


def _save_animation(anim: object, out_path: Path, fps: int) -> None:
	"""Save a Matplotlib FuncAnimation to disk.

	Supported formats:
	- .mp4 (requires ffmpeg)
	- .gif (uses PillowWriter)
	"""
	from matplotlib.animation import PillowWriter, writers

	out_path = out_path.expanduser()
	_ensure_parent_dir(out_path)

	suffix = out_path.suffix.lower()
	if suffix == ".gif":
		writer = PillowWriter(fps=fps)
		anim.save(str(out_path), writer=writer)  # type: ignore[attr-defined]
		return

	if suffix == ".mp4":
		if not writers.is_available("ffmpeg"):
			raise RuntimeError(
				"Cannot save MP4: Matplotlib 'ffmpeg' writer not available. "
				"Install ffmpeg or use --save with a .gif extension."
			)
		writer = writers["ffmpeg"](fps=fps, bitrate=-1)
		anim.save(str(out_path), writer=writer)  # type: ignore[attr-defined]
		return

	raise RuntimeError(f"Unsupported output format '{suffix}'. Use .mp4 or .gif")


def _compute_alignment_yaw(px: List[float], py: List[float]) -> float:
	"""Return yaw angle (rad) to align the main XY direction with +X.

	This is the same alignment strategy used by `offline_plotting._plot_ee_trajectories`.
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
	return -alpha


def _rotate_about_z(x: float, y: float, yaw: float) -> Tuple[float, float]:
	c = math.cos(yaw)
	s = math.sin(yaw)
	xr = c * x - s * y
	yr = s * x + c * y
	return xr, yr


def _build_yaw_aligned_trajectories(
	desired_px: List[float],
	desired_py: List[float],
	desired_pz: List[float],
	real_px: List[float],
	real_py: List[float],
	real_pz: List[float],
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
	"""Yaw-align desired and real trajectories like offline_plotting."""
	yaw_align = _compute_alignment_yaw(desired_px, desired_py)

	dx_r: List[float] = []
	dy_r: List[float] = []
	dz_r: List[float] = list(desired_pz)

	rx_r: List[float] = []
	ry_r: List[float] = []
	rz_r: List[float] = list(real_pz)

	for x, y in zip(desired_px, desired_py):
		xr, yr = _rotate_about_z(x, y, yaw_align)
		dx_r.append(xr)
		dy_r.append(yr)
	for x, y in zip(real_px, real_py):
		xr, yr = _rotate_about_z(x, y, yaw_align)
		rx_r.append(xr)
		ry_r.append(yr)

	return dx_r, dy_r, dz_r, rx_r, ry_r, rz_r


def run(csv_path: Path, speed: float, save: Optional[Path]) -> None:
	"""Load a CSV file and show animated plots."""
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation

	from .offline_plotting import (
		_compute_pose_tracking_errors,
		_extract_pose_series,
		_load_csv_grouped,
		_nearest_indices,
	)

	if speed <= 0.0:
		raise ValueError("speed must be > 0")

	csv_path = csv_path.expanduser()
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	groups: Dict[str, List[Dict[str, str]]] = _load_csv_grouped(csv_path)
	desired = _extract_pose_series(groups.get("/desired_ee_global_pose", []))
	real = _extract_pose_series(groups.get("/ee_world_pose", []))
	if desired is None or real is None:
		raise RuntimeError(
			"Missing required pose topics in CSV. "
			"Need /desired_ee_global_pose and /ee_world_pose."
		)

	# Match real samples to desired timestamps (same strategy as error plots).
	idx = _nearest_indices(real.t, desired.t)
	if not idx:
		raise RuntimeError("Cannot time-align trajectories (empty nearest-index list).")

	real_px_m = [real.px[j] for j in idx]
	real_py_m = [real.py[j] for j in idx]
	real_pz_m = [real.pz[j] for j in idx]

	dx_r, dy_r, dz_r, rx_r, ry_r, rz_r = _build_yaw_aligned_trajectories(
		desired_px=desired.px,
		desired_py=desired.py,
		desired_pz=desired.pz,
		real_px=real_px_m,
		real_py=real_py_m,
		real_pz=real_pz_m,
	)

	# Precompute tracking errors (same content as offline_plotting).
	err_t, pos_err_norm, ori_err_norm_deg = _compute_pose_tracking_errors(desired, real)

	# ---- Figure 1: 3D trajectory ----
	fig_traj = plt.figure()
	ax3d = fig_traj.add_subplot(111, projection="3d")
	ax3d.set_title(f"{csv_path.stem} - End-effector trajectory (yaw-aligned)")
	ax3d.set_xlabel("x [m]")
	ax3d.set_ylabel("y [m]")
	ax3d.set_zlabel("z [m]")

	(desired_line,) = ax3d.plot([], [], [], label="desired")
	(real_line,) = ax3d.plot([], [], [], label="real")
	ax3d.legend(loc="best")

	# Fix axes limits to avoid re-scaling while animating.
	x_all = dx_r + rx_r
	y_all = dy_r + ry_r
	z_all = dz_r + rz_r
	if x_all:
		x_min, x_max = min(x_all), max(x_all)
		x_pad = 0.05 * max(1e-6, (x_max - x_min))
		ax3d.set_xlim(x_min - x_pad, x_max + x_pad)
	if z_all:
		z_min, z_max = min(z_all), max(z_all)
		z_pad = 0.05 * max(1e-6, (z_max - z_min))
		ax3d.set_zlim(z_min - z_pad, z_max + z_pad)

	# Same "perpendicular axis" widening used by offline_plotting.
	perp_scale = 10.0
	if y_all:
		y0 = sum(y_all) / len(y_all)
		max_dev = max(abs(y - y0) for y in y_all)
		if max_dev <= 1e-9:
			max_dev = 0.01
		ax3d.set_ylim(y0 - perp_scale * max_dev, y0 + perp_scale * max_dev)

	# ---- Figure 2: pose error norm ----
	fig_err, axs = plt.subplots(2, 1, sharex=True)
	fig_err.suptitle(f"{csv_path.stem} - EE Pose tracking errors")
	
	(pos_line,) = axs[0].plot([], [], color="red")
	(ori_line,) = axs[1].plot([], [], color="red")
	axs[0].grid(True)
	axs[0].set_ylabel(r"$\|\|e_p\|\|\;[m]$")
	axs[1].grid(True)
	axs[1].set_xlabel("t [s] (desired timestamps)")
	axs[1].set_ylabel(r"$\|\|e_R\|\|\;[°]$")

	if err_t:
		axs[1].set_xlim(err_t[0], err_t[-1])
		p_max = max(pos_err_norm) if pos_err_norm else 1.0
		r_max = max(ori_err_norm_deg) if ori_err_norm_deg else 1.0
		axs[0].set_ylim(0.0, 1.05 * max(1e-6, p_max))
		axs[1].set_ylim(0.0, 1.05 * max(1e-6, r_max))

	# ---- Animation state ----
	n_frames = len(desired.t)
	anim_holder: Dict[str, object] = {}

	def _init_traj() -> Tuple[object, ...]:
		desired_line.set_data([], [])
		desired_line.set_3d_properties([])
		real_line.set_data([], [])
		real_line.set_3d_properties([])
		return (desired_line, real_line)

	def _init_err() -> Tuple[object, ...]:
		pos_line.set_data([], [])
		ori_line.set_data([], [])
		return (pos_line, ori_line)

	def _update_traj(k: int) -> Tuple[object, ...]:
		kk = max(1, min(n_frames, k + 1))

		desired_line.set_data(dx_r[:kk], dy_r[:kk])
		desired_line.set_3d_properties(dz_r[:kk])
		real_line.set_data(rx_r[:kk], ry_r[:kk])
		real_line.set_3d_properties(rz_r[:kk])

		# Real-time-ish pacing using the CSV timestamps.
		if err_t and kk >= 2 and "anim" in anim_holder:
			dt = err_t[min(kk - 1, len(err_t) - 1)] - err_t[min(kk - 2, len(err_t) - 1)]
			if dt < 0.0:
				dt = 0.0
			dt_ms = max(1, int((1000.0 * dt) / speed))
			anim = anim_holder["anim"]
			try:
				anim.event_source.interval = dt_ms  # type: ignore[attr-defined]
			except Exception:
				pass

		return (desired_line, real_line)

	def _update_err(k: int) -> Tuple[object, ...]:
		kk = max(1, min(n_frames, k + 1))
		if err_t:
			j = kk
			if j > len(err_t):
				j = len(err_t)
			pos_line.set_data(err_t[:j], pos_err_norm[:j])
			ori_line.set_data(err_t[:j], ori_err_norm_deg[:j])
		return (pos_line, ori_line)

	anim_traj = FuncAnimation(
		fig_traj,
		_update_traj,
		frames=n_frames,
		init_func=_init_traj,
		interval=max(1, int(50 / speed)),
		blit=False,
		repeat=False,
	)
	anim_holder["anim"] = anim_traj

	anim_err = FuncAnimation(
		fig_err,
		_update_err,
		frames=n_frames,
		init_func=_init_err,
		interval=max(1, int(50 / speed)),
		blit=False,
		repeat=False,
	)

	# Optional export.
	if save is not None:
		traj_out, err_out = _resolve_save_paths(save, csv_path.stem)
		fps = _infer_fps_from_timestamps(desired.t, speed=speed, default_fps=30)
		_save_animation(anim_traj, traj_out, fps=fps)
		# Export error animation only if error data exists.
		if err_out is not None and err_t:
			_save_animation(anim_err, err_out, fps=fps)

	plt.show()


def main(argv: Optional[List[str]] = None) -> None:
	"""CLI entry point."""
	parser = argparse.ArgumentParser(description="Animated plotting for uam_logger_pkg CSV logs")
	parser.add_argument(
		"--csv",
		required=True,
		help="Path del file CSV generato dal logger",
	)
	parser.add_argument(
		"--speed",
		type=float,
		default=1.0,
		help="Fattore di velocita' dell'animazione (1.0=real-time, 2.0=2x piu' veloce, 0.5=2x piu' lento)",
	)
	parser.add_argument(
		"--save",
		default=None,
		help=(
			"Percorso di export dell'animazione. Puoi passare una directory (salva due file) "
			"oppure un file .mp4/.gif (salva traiettoria li' e errori in un secondo file *_errors)."
		),
	)
	args = parser.parse_args(argv)

	save_path = Path(args.save) if args.save else None
	run(Path(args.csv), speed=float(args.speed), save=save_path)


if __name__ == "__main__":
	main()
