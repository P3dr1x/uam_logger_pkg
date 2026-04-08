"""Mean ± SD bar chart comparison across multiple controller families.

This script compares the *average* (time-mean) behavior of a few metrics across
multiple runs (CSV logs) of different controller families.

Metrics are computed with the same definitions used in `offline_plotting.py`:
- End-effector position tracking error norm ||e_p|| [m]
- UAV translational displacement norm ||Δp|| [m] w.r.t. initial pose
- UAV rotational displacement norm ||Δθ|| [deg] w.r.t. initial attitude

For each run (CSV file), each metric is first averaged over time.
Then, for each family, we compute mean ± standard deviation across runs.

Example:

	ros2 run uam_logger_pkg bar_chart_plotting -- \
		--family Jpinv run1.csv run2.csv run3.csv \
		--family QP run4.csv run5.csv \
		--save-dir ~/.ros/uam_logger/plots

"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import offline_plotting


@dataclass(frozen=True)
class MetricStats:
	"""Aggregate statistics for a metric across runs."""

	mean: float
	sd: float
	n_runs: int
	n_samples: int


def _mean(values: Sequence[float]) -> Optional[float]:
	"""Return the arithmetic mean of a non-empty sequence."""
	if not values:
		return None
	return sum(values) / float(len(values))


def _mean_and_sd_over_time(values: Sequence[float]) -> Optional[MetricStats]:
	"""Return mean and temporal SD of a signal.

	The SD quantifies how the *whole time series* deviates around its own mean.
	This makes SD non-zero even when a family has a single run.
	"""
	n = len(values)
	if n == 0:
		return None
	m = sum(values) / float(n)
	if n >= 2:
		try:
			sd = float(statistics.stdev(values))
		except statistics.StatisticsError:
			sd = 0.0
	else:
		sd = 0.0
	return MetricStats(mean=m, sd=sd, n_runs=1, n_samples=n)


def _aggregate_family_from_runs(run_stats: Sequence[MetricStats]) -> Optional[MetricStats]:
	"""Aggregate per-run (mean, temporal SD) into per-family stats.

	- family mean: average of run means (unweighted, each run counts equally)
	- family SD: pooled temporal SD (weighted by sample counts per run)

	This represents typical within-run variability for the family.
	"""
	if not run_stats:
		return None

	mean_across_runs = sum(s.mean for s in run_stats) / float(len(run_stats))
	# Pooled variance across runs using (n_i - 1) weights.
	den = sum(max(0, s.n_samples - 1) for s in run_stats)
	if den > 0:
		var = sum((max(0, s.n_samples - 1)) * (s.sd ** 2) for s in run_stats) / float(den)
		sd_pooled = math.sqrt(max(0.0, var))
	else:
		sd_pooled = 0.0

	return MetricStats(
		mean=mean_across_runs,
		sd=sd_pooled,
		n_runs=len(run_stats),
		n_samples=sum(s.n_samples for s in run_stats),
	)


def _compute_run_metrics(
	csv_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
	"""Compute time-mean metrics for a single CSV run.

	Returns (ee_pos_err_mean_m, uav_trans_disp_mean_m, uav_rot_disp_mean_deg).
	Any metric can be None if the required signals are missing.
	"""
	exp = offline_plotting._load_experiment(csv_path=csv_path, label=csv_path.stem)

	ee_mean: Optional[float] = None
	if exp.desired_pose is not None and exp.real_pose is not None:
		_, pos_err_norm, _ = offline_plotting._compute_pose_tracking_errors(
			desired=exp.desired_pose, real=exp.real_pose
		)
		ee_mean = _mean(pos_err_norm)

	trans_mean: Optional[float] = None
	rot_mean: Optional[float] = None
	if exp.odom is not None and exp.odom.t:
		pos_norm = offline_plotting._compute_uav_position_displacement_norm(exp.odom)
		ang_norm_deg = offline_plotting._compute_uav_rotational_displacement_norm_deg(
			exp.odom
		)
		trans_mean = _mean(pos_norm)
		rot_mean = _mean(ang_norm_deg)

	return ee_mean, trans_mean, rot_mean


def _compute_run_metric_stats(
	csv_path: Path,
) -> Tuple[Optional[MetricStats], Optional[MetricStats], Optional[MetricStats]]:
	"""Compute per-run stats: mean and temporal SD for each metric."""
	exp = offline_plotting._load_experiment(csv_path=csv_path, label=csv_path.stem)

	ee_stats: Optional[MetricStats] = None
	if exp.desired_pose is not None and exp.real_pose is not None:
		_, pos_err_norm, _ = offline_plotting._compute_pose_tracking_errors(
			desired=exp.desired_pose, real=exp.real_pose
		)
		ee_stats = _mean_and_sd_over_time(pos_err_norm)

	trans_stats: Optional[MetricStats] = None
	rot_stats: Optional[MetricStats] = None
	if exp.odom is not None and exp.odom.t:
		pos_norm = offline_plotting._compute_uav_position_displacement_norm(exp.odom)
		ang_norm_deg = offline_plotting._compute_uav_rotational_displacement_norm_deg(
			exp.odom
		)
		trans_stats = _mean_and_sd_over_time(pos_norm)
		rot_stats = _mean_and_sd_over_time(ang_norm_deg)

	return ee_stats, trans_stats, rot_stats


def _plot_bar_chart(
	family_names: Sequence[str],
	stats_by_family: Dict[
		str,
		Tuple[Optional[MetricStats], Optional[MetricStats], Optional[MetricStats]],
	],
) -> object:
	"""Return a figure with one subplot per metric.

	We intentionally use different Y-axis scales because metrics have different
	orders of magnitude (EE errors ~ mm, UAV displacement ~ m/deg).
	"""
	import matplotlib.pyplot as plt

	n_families = len(family_names)
	if n_families <= 0:
		raise ValueError("No families to plot")

	fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.5), sharey=False)
	fig.suptitle("Mean ± SD across runs (per family)")

	# Pick consistent colors per family (Matplotlib default cycle).
	prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
	if not prop_cycle:
		prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
	family_colors = [prop_cycle[i % len(prop_cycle)] for i in range(n_families)]

	from matplotlib.patches import Patch

	legend_handles = [
		Patch(facecolor=c, label=n) for n, c in zip(family_names, family_colors)
	]

	# X positions are family indices; each subplot has its own Y scale.
	x = list(range(n_families))
	bar_w = 0.7

	# Metric 0: EE position error (convert to mm for readability)
	metric_titles = [
		"End-effector position tracking error norm (mean ± SD)",
		"UAV translational displacement norm (mean ± SD)",
		"UAV rotational displacement norm (mean ± SD)",
	]
	ylabels = [
		r"$\overline{\|e_p\|}$ [mm]",
		r"$\overline{\|\Delta p\|}$ [m]",
		r"$\overline{\|\Delta\theta\|}$ [°]",
	]

	for mi, ax in enumerate(axs):
		heights: List[float] = []
		yerr: List[float] = []
		valid_mask: List[bool] = []

		for fname in family_names:
			stats = stats_by_family.get(fname)
			ms = stats[mi] if stats is not None else None
			if ms is None:
				heights.append(0.0)
				yerr.append(0.0)
				valid_mask.append(False)
				continue

			# Convert EE metric to mm.
			if mi == 0:
				heights.append(1000.0 * ms.mean)
				yerr.append(1000.0 * ms.sd)
			else:
				heights.append(ms.mean)
				yerr.append(ms.sd)
			valid_mask.append(True)

		bars = ax.bar(
			x,
			heights,
			width=bar_w,
			yerr=yerr,
			capsize=4,
			color=family_colors,
		)
		for b, ok in zip(bars, valid_mask):
			if not ok:
				b.set_alpha(0.0)

		ax.set_title(metric_titles[mi])
		ax.set_ylabel(ylabels[mi])
		ax.set_xticks(x)
		ax.set_xticklabels(list(family_names), rotation=0)
		ax.grid(True, axis="y")
		ax.legend(handles=legend_handles, loc="best")

	fig.tight_layout()
	return fig


def run(
	families: Sequence[Tuple[str, Sequence[Path]]],
	show: bool,
	save_dir: Optional[Path],
) -> None:
	"""Compute stats and show/save the bar chart."""
	stats_by_family: Dict[
		str,
		Tuple[Optional[MetricStats], Optional[MetricStats], Optional[MetricStats]],
	] = {}
	family_names: List[str] = []

	for family_name, csv_paths in families:
		family_names.append(family_name)

		ee_run_stats: List[MetricStats] = []
		trans_run_stats: List[MetricStats] = []
		rot_run_stats: List[MetricStats] = []

		for p in csv_paths:
			ee_stats, trans_stats, rot_stats = _compute_run_metric_stats(p)
			if ee_stats is None:
				print(f"[WARN] Missing EE pose data in: {p}")
			else:
				ee_run_stats.append(ee_stats)
			if trans_stats is None or rot_stats is None:
				print(f"[WARN] Missing odometry/mocap pose data in: {p}")
			else:
				trans_run_stats.append(trans_stats)
				rot_run_stats.append(rot_stats)

		ee_stats = _aggregate_family_from_runs(ee_run_stats)
		trans_stats = _aggregate_family_from_runs(trans_run_stats)
		rot_stats = _aggregate_family_from_runs(rot_run_stats)

		stats_by_family[family_name] = (ee_stats, trans_stats, rot_stats)

		print(f"Family '{family_name}':")
		if ee_stats is None:
			print("  EE pos err: n_runs=0")
		else:
			print(f"  EE pos err: n_runs={ee_stats.n_runs}, n_samples={ee_stats.n_samples}")
		if trans_stats is None:
			print("  UAV trans disp: n_runs=0")
		else:
			print(
				f"  UAV trans disp: n_runs={trans_stats.n_runs}, n_samples={trans_stats.n_samples}"
			)
		if rot_stats is None:
			print("  UAV rot disp: n_runs=0")
		else:
			print(
				f"  UAV rot disp: n_runs={rot_stats.n_runs}, n_samples={rot_stats.n_samples}"
			)

	fig = _plot_bar_chart(family_names=family_names, stats_by_family=stats_by_family)

	if save_dir is not None:
		save_dir = save_dir.expanduser()
		save_dir.mkdir(parents=True, exist_ok=True)
		out = save_dir / "bar_chart_mean_sd.png"
		fig.savefig(out, dpi=200, bbox_inches="tight")
		print(f"Saved: {out}")

	if show:
		import matplotlib.pyplot as plt

		plt.show()
	else:
		import matplotlib.pyplot as plt

		plt.close("all")


def _parse_families(
	family_args: Optional[List[List[str]]],
) -> List[Tuple[str, List[Path]]]:
	"""Parse `--family NAME csv...` repeated arguments."""
	if not family_args:
		raise ValueError("At least one --family is required")

	families: List[Tuple[str, List[Path]]] = []
	for item in family_args:
		if len(item) < 2:
			raise ValueError(
				"Each --family must be: --family NAME csv1.csv [csv2.csv ...]"
			)
		name = item[0]
		paths = [Path(p).expanduser() for p in item[1:]]
		families.append((name, paths))

	return families


def main(argv: Optional[List[str]] = None) -> None:
	"""CLI entry point."""
	parser = argparse.ArgumentParser(
		description="Compare mean±SD of metrics across controller families (multiple CSV runs)"
	)
	parser.add_argument(
		"--family",
		action="append",
		nargs="+",
		required=True,
		help="Repeatable. Usage: --family NAME csv1.csv csv2.csv ...",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Do not show the figure window (useful with --save-dir)",
	)
	parser.add_argument(
		"--save-dir",
		default=None,
		help="Directory where to save the PNG (optional)",
	)
	args = parser.parse_args(argv)

	families = _parse_families(args.family)
	for family_name, csv_paths in families:
		for p in csv_paths:
			if not p.exists():
				raise FileNotFoundError(
					f"CSV not found for family '{family_name}': {p}"
				)

	save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
	run(families=families, show=not args.no_show, save_dir=save_dir)


if __name__ == "__main__":
	main()
