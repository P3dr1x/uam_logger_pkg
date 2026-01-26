"""ROS2 node to deterministically log UAM simulation topics.

The logger is triggered by end-effector reference publications and records
multiple topics over a well-defined time window, then exports the data to CSV.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Accel, Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


try:
	from px4_msgs.msg import SensorCombined  # type: ignore

	HAVE_PX4_MSGS = True
except Exception:  # pragma: no cover
	SensorCombined = None  # type: ignore
	HAVE_PX4_MSGS = False


class LoggerState(str, Enum):
	"""Internal state machine states."""

	IDLE = "IDLE"
	RECORDING = "RECORDING"
	SAVING = "SAVING"


@dataclass
class LogRow:
	"""Single logged sample."""

	t_ns: int
	topic: str
	fields: Dict[str, Any]


def _clamp(x: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, x))


def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
	"""Convert quaternion (x,y,z,w) to roll, pitch, yaw (rad)."""
	# roll (x-axis rotation)
	sinr_cosp = 2.0 * (qw * qx + qy * qz)
	cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
	roll = math.atan2(sinr_cosp, cosr_cosp)

	# pitch (y-axis rotation)
	sinp = 2.0 * (qw * qy - qz * qx)
	sinp = _clamp(sinp, -1.0, 1.0)
	pitch = math.asin(sinp)

	# yaw (z-axis rotation)
	siny_cosp = 2.0 * (qw * qz + qx * qy)
	cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
	yaw = math.atan2(siny_cosp, cosy_cosp)

	return roll, pitch, yaw


def _join_seq(values: Sequence[Any]) -> str:
	return ";".join(str(v) for v in values)


class UamLoggerNode(Node):
	"""Experiment logger node."""

	def __init__(self) -> None:
		super().__init__("uam_logger_node")

		# Standard ROS2 parameter.
		# When launching, this is often already declared via parameter overrides.
		if not self.has_parameter("use_sim_time"):
			self.declare_parameter("use_sim_time", False)

		self.declare_parameter("experiment_name", "experiment")
		self.declare_parameter("output_dir", os.path.expanduser("~/.ros/uam_logger"))
		self.declare_parameter("reference_timeout_sec", 0.5)
		self.declare_parameter("watchdog_period_sec", 0.05)

		# Topics (overrideable)
		self.declare_parameter("topic_desired_ee_accel", "/desired_ee_accel")
		self.declare_parameter("topic_desired_ee_vel", "/desired_ee_vel")
		self.declare_parameter("topic_desired_ee_pose", "/desired_ee_global_pose")

		self.declare_parameter("topic_odometry", "/model/t960a_0/odometry")
		self.declare_parameter("topic_ee_pose", "/ee_world_pose")
		self.declare_parameter("topic_joint_states", "/joint_states")
		self.declare_parameter("topic_arm_commands", "/arm_controller/commands")
		self.declare_parameter("topic_sensor_combined", "/fmu/out/sensor_combined")

		self.experiment_name: str = self.get_parameter("experiment_name").value
		# Expand '~' so output goes to the real home directory.
		self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
		self.reference_timeout_sec: float = float(self.get_parameter("reference_timeout_sec").value)
		watchdog_period_sec: float = float(self.get_parameter("watchdog_period_sec").value)

		self.topic_desired_ee_accel: str = self.get_parameter("topic_desired_ee_accel").value
		self.topic_desired_ee_vel: str = self.get_parameter("topic_desired_ee_vel").value
		self.topic_desired_ee_pose: str = self.get_parameter("topic_desired_ee_pose").value

		self.topic_odometry: str = self.get_parameter("topic_odometry").value
		self.topic_ee_pose: str = self.get_parameter("topic_ee_pose").value
		self.topic_joint_states: str = self.get_parameter("topic_joint_states").value
		self.topic_arm_commands: str = self.get_parameter("topic_arm_commands").value
		self.topic_sensor_combined: str = self.get_parameter("topic_sensor_combined").value

		self.state: LoggerState = LoggerState.IDLE

		self._rows: List[LogRow] = []
		self._t_trigger_start_ns: Optional[int] = None
		self._t_first_desired_vel_ns: Optional[int] = None
		self._last_reference_ns: Optional[int] = None

		# Trigger subscriptions
		self.create_subscription(Accel, self.topic_desired_ee_accel, self._desired_accel_cb, 10)
		self.create_subscription(Twist, self.topic_desired_ee_vel, self._desired_vel_cb, 10)
		self.create_subscription(Pose, self.topic_desired_ee_pose, self._desired_pose_cb, 10)

		# Logged data subscriptions
		self.create_subscription(Odometry, self.topic_odometry, self._odometry_cb, 10)
		self.create_subscription(Pose, self.topic_ee_pose, self._ee_pose_cb, 10)
		self.create_subscription(JointState, self.topic_joint_states, self._joint_states_cb, 10)
		self.create_subscription(Float64MultiArray, self.topic_arm_commands, self._arm_commands_cb, 10)

		if HAVE_PX4_MSGS:
			self.create_subscription(  # type: ignore[arg-type]
				SensorCombined,
				self.topic_sensor_combined,
				self._sensor_combined_cb,  # type: ignore[arg-type]
				10,
			)
		else:
			self.get_logger().warn(
				"px4_msgs non disponibile: salto la sottoscrizione a /fmu/out/sensor_combined"
			)

		self._watchdog_timer = self.create_timer(watchdog_period_sec, self._watchdog_tick)

		self.get_logger().info(
			f"UAM logger ready (use_sim_time={self.get_parameter('use_sim_time').value}). "
			f"Trigger: {self.topic_desired_ee_accel} / {self.topic_desired_ee_vel} / {self.topic_desired_ee_pose}"
		)

	def _now_ns(self) -> int:
		return int(self.get_clock().now().nanoseconds)

	def _start_recording_if_needed(self, reference_topic: str) -> None:
		if self.state != LoggerState.IDLE:
			return

		now_ns = self._now_ns()
		self.state = LoggerState.RECORDING
		self._rows = []
		self._t_trigger_start_ns = now_ns
		self._t_first_desired_vel_ns = None
		self._last_reference_ns = now_ns

		self.get_logger().info(
			f"START RECORDING (trigger={reference_topic}, t_trigger_start_ns={now_ns})"
		)

	def _touch_reference(self) -> None:
		now_ns = self._now_ns()
		if self.state == LoggerState.IDLE:
			# Start on the first reference message (priority is handled by user-side publishing)
			self._start_recording_if_needed("reference")
		self._last_reference_ns = now_ns

	def _append(self, topic: str, t_ns: int, fields: Dict[str, Any]) -> None:
		self._rows.append(LogRow(t_ns=t_ns, topic=topic, fields=fields))

	# ---------------------------- Trigger callbacks ----------------------------
	def _desired_accel_cb(self, msg: Accel) -> None:
		self._start_recording_if_needed(self.topic_desired_ee_accel)
		self._last_reference_ns = self._now_ns()

		if self.state != LoggerState.RECORDING:
			return
		t_ns = self._now_ns()
		self._append(
			topic=self.topic_desired_ee_accel,
			t_ns=t_ns,
			fields={
				"lin_x": msg.linear.x,
				"lin_y": msg.linear.y,
				"lin_z": msg.linear.z,
				"ang_x": msg.angular.x,
				"ang_y": msg.angular.y,
				"ang_z": msg.angular.z,
			},
		)

	def _desired_vel_cb(self, msg: Twist) -> None:
		self._start_recording_if_needed(self.topic_desired_ee_vel)
		now_ns = self._now_ns()
		self._last_reference_ns = now_ns
		if self._t_first_desired_vel_ns is None:
			self._t_first_desired_vel_ns = now_ns

		if self.state != LoggerState.RECORDING:
			return
		self._append(
			topic=self.topic_desired_ee_vel,
			t_ns=now_ns,
			fields={
				"lin_x": msg.linear.x,
				"lin_y": msg.linear.y,
				"lin_z": msg.linear.z,
				"ang_x": msg.angular.x,
				"ang_y": msg.angular.y,
				"ang_z": msg.angular.z,
			},
		)

	def _desired_pose_cb(self, msg: Pose) -> None:
		self._start_recording_if_needed(self.topic_desired_ee_pose)
		now_ns = self._now_ns()
		self._last_reference_ns = now_ns

		if self.state != LoggerState.RECORDING:
			return
		self._append(
			topic=self.topic_desired_ee_pose,
			t_ns=now_ns,
			fields={
				"px": msg.position.x,
				"py": msg.position.y,
				"pz": msg.position.z,
				"qx": msg.orientation.x,
				"qy": msg.orientation.y,
				"qz": msg.orientation.z,
				"qw": msg.orientation.w,
			},
		)

	# ---------------------------- Data callbacks ----------------------------
	def _odometry_cb(self, msg: Odometry) -> None:
		if self.state != LoggerState.RECORDING:
			return

		t_ns = self._now_ns()
		p = msg.pose.pose.position
		q = msg.pose.pose.orientation
		roll, pitch, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
		vlin = msg.twist.twist.linear
		vang = msg.twist.twist.angular

		self._append(
			topic=self.topic_odometry,
			t_ns=t_ns,
			fields={
				"x": p.x,
				"y": p.y,
				"z": p.z,
				"roll": roll,
				"pitch": pitch,
				"yaw": yaw,
				"vx": vlin.x,
				"vy": vlin.y,
				"vz": vlin.z,
				"wx": vang.x,
				"wy": vang.y,
				"wz": vang.z,
			},
		)

	def _ee_pose_cb(self, msg: Pose) -> None:
		if self.state != LoggerState.RECORDING:
			return
		t_ns = self._now_ns()
		self._append(
			topic=self.topic_ee_pose,
			t_ns=t_ns,
			fields={
				"px": msg.position.x,
				"py": msg.position.y,
				"pz": msg.position.z,
				"qx": msg.orientation.x,
				"qy": msg.orientation.y,
				"qz": msg.orientation.z,
				"qw": msg.orientation.w,
			},
		)

	def _joint_states_cb(self, msg: JointState) -> None:
		if self.state != LoggerState.RECORDING:
			return
		t_ns = self._now_ns()
		self._append(
			topic=self.topic_joint_states,
			t_ns=t_ns,
			fields={
				"name": _join_seq(msg.name),
				"position": _join_seq(msg.position),
				"velocity": _join_seq(msg.velocity),
				"effort": _join_seq(msg.effort),
			},
		)

	def _arm_commands_cb(self, msg: Float64MultiArray) -> None:
		if self.state != LoggerState.RECORDING:
			return
		t_ns = self._now_ns()
		layout_dim = []
		try:
			for d in msg.layout.dim:
				layout_dim.append(f"{d.label}:{d.size}:{d.stride}")
		except Exception:
			layout_dim = []
		self._append(
			topic=self.topic_arm_commands,
			t_ns=t_ns,
			fields={
				"data": _join_seq(msg.data),
				"layout_dim": _join_seq(layout_dim),
				"layout_data_offset": int(getattr(msg.layout, "data_offset", 0)),
			},
		)

	def _sensor_combined_cb(self, msg: Any) -> None:
		if self.state != LoggerState.RECORDING:
			return
		t_ns = self._now_ns()

		# Field name can vary across px4_msgs versions; handle defensively.
		accel = None
		if hasattr(msg, "accelerometer_m_s2"):
			accel = getattr(msg, "accelerometer_m_s2")
		elif hasattr(msg, "accel_m_s2"):
			accel = getattr(msg, "accel_m_s2")

		if accel is None:
			return

		ax = float(accel[0])
		ay = float(accel[1])
		az = float(accel[2])

		self._append(
			topic=self.topic_sensor_combined,
			t_ns=t_ns,
			fields={
				"ax": ax,
				"ay": ay,
				"az": az,
			},
		)

	# ---------------------------- State logic ----------------------------
	def _watchdog_tick(self) -> None:
		if self.state != LoggerState.RECORDING:
			return
		if self._last_reference_ns is None:
			return

		now_ns = self._now_ns()
		dt = (now_ns - self._last_reference_ns) / 1e9
		if dt <= self.reference_timeout_sec:
			return

		self.state = LoggerState.SAVING
		try:
			self._save_experiment()
		finally:
			self._reset_to_idle()

	def _reset_to_idle(self) -> None:
		self.state = LoggerState.IDLE
		self._rows = []
		self._t_trigger_start_ns = None
		self._t_first_desired_vel_ns = None
		self._last_reference_ns = None
		self.get_logger().info("IDLE")

	def _select_time_zero_ns(self) -> Optional[int]:
		# Requirement preference: t=0 on first /desired_ee_vel if available.
		if self._t_first_desired_vel_ns is not None:
			return self._t_first_desired_vel_ns
		return self._t_trigger_start_ns

	def _save_experiment(self) -> None:
		t_zero_ns = self._select_time_zero_ns()
		if t_zero_ns is None:
			self.get_logger().warn("Nessun t0 disponibile: niente da salvare")
			return
		if not self._rows:
			self.get_logger().warn("Buffer vuoto: niente da salvare")
			return

		self.output_dir.mkdir(parents=True, exist_ok=True)
		stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		filename = f"{self.experiment_name}_{stamp}.csv"
		out_path = (self.output_dir / filename)
		out_path_abs = out_path.resolve()

		# Build a consistent header across all rows.
		all_keys: List[str] = []
		seen = set()
		for row in self._rows:
			for k in row.fields.keys():
				if k not in seen:
					seen.add(k)
					all_keys.append(k)

		header = ["t", "topic"] + all_keys

		with out_path.open("w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=header)
			writer.writeheader()
			for row in self._rows:
				t = (row.t_ns - t_zero_ns) / 1e9
				out: Dict[str, Any] = {"t": f"{t:.9f}", "topic": row.topic}
				out.update(row.fields)
				writer.writerow(out)

		self.get_logger().info(
			f"SAVED {len(self._rows)} samples to {out_path_abs} (t0_ns={t_zero_ns})"
		)


def main(args: Optional[Sequence[str]] = None) -> None:
	rclpy.init(args=args)
	node = UamLoggerNode()
	try:
		rclpy.spin(node)
	finally:
		node.destroy_node()
		rclpy.shutdown()

