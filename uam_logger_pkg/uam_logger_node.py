"""ROS2 node to deterministically log UAM simulation topics.

The logger is triggered by end-effector reference publications and records
multiple topics over a well-defined time window, then exports the data to CSV.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

from geometry_msgs.msg import Accel, Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


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
		# Controller param recording
		self.declare_parameter("controller_node", "/clik_uam_node")
		self.declare_parameter("controller_param_query_timeout_sec", 2.0)
		# Downsampling: log 1 sample every N messages (per topic)
		# Example: 5 -> keep 1 out of 5
		self.declare_parameter("log_every_n", 5)

		# Topics (overrideable)
		self.declare_parameter("topic_desired_ee_accel", "/desired_ee_accel")
		self.declare_parameter("topic_desired_ee_vel", "/desired_ee_vel")
		self.declare_parameter("topic_desired_ee_pose", "/desired_ee_global_pose")

		self.declare_parameter("topic_odometry", "/model/t960a_0/odometry")
		# Real system (mocap): pose from motion capture (typically natnet_ros2)
		self.declare_parameter("topic_mocap_pose", "/t960a/pose")
		self.declare_parameter("topic_ee_pose", "/ee_world_pose")
		self.declare_parameter("topic_joint_states", "/joint_states")
		self.declare_parameter("topic_sensor_combined", "/fmu/out/sensor_combined")
		self.declare_parameter("topic_real_t960a_twist", "/real_t960a_twist")

		self.experiment_name: str = self.get_parameter("experiment_name").value
		# Expand '~' so output goes to the real home directory.
		self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
		self.reference_timeout_sec: float = float(self.get_parameter("reference_timeout_sec").value)
		watchdog_period_sec: float = float(self.get_parameter("watchdog_period_sec").value)
		self.controller_node: str = str(self.get_parameter("controller_node").value)
		self.controller_param_query_timeout_sec: float = float(
			self.get_parameter("controller_param_query_timeout_sec").value
		)
		self.log_every_n: int = int(self.get_parameter("log_every_n").value)
		if self.log_every_n < 1:
			self.log_every_n = 1

		self.topic_desired_ee_accel: str = self.get_parameter("topic_desired_ee_accel").value
		self.topic_desired_ee_vel: str = self.get_parameter("topic_desired_ee_vel").value
		self.topic_desired_ee_pose: str = self.get_parameter("topic_desired_ee_pose").value

		self.topic_odometry: str = self.get_parameter("topic_odometry").value
		self.topic_mocap_pose: str = self.get_parameter("topic_mocap_pose").value
		self.topic_ee_pose: str = self.get_parameter("topic_ee_pose").value
		self.topic_joint_states: str = self.get_parameter("topic_joint_states").value
		self.topic_sensor_combined: str = self.get_parameter("topic_sensor_combined").value
		self.topic_real_t960a_twist: str = self.get_parameter("topic_real_t960a_twist").value

		self.state: LoggerState = LoggerState.IDLE

		self._rows: List[LogRow] = []
		self._t_trigger_start_ns: Optional[int] = None
		self._t_first_desired_vel_ns: Optional[int] = None
		self._last_reference_ns: Optional[int] = None
		self._topic_msg_counts: Dict[str, int] = {}

		# Trigger subscriptions
		self.create_subscription(Accel, self.topic_desired_ee_accel, self._desired_accel_cb, 10)
		self.create_subscription(Twist, self.topic_desired_ee_vel, self._desired_vel_cb, 10)
		self.create_subscription(Pose, self.topic_desired_ee_pose, self._desired_pose_cb, 10)

		# Logged data subscriptions
		self.create_subscription(Odometry, self.topic_odometry, self._odometry_cb, 10)
		# Optional real-only topic; if not published, no messages will arrive.
		self.create_subscription(PoseStamped, self.topic_mocap_pose, self._mocap_pose_cb, 10)
		self.create_subscription(Pose, self.topic_ee_pose, self._ee_pose_cb, 10)
		self.create_subscription(JointState, self.topic_joint_states, self._joint_states_cb, 10)
		# Optional real-only topic; if not published, no messages will arrive.
		self.create_subscription(Twist, self.topic_real_t960a_twist, self._real_t960a_twist_cb, 10)

		if HAVE_PX4_MSGS:
			self.create_subscription(  # type: ignore[arg-type]
				SensorCombined,
				self.topic_sensor_combined,
				self._sensor_combined_cb,  # type: ignore[arg-type]
				qos_profile_sensor_data,
			)
		else:
			self.get_logger().warn(
				"px4_msgs non disponibile: salto la sottoscrizione a /fmu/out/sensor_combined"
			)

		self._watchdog_timer = self.create_timer(watchdog_period_sec, self._watchdog_tick)

		# Async controller param query state (used in SAVING).
		self._ctrl_param_client: Optional[Any] = None
		self._ctrl_param_future: Optional[Any] = None
		self._ctrl_param_query_started: bool = False
		self._ctrl_param_query_deadline_ns: Optional[int] = None
		self._ctrl_param_srv_name: Optional[str] = None
		self._ctrl_param_candidates: List[str] = []
		self._ctrl_param_candidate_idx: int = 0
		self._ctrl_param_name_idx: int = 0
		self._ctrl_param_values: Dict[str, Any] = {}
		self._ctrl_param_last_error: Optional[str] = None
		self._ctrl_param_names: List[str] = [
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
		self._pending_save_t_zero_ns: Optional[int] = None

		self.get_logger().info(
			f"UAM logger ready (use_sim_time={self.get_parameter('use_sim_time').value}). "
			f"Trigger: {self.topic_desired_ee_accel} / {self.topic_desired_ee_vel} / {self.topic_desired_ee_pose}"
		)

	def _controller_get_parameters_service(self) -> str:
		controller = (self.controller_node or "").strip()
		if not controller:
			controller = "/clik_uam_node"
		if not controller.startswith("/"):
			controller = "/" + controller
		controller = controller.rstrip("/")
		return f"{controller}/get_parameters"

	def _discover_controller_param_services(self) -> List[str]:
		"""Return candidate /<node>/get_parameters service names for clik_uam_node.

		This helps when the controller is launched under a namespace.
		"""
		candidates: List[str] = []
		try:
			names_and_ns = self.get_node_names_and_namespaces()
		except Exception:
			return candidates

		for name, ns in names_and_ns:
			if name != "clik_uam_node":
				continue
			ns_s = (ns or "").rstrip("/")
			full = f"{ns_s}/{name}" if ns_s else f"/{name}"
			full = full if full.startswith("/") else "/" + full
			candidates.append(f"{full}/get_parameters")
		return candidates

	def _parameter_value_to_python(self, value_msg: Any) -> Any:
		"""Convert rcl_interfaces/msg/ParameterValue to a plain Python value."""
		ptype = int(getattr(value_msg, "type", ParameterType.PARAMETER_NOT_SET))
		if ptype == ParameterType.PARAMETER_NOT_SET:
			return None
		if ptype == ParameterType.PARAMETER_BOOL:
			return bool(value_msg.bool_value)
		if ptype == ParameterType.PARAMETER_INTEGER:
			return int(value_msg.integer_value)
		if ptype == ParameterType.PARAMETER_DOUBLE:
			return float(value_msg.double_value)
		if ptype == ParameterType.PARAMETER_STRING:
			return str(value_msg.string_value)
		# Not expected for our controller params; keep it explicit.
		return None

	def _fetch_controller_parameters(self) -> Tuple[Dict[str, Any], Optional[str]]:
		"""Fetch selected parameters from the controller node via GetParameters.

		Returns:
			(params_dict, error_str)
		"""
		# NOTE: This method used to block using spin_until_future_complete().
		# It is kept for backward compatibility but is NOT used by the logger
		# state machine anymore (to avoid deadlocks when called inside callbacks).
		return {}, "disabled_blocking_fetch"

	def _reset_controller_param_query_state(self) -> None:
		self._ctrl_param_client = None
		self._ctrl_param_future = None
		self._ctrl_param_query_started = False
		self._ctrl_param_query_deadline_ns = None
		self._ctrl_param_srv_name = None
		self._ctrl_param_candidates = []
		self._ctrl_param_candidate_idx = 0
		self._ctrl_param_name_idx = 0
		self._ctrl_param_values = {}
		self._ctrl_param_last_error = None

	def _begin_controller_param_query(self, now_ns: int) -> None:
		self._reset_controller_param_query_state()
		# Build candidate list:
		# 1) configured controller_node
		# 2) autodetected nodes named clik_uam_node (namespaces)
		configured = self._controller_get_parameters_service()
		discovered = self._discover_controller_param_services()
		cands: List[str] = [configured] + [s for s in discovered if s != configured]
		self._ctrl_param_candidates = cands
		self._ctrl_param_candidate_idx = 0
		if self._ctrl_param_candidates:
			srv_name = self._ctrl_param_candidates[0]
			self._ctrl_param_srv_name = srv_name
			self._ctrl_param_client = self.create_client(GetParameters, srv_name)
		self._ctrl_param_name_idx = 0
		self._ctrl_param_values = {}
		self._ctrl_param_last_error = None
		timeout = max(0.0, float(self.controller_param_query_timeout_sec))
		self._ctrl_param_query_deadline_ns = now_ns + int(timeout * 1e9)

	def _advance_controller_param_candidate(self) -> bool:
		"""Move to next candidate service. Returns True if advanced."""
		if self._ctrl_param_candidate_idx + 1 >= len(self._ctrl_param_candidates):
			return False
		self._ctrl_param_candidate_idx += 1
		srv_name = self._ctrl_param_candidates[self._ctrl_param_candidate_idx]
		self._ctrl_param_srv_name = srv_name
		self._ctrl_param_client = self.create_client(GetParameters, srv_name)
		self._ctrl_param_future = None
		self._ctrl_param_query_started = False
		self._ctrl_param_name_idx = 0
		self._ctrl_param_values = {}
		self._ctrl_param_last_error = None
		return True

	def _try_start_controller_param_query(self) -> bool:
		# This starts (or restarts) a single-parameter GetParameters call.
		if self._ctrl_param_query_started:
			return True
		if self._ctrl_param_client is None:
			return False
		if self._ctrl_param_name_idx >= len(self._ctrl_param_names):
			return True
		# Non-blocking availability check.
		try:
			ready = bool(self._ctrl_param_client.wait_for_service(timeout_sec=0.0))
		except Exception:
			ready = False
		if not ready:
			return False

		req = GetParameters.Request()
		req.names = [self._ctrl_param_names[self._ctrl_param_name_idx]]
		self._ctrl_param_future = self._ctrl_param_client.call_async(req)
		self._ctrl_param_query_started = True
		return True

	def _consume_single_param_future(self) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
		"""Return (name, value, err) for the current in-flight single-param request."""
		if self._ctrl_param_future is None:
			return None, None, "no_future"
		if not self._ctrl_param_future.done():
			return None, None, "future_not_done"
		if self._ctrl_param_future.exception() is not None:
			return None, None, f"service_exception: {self._ctrl_param_future.exception()}"
		if self._ctrl_param_name_idx >= len(self._ctrl_param_names):
			return None, None, "name_idx_out_of_range"
		name = self._ctrl_param_names[self._ctrl_param_name_idx]

		resp = self._ctrl_param_future.result()
		if resp is None:
			return name, None, "empty_response"
		values = getattr(resp, "values", None)
		if values is None:
			return name, None, "malformed_response"
		# Per specifica ROS2, values dovrebbe avere la stessa lunghezza di req.names.
		# In pratica, alcuni nodi/implementazioni possono rispondere con lista vuota
		# quando il parametro non esiste/non è dichiarato. In tal caso, trattiamo il
		# valore come "non disponibile" (None) e continuiamo a loggare gli altri.
		if len(values) == 0:
			return name, None, None
		if len(values) != 1:
			return name, None, f"unexpected_values_len: {len(values)} != 1"
		return name, self._parameter_value_to_python(values[0]), None

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
		self._topic_msg_counts = {}

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
		# Downsampling: keep 1 sample every N messages (per topic).
		count = self._topic_msg_counts.get(topic, 0)
		keep = (count == 0) or (self.log_every_n == 1) or (count % self.log_every_n == 0)
		self._topic_msg_counts[topic] = count + 1
		if not keep:
			return
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

	def _mocap_pose_cb(self, msg: PoseStamped) -> None:
		"""Log drone pose coming from motion capture (/t960a/pose).

		We store both quaternion and RPY so offline_plotting can reuse the same
		base-drone plotting functions (which expect x/y/z + roll/pitch/yaw).
		"""
		if self.state != LoggerState.RECORDING:
			return

		t_ns = self._now_ns()
		p = msg.pose.position
		q = msg.pose.orientation
		roll, pitch, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
		self._append(
			topic=self.topic_mocap_pose,
			t_ns=t_ns,
			fields={
				"x": p.x,
				"y": p.y,
				"z": p.z,
				"qx": q.x,
				"qy": q.y,
				"qz": q.z,
				"qw": q.w,
				"roll": roll,
				"pitch": pitch,
				"yaw": yaw,
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

		gyro = None
		if hasattr(msg, "gyro_rad"):
			gyro = getattr(msg, "gyro_rad")
		elif hasattr(msg, "gyro"):
			gyro = getattr(msg, "gyro")

		if accel is None:
			return

		ax = float(accel[0])
		ay = float(accel[1])
		az = float(accel[2])

		gx = gy = gz = None
		if gyro is not None:
			try:
				gx = float(gyro[0])
				gy = float(gyro[1])
				gz = float(gyro[2])
			except Exception:
				gx = gy = gz = None

		self._append(
			topic=self.topic_sensor_combined,
			t_ns=t_ns,
			fields={
				"ax": ax,
				"ay": ay,
				"az": az,
				"gx": gx,
				"gy": gy,
				"gz": gz,
			},
		)

	def _real_t960a_twist_cb(self, msg: Twist) -> None:
		if self.state != LoggerState.RECORDING:
			return
		t_ns = self._now_ns()
		self._append(
			topic=self.topic_real_t960a_twist,
			t_ns=t_ns,
			fields={
				"vx": msg.linear.x,
				"vy": msg.linear.y,
				"vz": msg.linear.z,
				"wx": msg.angular.x,
				"wy": msg.angular.y,
				"wz": msg.angular.z,
			},
		)

	# ---------------------------- State logic ----------------------------
	def _watchdog_tick(self) -> None:
		now_ns = self._now_ns()

		if self.state == LoggerState.RECORDING:
			if self._last_reference_ns is None:
				return

			dt = (now_ns - self._last_reference_ns) / 1e9
			if dt <= self.reference_timeout_sec:
				return

			# Transition to SAVING. Do not block here.
			self.state = LoggerState.SAVING
			self._pending_save_t_zero_ns = self._select_time_zero_ns()
			self._begin_controller_param_query(now_ns)
			return

		if self.state != LoggerState.SAVING:
			return

		# In SAVING: try starting the param query (service might come up late).
		if (self._ctrl_param_client is None) and self._ctrl_param_candidates:
			# Defensive: ensure a client exists.
			srv_name = self._ctrl_param_candidates[self._ctrl_param_candidate_idx]
			self._ctrl_param_srv_name = srv_name
			self._ctrl_param_client = self.create_client(GetParameters, srv_name)
		if not self._ctrl_param_query_started:
			self._try_start_controller_param_query()

		deadline_ns = self._ctrl_param_query_deadline_ns
		timed_out = (deadline_ns is not None) and (now_ns >= deadline_ns)
		future_done = (self._ctrl_param_future is not None) and bool(self._ctrl_param_future.done())

		if future_done:
			name, val, err = self._consume_single_param_future()
			if err is None and name is not None:
				self._ctrl_param_values[name] = val
				self._ctrl_param_name_idx += 1
				# Start next parameter immediately (non-blocking).
				self._ctrl_param_future = None
				self._ctrl_param_query_started = False
				if self._ctrl_param_name_idx < len(self._ctrl_param_names):
					self._try_start_controller_param_query()
			else:
				self._ctrl_param_last_error = err
				# If response is unusable, try next candidate service (if any).
				if err is not None and err.startswith("unexpected_values_len"):
					if self._advance_controller_param_candidate():
						self._try_start_controller_param_query()

		all_done = self._ctrl_param_name_idx >= len(self._ctrl_param_names)
		if not (all_done or timed_out):
			return

		# Finalize save with either params or error.
		self._save_experiment()
		self._reset_to_idle()

	def _reset_to_idle(self) -> None:
		self.state = LoggerState.IDLE
		self._rows = []
		self._t_trigger_start_ns = None
		self._t_first_desired_vel_ns = None
		self._last_reference_ns = None
		self._topic_msg_counts = {}
		self._pending_save_t_zero_ns = None
		self._reset_controller_param_query_state()
		self.get_logger().info("IDLE")

	def _select_time_zero_ns(self) -> Optional[int]:
		# Requirement preference: t=0 on first /desired_ee_vel if available.
		if self._t_first_desired_vel_ns is not None:
			return self._t_first_desired_vel_ns
		return self._t_trigger_start_ns

	def _save_experiment(self) -> None:
		t_zero_ns = self._pending_save_t_zero_ns if self._pending_save_t_zero_ns is not None else self._select_time_zero_ns()
		if t_zero_ns is None:
			self.get_logger().warn("Nessun t0 disponibile: niente da salvare")
			return
		if not self._rows:
			self.get_logger().warn("Buffer vuoto: niente da salvare")
			return

		# --- Parameter recording (controller) ---
		ctrl_params: Dict[str, Any] = dict(self._ctrl_param_values)
		ctrl_err: Optional[str]
		if self._ctrl_param_name_idx >= len(self._ctrl_param_names):
			ctrl_err = self._ctrl_param_last_error
		else:
			srv_name = self._ctrl_param_srv_name or self._controller_get_parameters_service()
			ctrl_err = self._ctrl_param_last_error or f"timeout_calling_service: {srv_name}"
		metadata_fields: Dict[str, Any] = {
			"controller_node": self.controller_node,
			"controller_params_service": self._ctrl_param_srv_name,
			"controller_params_json": json.dumps(ctrl_params, sort_keys=True),
			"controller_params_error": ctrl_err,
		}
		# Put metadata first in the CSV for easy inspection.
		self._rows.insert(
			0,
			LogRow(
				t_ns=t_zero_ns,
				topic="__metadata__/controller_params",
				fields=metadata_fields,
			),
		)

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
	except KeyboardInterrupt:
		# Normal shutdown path when user presses Ctrl+C.
		pass
	finally:
		try:
			node.destroy_node()
		except Exception:
			pass

		# Under ros2 launch the context may already be shutdown.
		try:
			if rclpy.ok():
				rclpy.shutdown()
		except Exception:
			pass

