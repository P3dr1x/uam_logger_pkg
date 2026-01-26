# uam_logger_pkg

ROS2 (Humble) Python package (`rclpy`) to deterministically log UAM SITL data.

The logger is **fully autonomous**:

- it **starts recording** when end-effector trajectory reference messages begin
- it **stops recording** when reference messages stop for a configurable timeout
- it **exports a CSV** for offline plotting/analysis

All exported signals share the same time origin and experiment window.

## What it logs

Trigger (reference) topics (any of these can start the experiment):

- `/desired_ee_accel` (`geometry_msgs/msg/Accel`)
- `/desired_ee_vel` (`geometry_msgs/msg/Twist`)
- `/desired_ee_global_pose` (`geometry_msgs/msg/Pose`)

Logged data topics (during RECORDING):

- `/model/t960a_0/odometry` (`nav_msgs/msg/Odometry`): x,y,z, roll,pitch,yaw and twist (linear/angular velocity)
- `/ee_world_pose` (`geometry_msgs/msg/Pose`): all pose fields
- `/joint_states` (`sensor_msgs/msg/JointState`): all fields
- `/arm_controller/commands` (`std_msgs/msg/Float64MultiArray`): all fields
- `/fmu/out/sensor_combined` (`px4_msgs/msg/SensorCombined`) if available: accelerometer linear acceleration

## Time handling (alignment)

All timestamps are taken from the **node clock** (simulation time when `use_sim_time=true`).

- `t = (now - t0)` in seconds
- `t0` is **preferably** the time of the **first** `/desired_ee_vel` message received during the experiment
- if no `/desired_ee_vel` arrives, `t0` falls back to the time of the first received reference message

No message header timestamps are used.

## States

The node runs this internal state machine:

- `IDLE`: waiting for a reference message
- `RECORDING`: buffering samples in memory
- `SAVING`: exporting CSV and resetting

`RECORDING → SAVING` happens when no reference arrives for `reference_timeout_sec`.

## Output

At the end of each experiment, a CSV is automatically written to `output_dir` with filename:

`<experiment_name>_YYYY_MM_DD_HH_MM_SS.csv`

Each row includes:

- `t`: seconds since `t0` (string formatted with 9 decimals)
- `topic`: the ROS2 topic name
- topic-specific fields (columns are the union of all keys across rows; unused fields are empty)

## How to run

Build your workspace as usual (example):

```bash
colcon build --packages-select uam_logger_pkg
```

Launch (recommended, enables sim time by default):

```bash
ros2 launch uam_logger_pkg uam_logger.launch.py \
  experiment_name:=sitl_test \
  reference_timeout_sec:=0.5
```

Alternatively, run the node directly:

```bash
ros2 run uam_logger_pkg uam_logger_node --ros-args -p use_sim_time:=true
```

## Parameters

Common parameters:

- `experiment_name` (string): filename prefix
- `output_dir` (string): directory where CSV files are written (default: `~/.ros/uam_logger`)
- `reference_timeout_sec` (float): stop condition timeout

Topic remapping via parameters (advanced):

- `topic_desired_ee_accel`, `topic_desired_ee_vel`, `topic_desired_ee_pose`
- `topic_odometry`, `topic_ee_pose`, `topic_joint_states`, `topic_arm_commands`, `topic_sensor_combined`

## Notes

- For SITL, keep `use_sim_time=true` so timestamps follow the simulator clock.
- `px4_msgs` might not be available in every environment; in that case the node will warn and skip `/fmu/out/sensor_combined`.
