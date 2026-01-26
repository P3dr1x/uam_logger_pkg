# ROS2 Experiment Logger Package

## Context and Purpose

This package contains a ROS2 package whose purpose is to **log simulation data in a deterministic and reproducible way** during SITL experiments.

The logger is **triggered automatically** by the publication of end-effector trajectory references and records multiple ROS2 topics over a well-defined time window.
The recorded data are saved to disk for **offline analysis and plotting**, replacing manual tools such as PlotJuggler.

GitHub Copilot should generate code that follows the architectural and design constraints described below.

---

## High-Level Requirements

* Target middleware: **ROS2 (rclpy, Python)**
* Intended usage: **SITL simulations**
* The logger **must not control the system**, it only observes and records data
* Time alignment across all signals is critical
* The logger must work autonomously (no manual start/stop, just the main file is launched)

---

## Package Structure

Copilot should assume a standard ROS2 Python package structure:

```
uam_logger_pkg/
├── uam_logger_pkg/
│   ├── __init__.py
│   ├── uam_logger_node.py
├── launch/
│   └── uam_logger.launch.py
├── package.xml
├── setup.py
├── setup.cfg
└── resource/
    └── uam_logger_pkg
```

The main node must be implemented in **Python** and use `rclpy`.

---

## Core Node Responsibilities

The main node (e.g. `uam_logger_node.py`) must:

1. Subscribe to a **reference topic** used as a temporal trigger:
    1a. 
    * `/desired_ee_accel` (Message type: `geometry_msgs/msg/Accel`) if exists
    * `/desired_ee_vel` (Message type: `geometry_msgs/msg/Twist`)
    * `/desired_ee_global_pose` (Message type: `geometry_msgs/msg/Pose`)

2. Subscribe to **one or more data topics to be logged**, including:

   * `/model/t960a_0/odometry` (Message type: `nav_msgs/msg/Odometry`)
   * `/ee_world_pose` (Message type: `geometry_msgs/msg/Pose`)
   * `/joint_states` and `/arm_controller/commands` (Message type: `nsensor_msgs/msg/JointState` and `std_msgs/msg/Float64MultiArray`)
   * `/fmu/out/sensor_combined` if it is active (Message type: `px4_msgs/msg/SensorCombined`)


3. Automatically:

   * **Start recording** when the first reference message is received
   * **Stop recording** when no new reference messages are received for a configurable timeout

4. Use **simulation time** (`use_sim_time = true`) and the node clock for all timestamps.


---

## Quantities of interest to record

* From `/model/t960a_0/odometry` we are interested in the x,y,z displacement, roll, pitch and yaw angles and veocity components in the twist term.
* From the `ee_world_pose` topic we are interested in all the fields.
* From the `/joint_states` and `/arm_controller/commands` we are interested in all the fields
* From the `/fmu/out/sensor_combined` we are linear acceleration data from the simulated accelerometer

---

## Time Handling Rules

* All logged signals must share a **common time origin**
* Define `t = 0` as the time of the first `\desired_ee_vel` message
* All timestamps must be expressed as:

  ```
  t = (now - t_start) in seconds
  ```
* Do NOT rely on message header timestamps unless explicitly required

---

## Internal State Machine

The node must internally implement the following logical states:

* `IDLE`: waiting for reference messages
* `RECORDING`: actively logging data
* `SAVING`: exporting data to disk

Transitions:

* `IDLE → RECORDING`: first `\desired_ee_vel` message received
* `RECORDING → SAVING`: timeout exceeded since last reference
* `SAVING → IDLE`: data successfully written to disk

---

## Data Handling Guidelines

* Logged data must be stored in **in-memory buffers** during recording
* Use simple, explicit data structures:

  * Python lists
  * NumPy arrays
  * Dictionaries of time series
* Avoid complex abstractions or hidden side effects

Each logged quantity must be associated with a timestamp.

---

## Data Export Format

* Data must be saved **automatically** at the end of each experiment (which corresponds to the moment in which no more new messages arrive on the topic)
* Preferred formats:

  * CSV (if human readability is required)
* Each experiment should generate a **separate file**
* File naming should include:

  * experiment name
  * timestamp (date and time)

Example:

```
experiment_2026_01_23_15_42_10.csv
```

---

## Coding Style and Constraints

Copilot-generated code should:

* Follow ROS2 Python best practices (`rclpy`)
* Keep callbacks lightweight (no heavy logic inside callbacks)
* Separate:

  * subscription callbacks
  * state logic
  * data export logic
* Be deterministic and reproducible
* Avoid unnecessary dependencies

---

## Intended Usage Pattern

Typical workflow assumed by this package:

1. Launch SITL simulation
2. Run controller 
3. Run planner node publishing in `/desired_ee_vel`, `\desired_ee_global_pose` and possibily in `desired_ee_accel` 
3. Logger node records data automatically
4. Logger saves data on completion
5. User performs **offline plotting and analysis**

Copilot should generate code accordingly.

## Dependencies 

Since the node must register to topic where custom messages are published, it is important that `px4_msgs`, `interbotix_xs_msgs` and `px4_ros_com` are declared as dependendecies. Then identify by yourself which other dependencies are needed. 
