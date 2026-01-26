"""Launch file for the UAM experiment logger."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
	experiment_name_arg = DeclareLaunchArgument(
		"experiment_name",
		default_value="experiment",
		description="Prefix used for the output CSV filename.",
	)
	output_dir_arg = DeclareLaunchArgument(
		"output_dir",
		default_value="~/.ros/uam_logger",
		description="Directory where CSV logs are written.",
	)
	reference_timeout_arg = DeclareLaunchArgument(
		"reference_timeout_sec",
		default_value="0.5",
		description="Stop recording when no reference arrives for this time [s].",
	)

	node = Node(
		package="uam_logger_pkg",
		executable="uam_logger_node",
		name="uam_logger_node",
		output="screen",
		parameters=[
			{"use_sim_time": True},
			{"experiment_name": LaunchConfiguration("experiment_name")},
			{"output_dir": LaunchConfiguration("output_dir")},
			{"reference_timeout_sec": LaunchConfiguration("reference_timeout_sec")},
		],
	)

	return LaunchDescription([
		experiment_name_arg,
		output_dir_arg,
		reference_timeout_arg,
		node,
	])

