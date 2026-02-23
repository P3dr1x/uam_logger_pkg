"""Setuptools entrypoint for the uam_logger_pkg ROS2 Python package."""

from setuptools import setup


package_name = "uam_logger_pkg"


setup(
	name=package_name,
	version="0.0.0",
	packages=[package_name],
	data_files=[
		("share/ament_index/resource_index/packages", ["resource/" + package_name]),
		("share/" + package_name, ["package.xml"]),
		("share/" + package_name + "/launch", ["launch/uam_logger.launch.py"]),
	],
	install_requires=["setuptools"],
	zip_safe=True,
	maintainer="mattia",
	maintainer_email="mattia.pedrocco@phd.unipd.it",
	description="ROS2 logger package for deterministic SITL data export.",
	license="Apache License 2.0",
	tests_require=["pytest"],
	entry_points={
		"console_scripts": [
			"uam_logger_node = uam_logger_pkg.uam_logger_node:main",
			"uam_logger_offline_plot = uam_logger_pkg.offline_plotting:main",
			"uam_logger_animated_plot = uam_logger_pkg.animated_plot:main",
		],
	},
)
