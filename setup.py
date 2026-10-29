import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'worker_mission_control'

setup(
	name = package_name,
	version = "0.0.1",
	packages = find_packages(exclude = ["test"]),
	data_files = [
		("share/ament_index/resource_index/packages", ["resource/" + package_name]),
		("share/" + package_name, ["package.xml"]),
		(os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
	],
	install_requires = ["setuptools"],
	zip_safe = True,
	entry_points = {
		"console_scripts": [
			"mission_node = worker_mission_control.mission_node:main",
			"megadrone_bridge_node = worker_mission_control.megadrone_bridge_node:main",
		],
	},
)