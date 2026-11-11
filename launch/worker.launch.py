from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
	target_lat_arg = DeclareLaunchArgument(
		"target_lat",
		default_value = "0.0",
		description = "latitude of the gps target in degrees",
	)

	target_lon_arg = DeclareLaunchArgument(
		"target_lon",
		default_value = "0.0",
		description = "longitude of the gps target in degrees",
	)

	bridge_node = Node(
		package = "worker_mission_control",
		executable = "megadrone_bridge_node",
		name = "megadrone_bridge_node",
		output = "screen",
		parameters = [{
			"megadrone_websocket_url": "wss://ea20-76-78-191-9.ngrok-free.app",
			"megadrone_reconnect_delay": 5.0,
		}],
	)

	mission_node = Node(
		package = "worker_mission_control",
		executable = "mission_node",
		name = "mission_node",
		output = "screen",
		parameters = [{
			"mission_phase": 0,
			"initial_takeoff_altitude": 3,

			"target_lat": LaunchConfiguration("target_lat"),
			"target_lon": LaunchConfiguration("target_lon"),
		}],
	)

	return LaunchDescription([
		target_lat_arg,
		target_lon_arg,

		bridge_node,
		mission_node,
	])