from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
	bridge_node = Node(
		package = "worker_mission_control",
		executable = "megadrone_bridge_node",
		name = "megadrone_bridge_node",
		output = "screen",
		parameters = [{
			"megadrone_websocket_url": "ws://localhost:8765",
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
			"initial_takeoff_altitude": 15,
		}],
	)
	
	return LaunchDescription([
		bridge_node,
		mission_node,
	])