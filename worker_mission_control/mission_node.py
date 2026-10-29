# TODO
# Receive mission phase from ground control
# Match mission phase to entry point



# Phase 1: Take off, orient to target, and fly to target N & E
# Phase 2: Ensure we're within target radius, initiate land
# Phase 3: Ensure megadrone coordinates are stable for more than 5 seconds and is in a disarmed state, then initiate takeoff
# Phase 4: Fly to megadrone N & E
# Phase 5: Loop until we see the landing apriltag ID beneath the DEXI, once we see the landing apriltag ID for more than 5 seconds, allow phase 6
# Phase 6: Switch to apriltag landing mode and orient to apriltag, then land


import rclpy
from rclpy.node import Node

from mission_interfaces.msg import MissionPhase
from dexi_interfaces.srv import LEDRingColor


class WorkerNode(Node):
	def __init__(self):
		super().__init__("worker_node")
		
		self.phase_sub = self.create_subscription(MissionPhase, "/mission_phase", self.on_phase, 10)
		self.led_client = self.create_client(LEDRingColor, "/dexi/led_service/set_led_ring_color")
	
	def on_phase(self, msg):
		self.get_logger().info(f"phase {msg.phase} received: {msg.note}")
		self.set_led_ring("blue")
	
	def set_led_ring(self, color):
		if not self.led_client.service_is_ready():
			self.get_logger().warn("led service not ready, skipping")
			return
		
		request = LEDRingColor.Request()
		request.color = color
		
		self.led_client.call_async(request).add_done_callback(self.on_led_response)
	
	def on_led_response(self, future):
		try:
			self.get_logger().info(f"led set: {future.result().message}")
		except Exception as e:
			self.get_logger().error(f"led call failed: {e}")


def main():
	rclpy.init()
	
	node = WorkerNode()
	rclpy.spin(node)
	
	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main()