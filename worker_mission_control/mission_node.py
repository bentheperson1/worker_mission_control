import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from dexi_interfaces.srv import LEDRingColor
from std_msgs.msg import String

class MissionNode(Node):
	def __init__(self):
		super().__init__("mission_node")
		
		self.phase = 0
		self.phase_min = 1
		self.phase_max = 6
		
		self.declare_parameter("mission_phase", 0)
		self.add_on_set_parameters_callback(self.on_set_params)
		
		self.led_client = self.create_client(LEDRingColor, "/dexi/led_service/set_led_ring_color")
		self.mission_timer = self.create_timer(0.1, self.mission_step)

		self.megadrone_sub = self.create_subscription(String, "megadrone/incoming", self.on_megadrone_msg, 10)
	
	def on_megadrone_msg(self, msg):
		data = json.loads(msg.data)
		
		self.get_logger().info(data)

	def on_set_params(self, params):
		for param in params:
			if param.name != "mission_phase":
				continue
			
			if param.value < self.phase_min or param.value > self.phase_max:
				return SetParametersResult(successful = False, reason = f"phase {param.value} out of range")
							
			# if param.value != emergencyKill && self.phase.isComplete() == false:
				# return SetParametersResult(successful = False, reason = "Current phase not complete.")

			# Implement phase objects
			# 

			self.phase = param.value
			self.get_logger().info(f"phase set to {self.phase}")
			self.set_led_ring("blue")



		
		return SetParametersResult(successful = True, reason = f"now in phase {self.phase}")
	
	def mission_step(self):
		# runs 10x per second, put per-phase logic here later
		pass
	
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
	
	node = MissionNode()
	rclpy.spin(node)
	
	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main()