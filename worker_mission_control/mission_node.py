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
		
		self.declare_parameter("initial_takeoff_altitude", 10)

		self.declare_parameter("mission_phase", 0)
		self.add_on_set_parameters_callback(self.on_set_params)
		
		self.led_client = self.create_client(LEDRingColor, "/dexi/led_service/set_led_ring_color")
		self.mission_timer = self.create_timer(0.1, self.mission_step)

		self.megadrone_sub = self.create_subscription(String, "megadrone/incoming", self.on_megadrone_msg, 10)
	
		self.flight_client = self.create_client(FlightCommand, '/dexi/flight/command')
		
		if not self.flight_client.wait_for_service(timeout_sec=10.0):
			raise RuntimeError('/dexi/flight/command is not available; is the offboard manager running?')
	   
		self.flying = False

	def on_megadrone_msg(self, msg):
		data = json.loads(msg.data)
		
		self.get_logger().info(data)

	def on_set_params(self, params):
		for param in params:
			if param.name != "mission_phase":
				continue
			
			if param.value < self.phase_min or param.value > self.phase_max:
				return SetParametersResult(successful = False, reason = f"phase {param.value} out of range")
			
			self.phase = param.value
			self.get_logger().info(f"phase set to {self.phase}")
			
		return SetParametersResult(successful = True, reason = f"now in phase {self.phase}")
	
	def flight(self, command, value=0.0, timeout=30.0, north=0.0, east=0.0, down=0.0, yaw=0.0, check=True):
		request = FlightCommand.Request()
		request.command = command
		request.value = float(value)
		request.north = float(north)
		request.east = float(east)
		request.down = float(down)
		request.yaw = float(yaw)
		request.timeout = float(timeout)

		if command == 'offboard_takeoff':
			self.flying = True   # set before the call: a failed takeoff may still leave the ground
		
		future = self.flight_client.call_async(request)
		rclpy.spin_until_future_complete(self, future)
		result = future.result()

		if result is None or not result.success:
			message = result.message if result is not None else 'no response'
			self.get_logger().error(f"{command}: FAILED ({message})")

			if check:
				raise RuntimeError(f"{command} failed: {message}")
			return False

		shown = f" {value:g}" if value else ""
		self.get_logger().info(f"{command}{shown}: {result.message} ({result.execution_time:.1f} s)")
		
		if command == 'land':
			self.flying = False

		return True

	def mission_step(self):
		match phase:
			case 0:
				takeoff_alt = self.get_parameter("initial_takeoff_altitude").value

				self.flight("arm")
				self.flight("offboard_takeoff")
				

			case 1:
				pass
			
			case 2:
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