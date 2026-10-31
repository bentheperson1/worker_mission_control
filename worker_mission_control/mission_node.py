import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String

from px4_msgs.msg import VehicleLocalPosition
from dexi_interfaces.msg import OffboardNavCommand
from dexi_interfaces.srv import FlightCommand, LEDRingColor


class MissionNode(Node):
	def __init__(self):
		super().__init__("mission_node")

		self.declare_parameter("initial_takeoff_altitude", 10)
		self.declare_parameter("mission_phase", 0)
		self.add_on_set_parameters_callback(self.on_set_params)

		self.phase = 0
		self.phase_min = 0
		self.phase_max = 2
		self.phase_entered = False
		self.phase_requested = None
		self.phase_start_time = None

		self.task = None
		self.flying = False

		self.position_north = 0.0
		self.position_east = 0.0
		self.position_down = 0.0
		self.position_valid = False
		self.heading = 0.0

		self.target_north = 0.0
		self.target_east = 0.0
		self.target_down = 0.0
		self.target_tolerance = 0.5

		self.nav_gain = 0.6
		self.nav_max_speed_h = 2.0
		self.nav_max_speed_v = 1.0
		self.nav_timeout = 60.0

		self.arrive_ticks = 0
		self.arrive_ticks_needed = 10

		# phase: (async enter or None, sync step or None)
		self.phases = {
			0: (self.enter_takeoff, None),
			1: (self.enter_fly_to_point, self.step_fly_to_point),
			2: (self.enter_land, None),
		}

		px4_qos = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.TRANSIENT_LOCAL,
			history=HistoryPolicy.KEEP_LAST,
			depth=1,
		)

		self.led_client = self.create_client(LEDRingColor, "/dexi/led_service/set_led_ring_color")
		self.flight_client = self.create_client(FlightCommand, "/dexi/flight/command")

		# replace with the topic your offboard manager listens on
		self.offboard_publisher = self.create_publisher(OffboardNavCommand, "/dexi/offboard_nav_command", 10)

		self.position_sub = self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.on_position, px4_qos)
		self.megadrone_sub = self.create_subscription(String, "megadrone/incoming", self.on_megadrone_msg, 10)

		if not self.flight_client.wait_for_service(timeout_sec=10.0):
			raise RuntimeError("/dexi/flight/command is not available; is the offboard manager running?")

		self.mission_timer = self.create_timer(0.1, self.mission_step)

	# ---------- core loop (sync, never blocks) ----------

	def mission_step(self):
		if self.phase_requested is not None:
			if self.task is not None:
				self.task.cancel()
				self.task = None

			self.set_phase(self.phase_requested)
			self.phase_requested = None

		if self.task is not None:
			if not self.task.done():
				return

			error = self.task.exception()
			self.task = None

			if error is not None:
				self.get_logger().error(f"phase {self.phase} failed: {error}")
				self.abort()
				return

		if self.phase not in self.phases:
			return

		on_enter, on_step = self.phases[self.phase]

		if not self.phase_entered:
			self.phase_entered = True
			self.phase_start_time = self.get_clock().now()

			if on_enter is not None:
				self.task = self.executor.create_task(on_enter)
				return

		if on_step is None or on_step():
			self.set_phase(self.phase + 1)

	def set_phase(self, phase):
		self.phase = phase
		self.phase_entered = False
		self.get_logger().info(f"entering phase {self.phase}")

	def abort(self):
		self.phase = -1
		self.phase_entered = False

		if self.flying:
			self.task = self.executor.create_task(self.flight, "land", check=False)

	def phase_elapsed(self):
		return (self.get_clock().now() - self.phase_start_time).nanoseconds / 1e9

	# ---------- phases ----------

	async def enter_takeoff(self):
		takeoff_alt = self.get_parameter("initial_takeoff_altitude").value

		await self.flight("arm")
		await self.flight("offboard_takeoff", takeoff_alt)

	async def enter_fly_to_point(self):
		self.target_north = 10.0
		self.target_east = 0.0
		self.target_down = -self.get_parameter("initial_takeoff_altitude").value

		self.arrive_ticks = 0

	def step_fly_to_point(self):
		if not self.position_valid:
			self.get_logger().warn("no valid position yet, holding", throttle_duration_sec=1.0)
			self.send_velocity_body(0.0, 0.0, 0.0)
			return False

		elapsed = self.phase_elapsed()

		if elapsed > self.nav_timeout:
			self.get_logger().error(f"fly_to_point timed out after {elapsed:.1f} s")
			self.send_velocity_body(0.0, 0.0, 0.0)
			self.abort()
			return False

		dn = self.target_north - self.position_north
		de = self.target_east - self.position_east
		dd = self.target_down - self.position_down

		vn = dn * self.nav_gain
		ve = de * self.nav_gain
		vd = dd * self.nav_gain

		speed_h = math.hypot(vn, ve)

		if speed_h > self.nav_max_speed_h:
			scale = self.nav_max_speed_h / speed_h
			vn *= scale
			ve *= scale

		vd = max(-self.nav_max_speed_v, min(vd, self.nav_max_speed_v))

		cos_h = math.cos(self.heading)
		sin_h = math.sin(self.heading)

		forward = vn * cos_h + ve * sin_h
		right = -vn * sin_h + ve * cos_h

		if self.distance_to_target() < self.target_tolerance:
			self.arrive_ticks += 1
		else:
			self.arrive_ticks = 0

		arrived = self.arrive_ticks >= self.arrive_ticks_needed

		if arrived:
			forward, right, vd = 0.0, 0.0, 0.0
			self.get_logger().info(f"arrived at ({self.target_north:.1f}, {self.target_east:.1f}, {self.target_down:.1f})")

		self.send_velocity_body(forward, right, vd)

		return arrived

	async def enter_land(self):
		await self.flight("land")

	# ---------- helpers ----------

	def distance_to_target(self):
		dn = self.target_north - self.position_north
		de = self.target_east - self.position_east
		dd = self.target_down - self.position_down

		return math.sqrt(dn * dn + de * de + dd * dd)

	def send_velocity_body(self, forward, right, down, yaw_rate=0.0):
		msg = OffboardNavCommand()

		msg.command = "set_velocity_body"
		msg.north = float(forward)
		msg.east = float(right)
		msg.down = float(down)
		msg.yaw = float(yaw_rate)

		self.offboard_publisher.publish(msg)

	async def flight(self, command, value=0.0, timeout=30.0, north=0.0, east=0.0, down=0.0, yaw=0.0, check=True):
		request = FlightCommand.Request()
		request.command = command
		request.value = float(value)
		request.north = float(north)
		request.east = float(east)
		request.down = float(down)
		request.yaw = float(yaw)
		request.timeout = float(timeout)

		if command == "offboard_takeoff":
			self.flying = True   # set before the call: a failed takeoff may still leave the ground

		result = await self.flight_client.call_async(request)

		if result is None or not result.success:
			message = result.message if result is not None else "no response"
			self.get_logger().error(f"{command}: FAILED ({message})")

			if check:
				raise RuntimeError(f"{command} failed: {message}")

			return False

		shown = f" {value:g}" if value else ""
		self.get_logger().info(f"{command}{shown}: {result.message} ({result.execution_time:.1f} s)")

		if command == "land":
			self.flying = False

		return True

	# ---------- callbacks ----------

	def on_position(self, msg):
		self.position_north = msg.x
		self.position_east = msg.y
		self.position_down = msg.z
		self.heading = msg.heading

		self.position_valid = msg.xy_valid and msg.z_valid

	def on_set_params(self, params):
		for param in params:
			if param.name != "mission_phase":
				continue

			if param.value < self.phase_min or param.value > self.phase_max:
				return SetParametersResult(successful=False, reason=f"phase {param.value} out of range")

			self.phase_requested = param.value

		return SetParametersResult(successful=True)

	def on_megadrone_msg(self, msg):
		data = json.loads(msg.data)

		self.get_logger().info(str(data))

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