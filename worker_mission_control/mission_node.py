import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.task import Future
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from std_srvs.srv import Trigger

from px4_msgs.msg import VehicleLocalPosition
from dexi_interfaces.srv import FlightCommand, LEDRingColor


class MissionNode(Node):
	def __init__(self):
		super().__init__("mission_node")

		self.declare_parameter("initial_takeoff_altitude", 10)
		self.declare_parameter("mission_phase", 0)
		self.add_on_set_parameters_callback(self.on_set_params)

		self.tick_rate = 0.1
		self.tick_waiters = []

		self.task = None
		self.phase = -1
		self.phase_requested = self.get_parameter("mission_phase").value
		self.flying = False

		self.go_received = False

		self.position_north = 0.0
		self.position_east = 0.0
		self.position_down = 0.0
		self.position_valid = False
		self.heading = 0.0

		self.goto_timeout = 60.0
		self.land_timeout = 60.0

		self.megadrone_lat = 0.0
		self.megadrone_lon = 0.0
		self.megadrone_alt = 0.0
		self.megadrone_fix_type = 0
		self.megadrone_satellites = 0
		self.megadrone_time = 0.0

		px4_qos = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.TRANSIENT_LOCAL,
			history=HistoryPolicy.KEEP_LAST,
			depth=1,
		)

		self.led_client = self.create_client(LEDRingColor, "/dexi/led_service/set_led_ring_color")
		self.flight_client = self.create_client(FlightCommand, "/dexi/flight/command")

		self.go_service = self.create_service(Trigger, "~/go", self.on_go)

		self.position_sub = self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.on_position, px4_qos)
		self.megadrone_sub = self.create_subscription(String, "/megadrone/incoming", self.on_megadrone_msg, 10)

		if not self.flight_client.wait_for_service(timeout_sec=10.0):
			raise RuntimeError("/dexi/flight/command is not available; is the offboard manager running?")

		self.tick_timer = self.create_timer(self.tick_rate, self.on_tick)


	def build_mission(self):
		alt = float(self.get_parameter("initial_takeoff_altitude").value)

		return [
			("confirm arm", lambda: self.wait_for_go("ready to arm and take off")),
			("takeoff", lambda: self.takeoff(alt)),
			("confirm fly out", lambda: self.wait_for_go("hovering, ready to fly out")),
			("fly out", lambda: self.fly_to(10.0, 0.0, -alt)),
			("confirm land", lambda: self.wait_for_go("at target, ready to land")),
			("land", lambda: self.land()),
			("confirm second takeoff", lambda: self.wait_for_go("landed, ready to take off again")),
			("takeoff", lambda: self.takeoff(alt)),
			("confirm fly home", lambda: self.wait_for_go("hovering, ready to fly home")),
			("fly home", lambda: self.fly_to(0.0, 0.0, -alt)),
			("confirm final land", lambda: self.wait_for_go("home, ready to land")),
			("land", lambda: self.land()),
		]

	def on_tick(self):
		if self.phase_requested is not None:
			start = self.phase_requested
			self.phase_requested = None
			self.start_mission(start)

		waiters = self.tick_waiters
		self.tick_waiters = []

		for future in waiters:
			future.set_result(None)

	def start_mission(self, start):
		if self.task is not None and not self.task.done():
			self.task.cancel()

		self.task = self.executor.create_task(self.run_mission, start)

	async def run_mission(self, start):
		mission = self.build_mission()

		try:
			for i in range(start, len(mission)):
				name, action = mission[i]

				self.phase = i
				self.get_logger().info(f"phase {i}: {name}")

				await action()
		except Exception as e:
			self.get_logger().error(f"phase {self.phase} failed: {e}")
			await self.abort()
			return

		self.phase = -1
		self.get_logger().info("mission complete")

	async def abort(self):
		self.phase = -1

		if self.flying:
			self.get_logger().warn("aborting, landing")

			if await self.flight("land", timeout=self.land_timeout, check=False):
				await self.flight("disarm", check=False)


	def now(self):
		return self.get_clock().now().nanoseconds / 1e9

	async def tick(self):
		future = Future()
		self.tick_waiters.append(future)
		await future

	async def sleep(self, seconds):
		end = self.now() + seconds

		while self.now() < end:
			await self.tick()

	async def wait_until(self, condition, timeout=None, name="condition"):
		start = self.now()

		while not condition():
			if timeout is not None and self.now() - start > timeout:
				raise RuntimeError(f"timed out waiting for {name}")

			await self.tick()

	async def wait_for_go(self, prompt="waiting for go"):
		self.go_received = False
		self.get_logger().info(f"{prompt}: ros2 service call /mission_node/go std_srvs/srv/Trigger")

		await self.wait_until(lambda: self.go_received)

	async def takeoff(self, alt):
		await self.flight("arm")
		await self.flight("offboard_takeoff", alt)

	async def land(self):
		await self.flight("land", timeout=self.land_timeout)
		await self.flight("disarm")

	async def fly_to(self, north, east, down, yaw=None, timeout=None):
		if north == 0.0 and east == 0.0 and down == 0.0:
			raise RuntimeError("goto_ned ignores an all-zero target; give a nonzero down")

		if yaw is None:
			yaw = math.degrees(self.heading)

		if timeout is None:
			timeout = self.goto_timeout

		await self.flight("goto_ned", north=north, east=east, down=down, yaw=yaw, timeout=timeout)

	async def set_led(self, color):
		if not self.led_client.service_is_ready():
			self.get_logger().warn("led service not ready, skipping")
			return

		request = LEDRingColor.Request()
		request.color = color

		result = await self.led_client.call_async(request)

		if result is not None:
			self.get_logger().info(f"led set: {result.message}")

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

	def on_go(self, request, response):
		self.go_received = True

		response.success = True
		response.message = f"go received at phase {self.phase}"

		return response

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

			mission_length = len(self.build_mission())

			if param.value < 0 or param.value >= mission_length:
				return SetParametersResult(successful=False, reason=f"phase {param.value} out of range (0 to {mission_length - 1})")

			self.phase_requested = param.value

		return SetParametersResult(successful=True)

	def on_megadrone_msg(self, msg):
		try:
			data = json.loads(msg.data)
		except json.JSONDecodeError as e:
			self.get_logger().error(f"bad megadrone json: {e}")
			return

		self.megadrone_lat = data.get("lat", self.megadrone_lat)
		self.megadrone_lon = data.get("lon", self.megadrone_lon)
		self.megadrone_alt = data.get("alt", self.megadrone_alt)
		self.megadrone_fix_type = data.get("fix_type", self.megadrone_fix_type)
		self.megadrone_satellites = data.get("satellites", self.megadrone_satellites)
		self.megadrone_time = data.get("time", self.megadrone_time)

		self.get_logger().info(str(data), throttle_duration_sec=1.0)


def main():
	rclpy.init()

	node = MissionNode()
	rclpy.spin(node)

	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main()