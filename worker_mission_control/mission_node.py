import json
import math
import inspect
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.task import Future
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from std_srvs.srv import Trigger

from px4_msgs.msg import VehicleLocalPosition
from dexi_interfaces.srv import FlightCommand, LEDRingColor
from dexi_interfaces.msg import OffboardNavCommand


EARTH_RADIUS = 6371000.0


class MissionNode(Node):
	def __init__(self):
		super().__init__("mission_node")

		self.declare_parameter("initial_takeoff_altitude", 10)
		self.declare_parameter("mission_phase", 0)
		self.declare_parameter("target_lat", 0.0)
		self.declare_parameter("target_lon", 0.0)
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

		self.ref_lat = 0.0
		self.ref_lon = 0.0
		self.ref_alt = 0.0
		self.ref_valid = False

		self.goto_timeout = 60.0
		self.land_timeout = 60.0
		self.ref_timeout = 10.0

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

		self.velocity_publisher = self.create_publisher(OffboardNavCommand, "/dexi/offboard_manager", px4_qos)

		self.led_client = self.create_client(LEDRingColor, "/dexi/led_service/set_led_ring_color")
		self.flight_client = self.create_client(FlightCommand, "/dexi/flight/command")

		self.go_service = self.create_service(Trigger, "~/go", self.on_go)

		self.position_sub = self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.on_position, px4_qos)
		self.megadrone_sub = self.create_subscription(String, "/megadrone/incoming", self.on_megadrone_msg, 10)

		self.megadrone_message_publisher = self.create_publisher(String, "/megadrone/send", 10)

		if not self.flight_client.wait_for_service(timeout_sec=10.0):
			raise RuntimeError("/dexi/flight/command is not available; is the offboard manager running?")

		self.tick_timer = self.create_timer(self.tick_rate, self.on_tick)

	def send_megadrone_message(self, message):
		msg = String()
		msg.data = message
		self.megadrone_message_publisher.publish(msg)

	def set_hatch(self, is_open):
		self.send_megadrone_message(json.dumps({"open_hatch": is_open}))

	def build_mission(self):
		alt = float(self.get_parameter("initial_takeoff_altitude").value)

		target_lat = float(self.get_parameter("target_lat").value)
		target_lon = float(self.get_parameter("target_lon").value)

		return [
			("open hatch", lambda: self.set_hatch(True)),

			("confirm arm", lambda: self.wait_for_go("ready to arm and take off")),
			("takeoff", lambda: self.takeoff(alt)),

			("confirm fly out", lambda: self.wait_for_go(f"hovering, ready to fly to ({target_lat:.7f}, {target_lon:.7f})")),
			("close hatch", lambda: self.set_hatch(False)),
			#("fly out", lambda: self.fly_to_gps(target_lat, target_lon, alt)),
			("fly out", lambda: self.fly_to(10.0, 0, 0, 0, 5.0)),

			("confirm land", lambda: self.wait_for_go("at target, ready to land")),
			("land", lambda: self.land()),

			("confirm second takeoff", lambda: self.wait_for_go("landed, ready to take off again")),
			("takeoff", lambda: self.takeoff(alt)),

			("confirm fly home", lambda: self.wait_for_go("hovering, ready to fly home")),
			("fly home", lambda: self.fly_to_gps(0.0, 0.0, -alt)),
			("open hatch", lambda: self.set_hatch(True)),

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
				
				result = action()
				
				if inspect.isawaitable(result):
					await result
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

	def gps_to_ned(self, lat, lon):
		lat_rad = math.radians(lat)
		lon_rad = math.radians(lon)
		ref_lat_rad = math.radians(self.ref_lat)
		ref_lon_rad = math.radians(self.ref_lon)

		sin_lat = math.sin(lat_rad)
		cos_lat = math.cos(lat_rad)
		ref_sin_lat = math.sin(ref_lat_rad)
		ref_cos_lat = math.cos(ref_lat_rad)
		cos_d_lon = math.cos(lon_rad - ref_lon_rad)

		arg = max(-1.0, min(1.0, ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon))
		c = math.acos(arg)
		k = 1.0 if abs(c) < 1e-9 else c / math.sin(c)

		north = k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * EARTH_RADIUS
		east = k * cos_lat * math.sin(lon_rad - ref_lon_rad) * EARTH_RADIUS

		return north, east

	async def fly_to_gps(self, lat, lon, alt=None, alt_relative=True, yaw=None, timeout=None):
		if lat == 0.0 and lon == 0.0:
			raise RuntimeError("gps target is (0, 0); set target_lat and target_lon")

		await self.wait_until(lambda: self.ref_valid, timeout=self.ref_timeout, name="global position reference")

		north, east = self.gps_to_ned(lat, lon)

		if alt is None:
			down = self.position_down
		elif alt_relative:
			down = -alt
		else:
			down = -(alt - self.ref_alt)

		self.get_logger().info(f"gps ({lat:.7f}, {lon:.7f}) -> ned ({north:.1f}, {east:.1f}, {down:.1f})")

		await self.fly_to(north, east, down, yaw, timeout)

	async def fly_to_megadrone(self, alt):
		if self.megadrone_fix_type < 3:
			raise RuntimeError(f"megadrone has no 3d fix (fix_type {self.megadrone_fix_type})")

		await self.fly_to_gps(self.megadrone_lat, self.megadrone_lon, alt)

	async def fly_velocity(self, forward, right, down, yaw_rate, duration):
		self.get_logger().info(f"velocity: forward={forward} right={right} down={down} yaw_rate={yaw_rate:.1f} for {duration:.1f}s")

		msg = OffboardNavCommand()
		msg.command = "set_velocity_body"
		msg.north = float(forward)
		msg.east = float(right)
		msg.down = float(down)
		msg.yaw = float(yaw_rate)

		end = self.now() + duration

		while self.now() < end:
			self.velocity_publisher.publish(msg)
			await self.tick()

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

		self.ref_lat = msg.ref_lat
		self.ref_lon = msg.ref_lon
		self.ref_alt = msg.ref_alt
		self.ref_valid = msg.xy_global and msg.z_global

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

		self.get_logger().info(str(data), throttle_duration_sec=5.0)


def main():
	rclpy.init()

	node = MissionNode()
	rclpy.spin(node)

	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main()