"""ROS 2 adapter for the precision landing protocol (DEXI: PX4 + apriltag_ros).

This is a thin translation layer. All landing logic lives in the ROS-free
`precision_landing` package so it can be unit-tested and simulated off-drone.

	in:  apriltag_msgs/AprilTagDetectionArray   (detections: corners + id)
	     sensor_msgs/CameraInfo                 (intrinsics + distortion)
	     px4_msgs VehicleLocalPosition / VehicleAttitude / VehicleStatus / VehicleLandDetected
	out: px4_msgs OffboardControlMode + TrajectorySetpoint (velocity, NED) at control_rate_hz
	     px4_msgs VehicleCommand (offboard / LAND / hold mode switches)
	     std_msgs/Bool on /dexi/pause_setpoints so dexi_offboard stops streaming its own setpoints
	     std_msgs/String JSON status on ~/status
	srv: ~/start, ~/abort  (std_srvs/Trigger)

After moving into worker_mission_control, change the precision_landing import
below to `worker_mission_control.logic.precision_landing` and add to package.xml:
px4_msgs, sensor_msgs, std_srvs (and python3-numpy, python3-yaml).
"""
import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from apriltag_msgs.msg import AprilTagDetectionArray
from px4_msgs.msg import (
	OffboardControlMode,
	TrajectorySetpoint,
	VehicleAttitude,
	VehicleCommand,
	VehicleLandDetected,
	VehicleLocalPosition,
	VehicleStatus,
)
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from precision_landing import (
	CameraModel,
	CommandMode,
	LanderConfig,
	PrecisionLander,
	TagDetection,
	VehicleState,
)

NAN3 = [math.nan, math.nan, math.nan]


def _qos(reliability, durability, depth = 1):
	return QoSProfile(reliability = reliability, durability = durability, history = HistoryPolicy.KEEP_LAST, depth = depth)


# PX4 /fmu/out writers are best-effort; a volatile reader matches any writer durability
QOS_PX4_OUT = _qos(ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE)
# what dexi_offboard uses for /fmu/in; compatible with PX4's best-effort readers
QOS_PX4_IN = _qos(ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.TRANSIENT_LOCAL)
# dexi_offboard subscribes to /dexi/pause_setpoints as transient_local: the writer must be too
QOS_LATCHED = _qos(ReliabilityPolicy.RELIABLE, DurabilityPolicy.TRANSIENT_LOCAL)
QOS_SENSOR = _qos(ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE, depth = 5)


class PrecisionLandingNode(Node):
	def __init__(self):
		super().__init__("precision_landing_node")

		self.declare_parameter("config_file", "")
		self.declare_parameter("control_rate_hz", 20.0)
		# apriltag_ros publishes "detections" (relative); DEXI's launch may remap/namespace it,
		# so verify with `ros2 topic list | grep -i detection`
		self.declare_parameter("detections_topic", "/detections")
		self.declare_parameter("camera_info_topic", "/cam0/camera_info")
		self.declare_parameter("detections_are_rectified", False)  # True if apriltag runs on image_rect
		self.declare_parameter("fallback_hfov_deg", 0.0)  # used only if camera_info is uncalibrated (DEXI IMX219 ~62.2)
		self.declare_parameter("local_position_topic", "/fmu/out/vehicle_local_position")
		self.declare_parameter("attitude_topic", "/fmu/out/vehicle_attitude")
		self.declare_parameter("vehicle_status_topic", "/fmu/out/vehicle_status")  # some firmware: _v1
		self.declare_parameter("land_detected_topic", "/fmu/out/vehicle_land_detected")
		self.declare_parameter("pause_setpoints_topic", "/dexi/pause_setpoints")  # "" to disable
		self.declare_parameter("auto_engage_offboard", True)

		config = self._load_config(self.get_parameter("config_file").value)
		self.lander = PrecisionLander(config)
		self.lander.on_transition = self._on_transition
		self.rectified = self.get_parameter("detections_are_rectified").value
		self.fallback_hfov = self.get_parameter("fallback_hfov_deg").value
		self.auto_offboard = self.get_parameter("auto_engage_offboard").value

		self.attitude_q = None
		self.nav_state = None
		self.streaming = False
		self.stream_ticks = 0
		self.paused_dexi = False
		self.last_mode_request = -1e9

		p = self.get_parameter
		self.create_subscription(AprilTagDetectionArray, p("detections_topic").value, self._on_detections, QOS_SENSOR)
		self.create_subscription(CameraInfo, p("camera_info_topic").value, self._on_camera_info, QOS_SENSOR)
		self.create_subscription(VehicleLocalPosition, p("local_position_topic").value, self._on_local_position, QOS_PX4_OUT)
		self.create_subscription(VehicleAttitude, p("attitude_topic").value, self._on_attitude, QOS_PX4_OUT)
		self.create_subscription(VehicleStatus, p("vehicle_status_topic").value, self._on_status, QOS_PX4_OUT)
		self.create_subscription(VehicleLandDetected, p("land_detected_topic").value, self._on_land_detected, QOS_PX4_OUT)

		self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", QOS_PX4_IN)
		self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", QOS_PX4_IN)
		self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", QOS_PX4_IN)
		pause_topic = p("pause_setpoints_topic").value
		self.pause_pub = self.create_publisher(Bool, pause_topic, QOS_LATCHED) if pause_topic else None
		self.status_pub = self.create_publisher(String, "~/status", 10)

		self.create_service(Trigger, "~/start", self._srv_start)
		self.create_service(Trigger, "~/abort", self._srv_abort)

		self.create_timer(1.0 / p("control_rate_hz").value, self._control_step)
		self.create_timer(0.2, self._publish_status)
		self.get_logger().info("precision landing node ready (call ~/start to begin)")

	# ---------------------------------------------------------------- setup
	def _load_config(self, path):
		if not path:
			self.get_logger().warn("no config_file given, using built-in defaults")
			return LanderConfig()
		import yaml
		with open(path) as f:
			data = yaml.safe_load(f) or {}
		return LanderConfig.from_dict(data.get("precision_landing", data))

	def _now(self):
		return self.get_clock().now().nanoseconds * 1e-9

	def _px4_timestamp(self):
		return int(self.get_clock().now().nanoseconds / 1000)

	def _on_transition(self, old, new, reason):
		self.get_logger().info(f"[precision landing] {old.value} -> {new.value}: {reason}")

	# ---------------------------------------------------------------- inputs
	def _on_camera_info(self, msg):
		if self.lander.camera is not None:
			return
		if msg.k[0] > 0.0:
			if msg.d and msg.distortion_model not in ("plumb_bob", "rational_polynomial") and any(msg.d):
				self.get_logger().warn(f"distortion model '{msg.distortion_model}' not supported, ignoring distortion")
				camera = CameraModel.from_camera_info(msg.k, [], msg.width, msg.height, msg.p, self.rectified)
			else:
				camera = CameraModel.from_camera_info(msg.k, list(msg.d), msg.width, msg.height, msg.p, self.rectified)
		elif self.fallback_hfov > 0.0:
			self.get_logger().warn("camera_info is uncalibrated: using fallback_hfov_deg (calibrate for accuracy!)")
			camera = CameraModel.from_fov(msg.width, msg.height, self.fallback_hfov)
		else:
			self.get_logger().warn("camera_info is uncalibrated and fallback_hfov_deg is 0: waiting", throttle_duration_sec = 5.0)
			return
		self.lander.set_camera(camera)
		self.get_logger().info(f"camera: {camera.width}x{camera.height} fx={camera.fx:.1f} fy={camera.fy:.1f} hfov={math.degrees(camera.hfov):.1f} deg")

	def _on_attitude(self, msg):
		self.attitude_q = np.array(msg.q, dtype = float)  # (w, x, y, z), FRD -> NED

	def _on_local_position(self, msg):
		if self.attitude_q is None or not (msg.xy_valid and msg.z_valid and msg.v_xy_valid and msg.v_z_valid):
			return
		self.lander.add_vehicle_state(VehicleState(
			stamp = self._now(),
			position = np.array([msg.x, msg.y, msg.z]),
			velocity = np.array([msg.vx, msg.vy, msg.vz]),
			attitude = self.attitude_q.copy(),
			range_bottom = float(msg.dist_bottom) if msg.dist_bottom_valid else math.nan,
		))

	def _on_status(self, msg):
		self.nav_state = msg.nav_state

	def _on_land_detected(self, msg):
		if msg.landed:
			self.lander.notify_landed(self._now())

	def _on_detections(self, msg):
		stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		if stamp <= 0.0:
			stamp = self._now()
		detections = [
			TagDetection(
				tag_id = int(d.id),
				corners = np.array([[c.x, c.y] for c in d.corners]),
				decision_margin = float(d.decision_margin),
				hamming = int(d.hamming),
			)
			for d in msg.detections
		]
		self.lander.add_detections(stamp, detections)

	# ---------------------------------------------------------------- services
	def _srv_start(self, request, response):
		if self.lander.camera is None:
			response.success = False
			response.message = "no camera calibration yet (check camera_info topic)"
			return response
		ok, message = self.lander.start(self._now())
		response.success = ok
		response.message = message
		return response

	def _srv_abort(self, request, response):
		response.success = self.lander.abort(self._now(), "abort service called")
		response.message = "aborted" if response.success else "not running"
		return response

	# ---------------------------------------------------------------- outputs
	def _control_step(self):
		now = self._now()
		cmd = self.lander.update(now)
		if cmd.mode == CommandMode.VELOCITY:
			self._stream_velocity(cmd, now)
		elif cmd.mode == CommandMode.LAND:
			self.streaming = False
			if self.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
				self._request(now, VehicleCommand.VEHICLE_CMD_NAV_LAND)
		elif cmd.mode == CommandMode.HOLD:
			self.streaming = False
			if self.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
				# custom mode: main AUTO (4), sub LOITER (3)
				self._request(now, VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
			else:
				self._set_dexi_paused(False)
		else:  # NONE: lander idle / landed
			self.streaming = False
			self._set_dexi_paused(False)

	def _stream_velocity(self, cmd, now):
		if not self.streaming:
			self.streaming = True
			self.stream_ticks = 0
			self._set_dexi_paused(True)
		ts = self._px4_timestamp()
		mode = OffboardControlMode()
		mode.timestamp = ts
		mode.position = False
		mode.velocity = True
		mode.acceleration = False
		mode.attitude = False
		mode.body_rate = False
		self.offboard_pub.publish(mode)
		sp = TrajectorySetpoint()
		sp.timestamp = ts
		sp.position = NAN3
		sp.velocity = [float(v) for v in cmd.velocity]
		sp.acceleration = NAN3
		sp.yaw = float(cmd.yaw)
		sp.yawspeed = math.nan
		self.setpoint_pub.publish(sp)
		self.stream_ticks += 1
		# PX4 only accepts OFFBOARD after it has been receiving setpoints for a moment
		if self.auto_offboard and self.stream_ticks >= 10 and self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
			self._request(now, VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)  # custom main mode OFFBOARD

	def _request(self, now, command, p1 = 0.0, p2 = 0.0, p3 = 0.0):
		"""Send a VehicleCommand, re-sending at most once a second until PX4 complies."""
		if now - self.last_mode_request < 1.0:
			return
		self.last_mode_request = now
		msg = VehicleCommand()
		msg.timestamp = self._px4_timestamp()
		msg.command = command
		msg.param1 = float(p1)
		msg.param2 = float(p2)
		msg.param3 = float(p3)
		msg.target_system = 1
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		self.command_pub.publish(msg)

	def _set_dexi_paused(self, paused):
		if self.pause_pub is None or paused == self.paused_dexi:
			return
		self.paused_dexi = paused
		self.pause_pub.publish(Bool(data = paused))

	def _publish_status(self):
		status = self.lander.status()
		status["px4_nav_state"] = self.nav_state
		self.status_pub.publish(String(data = json.dumps(status)))


def main():
	rclpy.init()
	node = PrecisionLandingNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.try_shutdown()


if __name__ == "__main__":
	main()
