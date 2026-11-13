"""AprilTag precision landing protocol, written from scratch (numpy only).

Only the tag *detector* is an external library (apriltag_ros on the drone).
Pose estimation, filtering, guidance and the landing state machine live here
and have no ROS dependency, so they can be unit-tested and simulated off-drone.
"""
from .camera import CameraModel
from .config import (
	CameraMount,
	ControlConfig,
	DetectionConfig,
	LanderConfig,
	ProtocolConfig,
	TagSpec,
	TrackerConfig,
)
from .datatypes import Command, CommandMode, LandingState, TagDetection, VehicleState
from .lander import PrecisionLander
from .tag_pose import TagPose, estimate_tag_pose

__all__ = [
	"CameraModel",
	"CameraMount",
	"Command",
	"CommandMode",
	"ControlConfig",
	"DetectionConfig",
	"LanderConfig",
	"LandingState",
	"PrecisionLander",
	"ProtocolConfig",
	"TagDetection",
	"TagPose",
	"TagSpec",
	"TrackerConfig",
	"VehicleState",
	"estimate_tag_pose",
]
