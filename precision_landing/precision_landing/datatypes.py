"""Framework-independent inputs and outputs of the landing protocol.

The ROS adapter (or the simulator, or a test) converts its own messages into
these types, so nothing in the core package depends on ROS.
"""
from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np


class LandingState(str, Enum):
	IDLE = "idle"
	SEARCH = "search"
	ACQUIRE = "acquire"
	ALIGN = "align"
	DESCEND = "descend"
	REACQUIRE = "reacquire"
	TOUCHDOWN = "touchdown"
	LANDED = "landed"
	FALLBACK_LAND = "fallback_land"
	ABORTED = "aborted"


ACTIVE_STATES = frozenset({
	LandingState.SEARCH,
	LandingState.ACQUIRE,
	LandingState.ALIGN,
	LandingState.DESCEND,
	LandingState.REACQUIRE,
})
LAND_STATES = frozenset({LandingState.TOUCHDOWN, LandingState.FALLBACK_LAND})


class CommandMode(str, Enum):
	NONE = "none"  # lander inactive: publish nothing
	VELOCITY = "velocity"  # stream the NED velocity + yaw setpoint (offboard)
	LAND = "land"  # hand control to the autopilot's LAND mode
	HOLD = "hold"  # hand control to the autopilot's position hold


@dataclass
class TagDetection:
	"""One detected tag, straight from the detector."""
	tag_id: int
	corners: np.ndarray  # (4, 2) pixels, detector corner order (see tag_pose.py)
	decision_margin: float = math.inf
	hamming: int = 0


@dataclass
class VehicleState:
	"""Autopilot state estimate at time `stamp` (seconds, same clock as detections)."""
	stamp: float
	position: np.ndarray  # NED, m
	velocity: np.ndarray  # NED, m/s
	attitude: np.ndarray  # quaternion (w, x, y, z), body FRD -> NED
	range_bottom: float = math.nan  # downward rangefinder distance if available, m

	def __post_init__(self):
		self.position = np.asarray(self.position, dtype = float)
		self.velocity = np.asarray(self.velocity, dtype = float)
		self.attitude = np.asarray(self.attitude, dtype = float)


@dataclass
class Command:
	"""What the vehicle should do right now."""
	mode: CommandMode
	state: LandingState
	velocity: np.ndarray = field(default_factory = lambda: np.zeros(3))  # NED, m/s
	yaw: float = math.nan  # NED heading to hold, rad (nan = don't care)
