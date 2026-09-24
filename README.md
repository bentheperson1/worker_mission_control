# worker_mission_control

ROS2 node controlling the full worker drone mission profile for the helicarrier project: takeoff, transit to target coordinates, carrier-drone avoidance, AprilTag-based precision descent, and landing.

## Package Info

- **Package name:** `worker_mission_control`
- **Node:** `mission_node`
- **Build type:** `ament_python`
- **Depends on:** `dexi_offboard` (PX4 offboard control) and `apriltag_ros` (tag detection), which are already running as part of the core DEXI stack

## Setup

Team members will mainly be working directly with drone hardware through a local SSH connection

Use the following steps as a reference on how to gain access:
1. Connect to the DEXI's hotspot or wifi network that the DEXI is already connected to
2. Run: 
```bash
ssh dexi@dexi-114b.local
```
3. Type ``yes`` if you get a message about fingerprints. the drone is safe and wont kill you (digitally at least)
4. Type the password ``dexi`` when pro

## Building

You must rebuild the node after making any changes or the code run will be an outdated version.

From the workspace root:

```bash
cd ~/dexi_ws
colcon build --packages-select worker_mission_control
source install/setup.bash
```

Using `--packages-select` keeps rebuilds fast and avoids touching the rest of the DEXI stack while you iterate. **Always re-source after building** — `colcon build` does not update your current shell, and this is the single most common "package not found" cause.

## Running

For development and testing, run the node directly rather than folding it into `dexi.service` yet:

```bash
ros2 run worker_mission_control mission_node
```

Useful checks while it's running:

```bash
ros2 node list                    # confirm mission_node is up
ros2 topic list                   # confirm it's seeing dexi_offboard / apriltag topics
```

## Development Workflow

1. Edit code under `worker_mission_control/worker_mission_control/`
2. Rebuild just this package: `colcon build --packages-select worker_mission_control`
3. Re-source: `source install/setup.bash`
4. Re-run: `ros2 run worker_mission_control mission_node`

No need to restart `dexi.service` for any of the above — that only matters once this node is folded into the boot-time launch config (see Roadmap).

Whenever you complete your work and want to push to this repo:
1. ``git add .`` - Adds everything changed to be pushed
2. ``git commit -m "[Your Name] - changes`` - You must include your name in your commit message as everything will be under a single Git email
3. ``git push origin main`` - Push to this repo

## Working Together on Shared Hardware

Everyone SSHes into the same physical drone rather than working locally, so a few habits avoid stepping on each other:

- **Check before running** — `ros2 node list` before starting `mission_node`. A duplicate instance publishing conflicting offboard setpoints is a real flight hazard, not just a nuisance.
- **Use `tmux` or `screen`** for your session so work survives an SSH disconnect, and so others can see what's already running (`tmux ls`) before starting their own.
- **Coordinate before restarting `dexi.service`** — it relaunches the entire stack this node depends on, which affects everyone's session, not just yours.
- **Branch your own work** rather than committing straight to the drone's checkout of `main`, especially while multiple people are actively testing on the same box. Pull/rebase rather than letting uncommitted local changes pile up on shared hardware.

## Mission Phases

`mission_node` will run as a state machine:

1. **Takeoff** — arm and climb via `dexi_offboard`
2. **Transit** — fly to the target coordinates
3. **Search / acquire tag** — locate and confirm a stable AprilTag detection before trusting it
4. **Precision descent** — null out XY error against the tag while descending
5. **Final touchdown handoff** — hand off to a standard PX4 `LAND` below a safety altitude rather than vision-controlling all the way to the ground
6. **Timeout / fallback** — land at current position if the tag is never acquired, rather than hovering indefinitely

Carrier-drone avoidance runs as an ongoing check throughout the mission, not a discrete phase.

## Roadmap

- Expose mission triggering as a ROS2 action (goal = target coordinates, feedback = current phase, native cancel support) so the web dashboard can start/monitor/abort a mission over rosbridge
- Add an entry to `dexi_bringup`'s `dexi.repos` once stable, so it's pulled automatically along with the rest of the stack
- Fold into `.dexi-config.yaml` / boot-time launch once validated in manual testing

## Safety Notes

- This node commands live offboard flight — validate the state machine with props off or in a controlled, spotted area before trusting it in the air
- Never trust the precision descent phase below a safety altitude without first confirming tag-lock stability over several consecutive frames
