# precision_landing

AprilTag precision landing protocol for the worker drone (DEXI: PX4 + ROS 2), written from scratch.

Only the **tag detector** (finding corners and decoding IDs) comes from a library: `apriltag_ros`, already running in the DEXI stack. Everything after the corners is plain numpy written here:
- pose estimation
- frame transforms
- latency compensation
- filtering
- guidance
- search
- the landing state machine

The core package has **no ROS dependency**, so it is unit-tested and flown in a closed-loop simulator on a laptop before it ever touches the drone.

For now this lives as a self-contained folder in the `worker_mission_control` repo and is not yet wired into the ROS package (`setup.py` / `package.xml` are unchanged). See [Moving into worker_mission_control](#moving-into-worker_mission_control) for the integration steps.

```
precision_landing/
├── precision_landing/          # the protocol (pure Python + numpy) -> logic/precision_landing/
│   ├── lander.py               # state machine; the public entry point
│   ├── tag_pose.py             # corners -> 6-DOF pose (DLT homography + Levenberg-Marquardt)
│   ├── estimator.py            # Kalman filter tracking the pad in the local NED frame
│   ├── controller.py           # PID guidance, descent cone, speed schedules, slew limiting
│   ├── search.py               # expanding square spiral when the tag isn't in view
│   ├── camera.py               # pinhole + Brown-Conrady distortion (project / undistort)
│   ├── history.py              # vehicle-pose buffer to pair frames with capture-time pose
│   ├── geometry.py             # rotations, quaternions, SO(3) exp/log
│   ├── config.py               # every tunable number, YAML-loadable, typo-checked
│   └── datatypes.py            # framework-independent inputs/outputs
├── ros/precision_landing_node.py   # thin ROS 2 adapter (apriltag_ros + px4_msgs + dexi_offboard)
├── config/precision_landing.yaml   # example config (node + simulator share the format)
├── sim/                        # closed-loop simulator + Monte Carlo
└── tests/                      # 52 tests: math, estimator, controller, state machine, closed loop
```

## Quick start (laptop, no ROS)

```bash
cd precision_landing
python3 -m pytest tests -q                          # ~20 s
```

```bash
python3 -m sim.run_sim --seed 2                     # one landing, prints transitions, saves sim_output/episode.png
```

```bash
python3 -m sim.run_sim --monte-carlo 100 --scenario all
```

```bash
python3 -m sim.run_sim --monte-carlo 100 --set control.kp_xy=1.2 --set protocol.final_descent=guided
```

Needs `numpy` (plus `matplotlib` for plots, `pyyaml` for `--config`, `pytest` for tests).

## How it works

```
camera frame ──► apriltag_ros ──► 4 corner pixels + id            (library)
                                      │
     undistort ─► DLT homography ─► decompose ─► LM refine (+ level-pad prior)   tag_pose.py
                                      │  tag position in camera frame
     vehicle pose AT CAPTURE TIME (interpolated) ─► camera mount ─► NED          history.py / lander.py
                                      │  pad position measurement + range-dependent covariance
     Kalman filter, chi-square gate, re-init on persistent disagreement          estimator.py
                                      │  pad estimate (NED), survives dropouts
     state machine ─► PID + descent cone + speed schedule + accel limit          lander.py / controller.py
                                      │
                         NED velocity setpoint / LAND / HOLD  ──► PX4
```

**1. Pose from corners** (`tag_pose.py`)
- The corners are undistorted (iterative inverse of the lens model).
- A plane-to-image homography is fit with the Hartley-normalized DLT and decomposed into rotation and translation.
- That initial pose is refined by Levenberg-Marquardt on reprojection error, with rotation updated on SO(3).
- A **soft "pad is level" prior** uses the vehicle attitude, and it matters. A small tag seen nearly head-on has a poorly observable tilt, so noise invents one, and a tilted square looks smaller. The unconstrained fit is therefore biased toward the camera: about −16 cm at 4 m in tests. The prior removes the bias (−0.2 cm) and cuts lateral noise about 30%, even with 1° attitude error. Set `level_prior_sigma_deg: 0` if the pad itself tilts.
- The tag *position* doesn't depend on which corner the detector numbers first, or on the winding direction (there's a test for this), so detector conventions can't break landing.

**2. World-frame measurement** (`lander.add_detections`)
- Each frame is paired with the vehicle pose **at the image timestamp**, interpolated from a 2 s buffer (linear position, slerp attitude). On a Pi, detection latency is 50–150 ms, which is 5–15 cm at 1 m/s if ignored.
- The tag position goes through the camera mount (`R_body_cam` and offset) into NED.
- Each measurement gets a range-dependent covariance:
  - lateral noise ∝ d (pixel noise plus attitude error)
  - depth noise ∝ d² (from d = f·s / size_px)

**3. Tracking** (`estimator.py`)
- A Kalman filter tracks the landing point in the local NED frame.
- The default `static` model is a position random walk. It absorbs the slow drift of PX4's local frame without the noise-driven velocity states that would extrapolate garbage during a dropout. Use `moving` (constant velocity with decay) for a pad on a moving platform.
- A chi-square innovation gate rejects outliers. Eight consecutive rejections re-initialize the filter (it's the estimate that's wrong, not the measurements).
- Tracking the pad in the world frame, instead of steering on raw pixel offsets, means camera latency is outside the control loop and dropouts don't create fake zero errors.

**4. Guidance** (`controller.py`)
- PX4 already closes the velocity loop. We run the outer position loop: `v = Kp·e + Ki·∫e + Kd·(v_pad − v_vehicle)`, with conditional-integration anti-windup.
- Speed limits are scheduled by height.
- Descent is allowed only inside an **alignment cone** (`error < h·tan 12°`). It runs at full speed in the inner half of the cone and tapers to zero at the edge.
- Every command passes an acceleration (slew) limiter.

**5. State machine** (`lander.py`)

```
         sighted        locked         aligned          low + centered + slow
SEARCH --------> ACQUIRE ------> ALIGN --------> DESCEND ----------------------> TOUCHDOWN -> LANDED
  ^  ^              |              ^  |              |
  |  +--- lost -----+              |  +---- lost ----+--> REACQUIRE (climb above last estimate)
  |                                +------ locked ---------------+   |
  +-------------------------- climb done, no lock -------------------+
search timeout / no descent progress / too many reacquires / overall timeout -> FALLBACK_LAND
abort service / vehicle data stale -> ABORTED (PX4 hold)
```

| State | Behavior |
|---|---|
| SEARCH | Hover, or fly an expanding square spiral sized from the camera footprint. Times out to a fallback landing. |
| ACQUIRE | Hold still until the tag is **locked**: ≥6 accepted detections in 1 s and estimate std < 10 cm. This is the README's "confirm stable detection before trusting it". |
| ALIGN | Center over the pad at constant altitude until within 15 cm for 1 s. |
| DESCEND | Descend inside the cone. If the tag is lost, never descend blind: steer to the estimate and climb gently, which widens the footprint. If it's still lost after 2 s, go to REACQUIRE. If the tag vanishes below 0.6 m while the last *measured* error was < 12 cm, commit to landing (the tag often fills the image there). |
| TOUCHDOWN | At ≤0.4 m, centered within 5 cm and moving < 8 cm/s, hand off. `final_descent: land` (default, per the mission README) uses **PX4 LAND**. `guided` keeps steering on the estimate down to contact. |
| REACQUIRE | Climb 1 m above the last pad estimate. If there's no lock, spiral-search around it. Max 3 attempts. |
| FALLBACK_LAND | Land in place. Covers: tag never found, stall watchdog (no new lowest height for 30 s), too many reacquires, overall timeout. |
| ABORTED | Abort service, or vehicle state older than 0.5 s. Switches PX4 to Hold. |

## Simulator results

Run with `python3 -m sim.run_sim --monte-carlo 100 --scenario all`, using the default config.

Every scenario includes all of these:
- PX4-default velocity PI loop (`MPC_XY_VEL_P_ACC` 1.8, `_I_ACC` 0.4) with attitude lag
- steady wind plus gusts
- odometry drift
- attitude bias/noise
- 0.5° camera mount misalignment
- 1% focal-length calibration error
- lens distortion
- 80 ± 20 ms camera latency
- 5% frame dropouts

| scenario | precision landings | fallback | median error | p95 | max |
|---|---|---|---|---|---|
| nominal (light gusts) | 100 / 100 | 0 | 6.8 cm | 15.5 cm | 24.1 cm |
| far (start 2–3 m off the pad → spiral search) | 100 / 100 | 0 | 7.7 cm | 19.2 cm | 23.5 cm |
| occlusions (multi-second blackouts + 15% dropouts) | 100 / 100 | 0 | 8.8 cm | 19.1 cm | 21.7 cm |
| noisy (2 px noise, false detections, 3× drift) | 100 / 100 | 0 | 7.9 cm | 22.7 cm | 26.1 cm |
| rangefinder available | 100 / 100 | 0 | 7.7 cm | 18.2 cm | 21.1 cm |
| windy (strong gusts) | 12 / 100 | 88 | 12.1 cm | 33.5 cm | 34.0 cm |

What limits accuracy:
- In **calm air with every sensor error still on**, error is **median 0.7 cm, max 1.9 cm** (48 seeds). The remaining error is gusts pushing the airframe faster than PX4's default velocity loop corrects.
- **Stiffer PX4 velocity gains help more than anything in this package.** With `MPC_XY_VEL_P_ACC` 3.0 / `MPC_XY_VEL_I_ACC` 1.0, the windy scenario goes from 10/60 to 59/60 precision landings, and the nominal p95 drops from 15 to 13 cm. Tune these on the real airframe, not from the sim.
- In winds it can't handle, the protocol always ends in a controlled fallback landing (100/100 landed in every scenario).

The simulator is a model, not the drone. Treat these numbers as evidence the logic is sound, not as a flight accuracy guarantee.

## Moving into worker_mission_control

1. Copy `precision_landing/precision_landing/` to `worker_mission_control/worker_mission_control/logic/precision_landing/`. It only uses relative imports, so it works there unchanged.
2. Copy `ros/precision_landing_node.py` to `worker_mission_control/worker_mission_control/`, and change its import to `from worker_mission_control.logic.precision_landing import ...`.
3. Copy `config/precision_landing.yaml` to `worker_mission_control/config/` and install it from `setup.py`:
   `(os.path.join("share", package_name, "config"), glob("config/*.yaml"))`
4. In `setup.py` `console_scripts`, add:
   `"precision_landing_node = worker_mission_control.precision_landing_node:main"`
5. In `package.xml`, add `px4_msgs`, `sensor_msgs`, `std_srvs`, `python3-numpy` and `python3-yaml`. `apriltag_msgs` is already there.
6. Launch it next to `mission_node`, passing `config_file` (the path from `get_package_share_directory`) and the topic parameters below.
7. Mission phases 3–6 map onto it. `mission_node` calls `/precision_landing_node/start` (`std_srvs/Trigger`) when transit ends, then watches `/precision_landing_node/status` (JSON with `state`, `reason`, errors and heights) to drive the LED ring and the next phase. `/precision_landing_node/abort` switches PX4 to Hold.

The core can also be embedded directly in `mission_node`, since it is only `add_vehicle_state` / `add_detections` / `update`. The separate node keeps offboard setpoint ownership in one place.

### ROS interface

| | topic / service | type |
|---|---|---|
| sub | `detections_topic` (default `/detections`) | `apriltag_msgs/AprilTagDetectionArray` |
| sub | `camera_info_topic` (default `/cam0/camera_info`) | `sensor_msgs/CameraInfo` |
| sub | `/fmu/out/vehicle_local_position`, `/fmu/out/vehicle_attitude`, `vehicle_status_topic`, `/fmu/out/vehicle_land_detected` | `px4_msgs` |
| pub | `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint` (velocity only, NED, 20 Hz) | `px4_msgs` |
| pub | `/fmu/in/vehicle_command` (OFFBOARD / NAV_LAND / AUTO_LOITER) | `px4_msgs/VehicleCommand` |
| pub | `/dexi/pause_setpoints` (true while we stream, false when done) | `std_msgs/Bool` |
| pub | `~/status` (5 Hz JSON) | `std_msgs/String` |
| srv | `~/start`, `~/abort` | `std_srvs/Trigger` |

## Hardware bring-up checklist

**Verified so far:**
- The core package passes its tests.
- The simulator lands across all scenarios.
- The ROS adapter was driven end-to-end against **stub** ROS modules and the simulator. It paused `dexi_offboard`, streamed velocity setpoints, sent `NAV_LAND` once, then unpaused.

**Not yet verified:** it has not run on real ROS 2 or the drone. Before flying:

1. **Topic names.** `ros2 topic list`. Confirm the detections topic (apriltag_ros publishes `detections`, which DEXI's launch may remap or namespace), and whether status is `/fmu/out/vehicle_status` or `_v1`. Also confirm `/fmu/out/vehicle_attitude` is exported by the DDS bridge.
2. **QoS.** `ros2 topic info -v /fmu/out/vehicle_local_position`. We subscribe best-effort/volatile, which matches any writer.
3. **Camera calibration.** `ros2 topic echo /cam0/camera_info --once`. If `k` is all zeros, calibrate the camera; `fallback_hfov_deg: 62.2` is a stopgap only. If apriltag runs on a *rectified* image, set `detections_are_rectified: true`.
4. **Tag size.** Measure the black square edge, and put it in the config. Every distance scales with it.
5. **Camera mount (props off).** Start the node, hold the drone level over the tag, and call `~/start`. In `~/status`, `last_tag_body_frd` must read **+x when the tag is toward the nose** and +y when it's to the right. If it doesn't, fix `image_up_heading_deg`.
6. **dexi_offboard hand-off.** Confirm that pausing via `/dexi/pause_setpoints` stops *all* of its `/fmu/in` traffic, including OffboardControlMode, so only this node commands PX4 during landing. (DroneBlocks' `dexi_apriltag` docs describe the same pause mechanism.)
7. **Failsafes.** Check PX4's offboard-loss behavior (`COM_OF_LOSS_T`, `COM_OBL_RC_ACT`). If this node dies mid-descent, PX4 must do something sane.
8. **First flights.** Use a spotted area with RC override ready. Start at ~2 m with `final_descent: land`. Watch `~/status` (`horizontal_error_m`, `target_std_m`, `tag_fresh`). If it oscillates over the pad, lower `control.kp_xy`.

## Tuning: symptom → knob

| Symptom | Knob |
|---|---|
| Oscillates or overshoots over the pad | lower `control.kp_xy` (and/or raise `kd_xy`) |
| Slow to center, drifts in steady wind | raise `control.ki_xy` |
| Pushed around by gusts | PX4 `MPC_XY_VEL_P_ACC` / `MPC_XY_VEL_I_ACC` (see results above) |
| Stalls at one height ("no descent progress") | widen `control.cone_half_angle_deg`, or improve wind rejection |
| Loses the tag low down | raise `protocol.handoff_height`, or add a small nested tag (`tags[].offset`) |
| Estimate jumps / many `rejected_gate` in status | raise `detection.pixel_noise_px` or `noise_inflation`; check calibration and `extra_latency_s` |
| `height_vision_m` disagrees with `range_bottom_m` | tag `size` is wrong |

## Conventions

- **World:** PX4 local NED.
- **Body:** FRD.
- **Camera:** OpenCV (x right, y down, z out of the lens).
- **Tag:** x right, y down, z into the tag.
- **Quaternions:** (w, x, y, z), child→parent, matching PX4's `VehicleAttitude.q`.
- **Rotation names:** `R_a_b` maps vectors in frame b into frame a.
- **Units:** SI throughout, except config keys ending in `_deg`.
