# RoverPathAlgo Wiki

Welcome to the RoverPathAlgo wiki. This document describes the intended real-world roadmap for the project as it evolves from a simulation to a multi-rover orchestration system that can dock at mixing/filling stations with real hardware integrations.

---

## Project vision

RoverPathAlgo will grow into the navigation and coordination layer for a fleet of rovers that can:

- Navigate between **mixing/filling stations** in a shared workspace.
- **Dock** precisely using AprilTag-assisted procedures.
- Localize using **UWB TDoA** (time-difference of arrival) for relative positioning.
- Accept **job-level mission plans** from higher-level logic (e.g., “produce mixture X”).
- Respect safety and traffic policies to avoid collisions and deadlocks.

The current repository contains a **simulation-centric path planner** (`main.py`) and is the foundation for the navigation layer. The future system will keep the planner core while extending inputs/outputs to integrate with real sensors, actuators, and station workflows.

---

## System layers (target architecture)

The overall system is expected to evolve into the following layers. RoverPathAlgo focuses on **Layer 2** today and will expand its integration surface with **Layer 1** and **Layer 3**.

### 1) Fleet coordination layer (mission planning)
**Responsibility:** decides *what* a rover should do next.

Examples:
- Given a target mixture, choose a **sequence of stations**.
- Assign a rover to a job and manage resource contention (e.g., one station at a time).
- Provide a time-ordered list of docking tasks to RoverPathAlgo.

**Outputs to RoverPathAlgo:**
- A list of station IDs in the required order.
- A target station for each active rover.
- Optional constraints (time windows, station availability, priorities).

### 2) Navigation and docking layer (RoverPathAlgo)
**Responsibility:** decides *how* the rover gets to each station safely.

Current capabilities:
- PRM + A* path planning in a 2D workspace.
- Basic traffic resolution for multiple rovers.
- Gate-based docking/undocking logic (approach + dock pose).

Planned additions:
- Interfaces for sensor-based localization (UWB TDoA).
- Interfaces for visual docking alignment (AprilTags).
- Live obstacle updates and dynamic re-planning.

### 3) Hardware control layer
**Responsibility:** commands motors/actuators and exposes sensor data.

Expected responsibilities:
- Convert target poses (x, y, heading) into low-level control commands.
- Publish odometry, IMU data, UWB ranges/TDoA data, and tag observations.
- Provide feedback loops for docking precision.

---

## Navigation model: gates → stations

In the simulation, a **gate** is a U-shaped docking structure with two poses:
- **Approach pose**: where the rover lines up.
- **Dock pose**: final stopping position.

In the real system, a **station** will map to a gate-like configuration:

- **Station geometry**: approach location, dock location, orientation.
- **Docking behavior**: how to transition from approach to dock using AprilTags.
- **Clearance boundaries**: safety regions for collision checking.

**Recommended mapping:**
- Station ID ↔ gate ID (unique per docking station).
- Station calibration step defines: `(approach pose, dock pose, docking orientation)`.

---

## Localization integration: UWB TDoA

The real system will require consistent, low-drift localization. A likely approach:

- Use UWB anchors (fixed) and rover tags (mobile).
- Compute rover pose using **TDoA** for relative position.
- Fuse UWB with wheel odometry/IMU for smooth local tracking.

**Integration path:**

1. **Define a localization adapter** that exposes: `get_pose()` and `get_covariance()`.
2. **Replace/supplement** simulated rover poses with localization outputs.
3. Provide a **pose confidence** signal to path planning to trigger re-planning or speed reduction.

**Considerations:**
- UWB typically provides **position**, not heading. Heading can be fused via IMU/odometry or a dual-tag setup.
- TDoA error increases near multipath reflections; keep fallback behaviors for degraded accuracy.

---

## Docking integration: AprilTags

Docking requires precise alignment and station-relative pose estimation. The docking layer should:

1. Detect AprilTags on the station (fiducial IDs mapped to station IDs).
2. Estimate rover pose relative to the station coordinate frame.
3. Execute a **fine alignment routine** from the approach pose into the dock pose.

**Suggested docking pipeline:**

1. **Navigate** to the station’s approach pose using the path planner.
2. **Search** for the AprilTag if not visible at the approach pose.
3. **Align** using tag observations and a local controller.
4. **Dock** until a confirmation signal (limit switch or proximity sensor).
5. **Hand off** to the station procedure (fill/mix).

---

## Station sequence planning (mixture logic)

Sequence planning belongs to the mission/fleet coordination layer. It should:

- Know **recipes** (sequence of station visits, dwell time, required docking orientation).
- Manage **resource contention** (which station is available).
- Provide the **next station** to RoverPathAlgo as rovers complete tasks.

**Data model (suggested):**

```json
{
  "job_id": "mix_001",
  "station_sequence": ["station_A", "station_C", "station_B"],
  "dwell_times": {"station_A": 30, "station_C": 45, "station_B": 20},
  "priority": 2
}
```

---

## Calibration (initial setup)

Before running rovers, the environment must be calibrated:

1. **Station placement survey**: measure and record approach/dock poses for each station.
2. **Anchor calibration**: align UWB anchors with the map coordinate frame.
3. **Tag registration**: map AprilTag IDs to station IDs and coordinate transforms.
4. **Safety boundaries**: encode forbidden zones or keep-out regions.

Calibration outputs should be stored in a structured file (e.g., YAML/JSON) and loaded by the navigation system at startup.

---

## Interfaces to add (future extensions)

Below are proposed interfaces to bridge simulation and real hardware. These can live in a new module (e.g., `adapters/`).

### 1) LocalizationAdapter
**Role:** returns rover pose and covariance.

```python
class LocalizationAdapter:
    def get_pose(self, rover_id: int) -> tuple[float, float, float]:
        ...

    def get_covariance(self, rover_id: int) -> list[list[float]]:
        ...
```

### 2) DockingAdapter
**Role:** handles AprilTag alignment and docking.

```python
class DockingAdapter:
    def dock(self, rover_id: int, station_id: str) -> bool:
        ...
```

### 3) MissionPlannerAPI
**Role:** yields station sequences for a rover.

```python
class MissionPlannerAPI:
    def next_station(self, rover_id: int) -> str:
        ...
```

---

## Current code reference (today)

All logic lives in `main.py` and includes:

- PRM + A* planner for global navigation.
- Gate-based docking with approach/dock poses.
- Traffic resolution for multiple rovers.

This file is the **best starting point** for integrating real-world components.

---

## Next steps (recommended milestones)

1. **Extract core planner** into a reusable module (so it can run headless without Matplotlib).
2. **Add adapters** for localization and docking.
3. **Load station layout** from a calibration file instead of random generation.
4. **Integrate mission planner** for station sequences.
5. **Run hardware-in-the-loop tests** with a single rover.

---

## Glossary

- **PRM**: Probabilistic Roadmap; sampling-based method for motion planning.
- **A***: Graph search algorithm used for shortest path planning.
- **UWB TDoA**: Ultra-wideband time-difference-of-arrival positioning.
- **AprilTag**: Fiducial marker system for precise visual localization.
- **Approach pose**: a pre-docking waypoint for alignment.
- **Dock pose**: final docking position.
