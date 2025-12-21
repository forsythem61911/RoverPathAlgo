# RoverPathAlgo

RoverPathAlgo is a self-contained Python simulation that animates one or more square “rovers” navigating between dockable gates inside a bounded world. It combines geometry utilities, a probabilistic roadmap (PRM) planner with A* search, optional path smoothing, and a lightweight traffic-resolution policy so multiple rovers can move without colliding. The entire application lives in `main.py` and runs with Matplotlib.

## Quick start

```bash
python main.py
```

A Matplotlib window opens with controls for the number of docks/rovers, smoothing, and play/stop. Click a gate while paused to command a docked rover to travel there.

## High-level architecture

`main.py` is organized into a few major sections that work together:

1. **Configuration constants** (top of file): world dimensions, PRM parameters, collision radii, animation settings, and traffic policy constants.
2. **Geometry + collision utilities**: vector math, segment intersection, and collision checks.
3. **World objects**: `Gate` and `Rover` dataclasses.
4. **PRM + path search**: building and querying a roadmap, and A* over the graph.
5. **Path smoothing**: shortcutting, Chaikin smoothing, resampling, and heading reconstruction.
6. **Background planner**: runs path planning in a worker process for responsiveness.
7. **Simulation/visualization**: `Sim` class handles initialization, interaction, animation, and rover state updates.

## World model

### Gates
A **gate** is a U-shaped docking structure defined by:
- A center position and a heading angle (`theta`).
- Three walls (left, right, back) that form the “U” shape.
- Two key points: **approach** (where a rover lines up) and **dock** (the final stopping pose).

`Gate.__post_init__` computes these using simple 2D rotation (`rot2`) and stores the walls for collision testing.

### Rovers
A **rover** is a square with a pose `(x, y, heading)` and state including:
- Current and previous gate indices.
- Optional path (dense sequence of poses) and a path index cursor.
- Flags for movement, docking, and collision-avoidance behavior (yielding, reversing, replanning).

The rover state is updated frame-by-frame to follow its assigned path while avoiding obstacles (walls + other rovers).

## Geometry and collision

Collision logic is centralized in the `WallIndex` helper:
- `collides_pose` checks a single position against world bounds and wall segments.
- `collides_seg` checks a straight-line segment against walls (with radius).

Other helpers like `seg_intersect`, `pt_seg_d2`, and `seg_seg_d2` are used for safe path and traffic checks.

## Path planning pipeline

### 1. Probabilistic Roadmap (PRM)
A PRM is built by sampling collision-free nodes in the world and connecting each node to its nearest neighbors (`PRM_K`). The graph is used for global navigation.

Key functions:
- `knn_lists` precomputes neighbor indices for static PRM nodes.
- `knn_temp_xy` connects temporary start/goal nodes to nearby PRM nodes.

### 2. A* search on the roadmap
`astar_lazy` runs A* over the PRM graph with:
- **Heuristic**: Euclidean distance to the goal.
- **Edge feasibility**: `edge_ok` (checks segment collision and rover clearance).
- **Edge cost**: `se2_cost` (distance plus turn penalty).

The result is a list of node indices which is then **densified** into a pose-by-pose path using `densify_edge`.

### 3. Docking/undocking segments
If a rover is docked, the path first “backs out” from the dock to the approach point. Similarly, the final segment moves from the goal approach point into the dock.

### 4. Smoothing (optional)
`smooth_mid_points` tries to reduce zig-zags in the middle portion of the path:
- **Shortcutting** removes unnecessary turns.
- **Chaikin** creates a smoother curve.
- **Resampling** creates evenly spaced points.
- **Heading reconstruction** (`tan_heads`) rebuilds orientations.

Smoothing only happens if the smoothed path remains collision-free.

## Background planning
Planning is CPU-heavy, so `Sim` uses a `ProcessPoolExecutor` to run `_plan_worker` asynchronously. This allows UI and animation updates to keep running while a new path is computed.

The worker receives:
- Current rover pose and docking status.
- Current/goal gate approach + dock poses.
- Other rover positions as dynamic obstacles.

When the worker finishes, `poll_plan_jobs` installs the new path and updates the rover’s state.

## Traffic management and collision avoidance
When multiple rovers move simultaneously, `resolve_traffic` enforces a deterministic policy:

- Each rover proposes a target index along its path.
- The simulator checks for conflicts with other rovers’ proposed positions and motion segments.
- If blocked, the rover waits and increments a **stuck** counter.
- After waiting long enough, the higher-ID rover yields by **reversing** along its path or injecting a small backoff segment.
- Cycles in the blocked graph are broken by forcing the highest-ID rover to reverse.
- Long stalls trigger background replanning.

This keeps the system moving without teleporting rovers or violating safety distances.

## Visualization and UI
`Sim.setup_vis` builds the Matplotlib UI:
- Sliders for **Docks** and **Rovers**.
- Buttons for **Play/Stop**, **Rerandomize gates**, and **Smooth ON/OFF**.
- Click-to-command: while paused, click a gate to send a docked rover there.

The `animate` method advances rover states, resolves traffic, installs new paths, and updates the rendered polygons and trails.

## Main entry point

```python
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    Sim().run()
```

Running the script sets up multiprocessing (so the planner can run in a worker process) and launches the interactive simulation.

## Key constants to tweak

- `PRM_SAMPLES`, `PRM_K`: roadmap size and connectivity.
- `W_TURN`: turn penalty for path cost.
- `SMOOTH_DEFAULT`: enable/disable smoothing.
- `NUM_GATES`: default gate count.
- `ROVER_SEP`: minimum rover-to-rover clearance.
- `REPLAN_TRIGGER`, `REPLAN_COOLDOWN`: replanning sensitivity.

## File map

- `main.py`: everything (simulation, planning, UI, and utilities).
- `README.md`: this documentation.

## Notes

- The simulation is deterministic given `BASE_SEED`.
- Planner performance depends on `PRM_SAMPLES` and `A_STAR_EXPANSION_LIMIT`.
- The code relies on Matplotlib and NumPy.
