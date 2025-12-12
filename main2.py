import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import math, random
from dataclasses import dataclass
import heapq

# ============================================================
# CONFIG
# ============================================================
BASE_SEED = 7
NUM_GATES = 5

S = 1.0                 # rover side length
DEPTH = 2.2 * S         # gate depth
APPROACH_L = 1.35 * S   # standoff before docking
MIN_GATE_SEP = 2.8 * S  # min separation between gate centers

WORLD_HALFSPAN = 9.0
BOUND_MARGIN = 1.5 * S  # keep rover inside world boundary

EPS = 2e-3              # touching allowed: shrink rover by 2*EPS in collision tests

# Lazy PRM settings
PRM_SAMPLES = 900       # random free samples
PRM_K = 20              # nearest neighbors per node
TEMP_K = 28             # neighbors for temp start/goal nodes
A_STAR_EXPANSION_LIMIT = 10000

# Motion sampling resolution
POS_RES = 0.08
TH_RES  = 0.12

# Costs
W_TURN = 0.18
W_REVERSE = 0.05

# Animation
FRAME_MS = 20
PLAYBACK_SKIP = 3

# ============================================================
# BASIC MATH
# ============================================================
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# ============================================================
# GEOMETRY (FAST)
# ============================================================
def seg_intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def on_segment(p, q, r):
        return (min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and
                min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    if abs(o1) < 1e-12 and on_segment(a, c, b): return True
    if abs(o2) < 1e-12 and on_segment(a, d, b): return True
    if abs(o3) < 1e-12 and on_segment(c, a, d): return True
    if abs(o4) < 1e-12 and on_segment(c, b, d): return True
    return False

def poly_edges(poly):
    return [(poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))]

def square_corners(center, theta, side):
    c = np.array(center, dtype=float)
    h = side / 2.0
    local = np.array([[-h, -h],
                      [ h, -h],
                      [ h,  h],
                      [-h,  h]], dtype=float)
    R = rot2(theta)
    return (c + (R @ local.T).T)

def point_segment_distance_sq(p, a, b):
    ap = p - a
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        d = p - a
        return float(np.dot(d, d))
    t = float(np.dot(ap, ab) / ab2)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    proj = a + t * ab
    d = p - proj
    return float(np.dot(d, d))

def point_in_convex_quad(p, quad):
    x, y = p
    s = None
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i+1) % 4]
        cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
        if abs(cross) < 1e-12:
            return False
        sign = cross > 0
        if s is None:
            s = sign
        elif s != sign:
            return False
    return True

def square_intersects_segment(center, theta, side, s0, s1):
    sq = square_corners(center, theta, side)
    if point_in_convex_quad(s0, sq) or point_in_convex_quad(s1, sq):
        return True
    for e0, e1 in poly_edges(sq):
        if seg_intersect(e0, e1, s0, s1):
            return True
    return False

def square_collides_walls(center, theta, side, walls, eps=EPS):
    side_eff = max(1e-6, side - 2.0*eps)
    c = np.array(center, dtype=float)
    r = (side_eff / 2.0) * math.sqrt(2.0)
    r2 = r * r

    for (w0, w1) in walls:
        if point_segment_distance_sq(c, w0, w1) > r2:
            continue
        if square_intersects_segment(c, theta, side_eff, w0, w1):
            return True
    return False

def within_bounds(center, bounds, margin):
    x, y = center
    xmin, xmax, ymin, ymax = bounds
    return (xmin + margin <= x <= xmax - margin) and (ymin + margin <= y <= ymax - margin)

# ============================================================
# GATE
# ============================================================
@dataclass
class Gate:
    center: np.ndarray
    theta: float
    s: float
    depth: float

    def local_to_world(self, xy):
        return self.center + rot2(self.theta) @ xy

    def inward_normal_world(self):
        return rot2(self.theta) @ np.array([0.0, 1.0])

    def opening_endpoints_world(self):
        p1 = self.local_to_world(np.array([-self.s/2, 0.0]))
        p2 = self.local_to_world(np.array([ self.s/2, 0.0]))
        return p1, p2

    def walls_world(self):
        L0 = self.local_to_world(np.array([-self.s/2, 0.0]))
        L1 = self.local_to_world(np.array([-self.s/2, self.depth]))
        R0 = self.local_to_world(np.array([ self.s/2, 0.0]))
        R1 = self.local_to_world(np.array([ self.s/2, self.depth]))
        B0 = self.local_to_world(np.array([-self.s/2, self.depth]))
        B1 = self.local_to_world(np.array([ self.s/2, self.depth]))
        return [(L0, L1), (R0, R1), (B0, B1)]

    def required_heading(self):
        n = self.inward_normal_world()
        return math.atan2(n[1], n[0])

    def dock_center(self):
        p1, p2 = self.opening_endpoints_world()
        m = 0.5*(p1+p2)
        n = self.inward_normal_world()
        return m + (self.s/2)*n

    def approach_center(self, L):
        n = self.inward_normal_world()
        return self.dock_center() - L*n

# ============================================================
# MOTION + COLLISION SAMPLING
# ============================================================
def rotate_in_place_free(x, y, th0, th1, side, walls, bounds, margin, th_res=TH_RES):
    if not within_bounds((x, y), bounds, margin):
        return False
    d = wrap_angle(th1 - th0)
    steps = max(2, int(abs(d) / th_res) + 1)
    for i in range(steps + 1):
        t = i / steps
        th = wrap_angle(th0 + t * d)
        if square_collides_walls((x, y), th, side, walls):
            return False
    return True

def drive_straight_free(x0, y0, x1, y1, th, side, walls, bounds, margin, pos_res=POS_RES):
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    steps = max(2, int(dist / pos_res) + 1)
    for i in range(steps + 1):
        t = i / steps
        x = x0 + t * dx
        y = y0 + t * dy
        if not within_bounds((x, y), bounds, margin):
            return False
        if square_collides_walls((x, y), th, side, walls):
            return False
    return True

def se2_edge_free_with_reverse(a, b, side, walls, bounds, margin):
    ax, ay, ath = a
    bx, by, bth = b
    dx, dy = bx - ax, by - ay
    dist2 = dx*dx + dy*dy

    if dist2 < 1e-12:
        return rotate_in_place_free(ax, ay, ath, bth, side, walls, bounds, margin), False

    travel = math.atan2(dy, dx)

    okA = (rotate_in_place_free(ax, ay, ath, travel, side, walls, bounds, margin) and
           drive_straight_free(ax, ay, bx, by, travel, side, walls, bounds, margin) and
           rotate_in_place_free(bx, by, travel, bth, side, walls, bounds, margin))

    travel_rev = wrap_angle(travel + np.pi)
    okB = (rotate_in_place_free(ax, ay, ath, travel_rev, side, walls, bounds, margin) and
           drive_straight_free(ax, ay, bx, by, travel_rev, side, walls, bounds, margin) and
           rotate_in_place_free(bx, by, travel_rev, bth, side, walls, bounds, margin))

    if okA:
        return True, False
    if okB:
        return True, True
    return False, False

def se2_edge_cost(a, b, used_reverse=False):
    ax, ay, ath = a
    bx, by, bth = b
    dist = math.hypot(bx-ax, by-ay)
    if dist < 1e-12:
        return W_TURN * abs(wrap_angle(bth-ath))

    travel = math.atan2(by-ay, bx-ax)
    if used_reverse:
        travel = wrap_angle(travel + np.pi)

    turn = abs(wrap_angle(travel - ath)) + abs(wrap_angle(bth - travel))
    return dist + W_TURN * turn + (W_REVERSE if used_reverse else 0.0)

def densify_edge(a, b, used_reverse=False, ds=0.06, dth=0.12):
    ax, ay, ath = a
    bx, by, bth = b
    out = [(ax, ay, ath)]
    dx, dy = bx-ax, by-ay
    dist = math.hypot(dx, dy)

    if dist < 1e-12:
        d = wrap_angle(bth-ath)
        n = max(1, int(abs(d)/dth))
        for i in range(1, n+1):
            t = i/n
            out.append((ax, ay, wrap_angle(ath + t*d)))
        return out

    travel = math.atan2(dy, dx)
    if used_reverse:
        travel = wrap_angle(travel + np.pi)

    d0 = wrap_angle(travel - ath)
    n0 = max(1, int(abs(d0)/dth))
    for i in range(1, n0+1):
        t = i/n0
        out.append((ax, ay, wrap_angle(ath + t*d0)))

    n1 = max(1, int(dist/ds))
    for i in range(1, n1+1):
        t = i/n1
        out.append((ax + t*dx, ay + t*dy, travel))

    d2 = wrap_angle(bth - travel)
    n2 = max(1, int(abs(d2)/dth))
    for i in range(1, n2+1):
        t = i/n2
        out.append((bx, by, wrap_angle(travel + t*d2)))

    return out

def densify_straight_pose(a_xy, b_xy, th, ds=0.05):
    ax, ay = a_xy
    bx, by = b_xy
    dx, dy = bx-ax, by-ay
    dist = math.hypot(dx, dy)
    n = max(1, int(dist/ds))
    out = []
    for i in range(n+1):
        t = i/n
        out.append((ax + t*dx, ay + t*dy, th))
    return out

# ============================================================
# WORLD GENERATION (DOCKABLE)
# ============================================================
def generate_world(num_gates, seed):
    rng_np = np.random.default_rng(seed)
    bounds = (-WORLD_HALFSPAN, WORLD_HALFSPAN, -WORLD_HALFSPAN, WORLD_HALFSPAN)

    gates = []
    tries = 0
    while len(gates) < num_gates and tries < 50000:
        tries += 1
        center = rng_np.uniform(-0.7*WORLD_HALFSPAN, 0.7*WORLD_HALFSPAN, size=2)
        theta = rng_np.uniform(-np.pi, np.pi)
        g = Gate(center=center, theta=theta, s=S, depth=DEPTH)

        if any(np.linalg.norm(g.center - gg.center) < MIN_GATE_SEP for gg in gates):
            continue

        other_walls = []
        for gg in gates:
            other_walls.extend(gg.walls_world())

        dock = g.dock_center()
        app  = g.approach_center(APPROACH_L)
        th   = g.required_heading()

        if not within_bounds(dock, bounds, BOUND_MARGIN): continue
        if not within_bounds(app,  bounds, BOUND_MARGIN): continue

        if square_collides_walls(dock, th, S, other_walls): continue
        if square_collides_walls(app,  th, S, other_walls): continue

        gw = g.walls_world()
        if square_collides_walls(dock, th, S, gw): continue
        if square_collides_walls(app,  th, S, gw): continue

        if not drive_straight_free(app[0], app[1], dock[0], dock[1], th, S, gw, bounds, BOUND_MARGIN):
            continue
        if not drive_straight_free(dock[0], dock[1], app[0], app[1], th, S, gw, bounds, BOUND_MARGIN):
            continue

        gates.append(g)

    if len(gates) < num_gates:
        raise RuntimeError("Failed to generate a dockable world. Try WORLD_HALFSPAN bigger or MIN_GATE_SEP smaller.")

    walls = []
    for g in gates:
        walls.extend(g.walls_world())
    return gates, walls, bounds

# ============================================================
# LAZY PRM (kNN by XY, edges validated on-demand)
# ============================================================
def build_knn_lists(nodes, k):
    xy = np.array([[n[0], n[1]] for n in nodes], dtype=float)
    N = len(nodes)
    nbrs = []
    for i in range(N):
        d = np.linalg.norm(xy - xy[i], axis=1)
        idx = np.argsort(d)
        nbrs.append([j for j in idx[1:k+1]])
    return nbrs

def connect_temp_neighbors(temp_state, base_nodes, k_temp):
    xy = np.array([[n[0], n[1]] for n in base_nodes], dtype=float)
    tx, ty, _ = temp_state
    d = np.linalg.norm(xy - np.array([tx, ty]), axis=1)
    idx = np.argsort(d)[:k_temp]
    return list(idx)

def astar_lazy(nodes, nbrs, start_idx, goal_idx, edge_free_fn, edge_cost_fn, expansion_limit=A_STAR_EXPANSION_LIMIT):
    def h(i):
        x, y, _ = nodes[i]
        gx, gy, _ = nodes[goal_idx]
        return math.hypot(gx-x, gy-y)

    edge_cache = {}  # (min,max) -> (ok, used_reverse)

    def edge_eval(i, j):
        a = i if i < j else j
        b = j if i < j else i
        key = (a, b)
        if key in edge_cache:
            return edge_cache[key]
        ok, used_rev = edge_free_fn(nodes[i], nodes[j])
        edge_cache[key] = (ok, used_rev)
        return ok, used_rev

    pq = []
    heapq.heappush(pq, (h(start_idx), 0.0, start_idx, -1))
    came = {}
    gbest = {start_idx: 0.0}
    closed = set()
    expansions = 0

    while pq:
        f, g, u, parent = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        came[u] = parent

        if u == goal_idx:
            idx_path = []
            cur = u
            while cur != -1:
                idx_path.append(cur)
                cur = came[cur]
            idx_path.reverse()

            rev_flags = []
            for a_i, b_i in zip(idx_path[:-1], idx_path[1:]):
                ok, used_rev = edge_eval(a_i, b_i)
                rev_flags.append(used_rev)
            return idx_path, rev_flags

        expansions += 1
        if expansions > expansion_limit:
            return None

        for v in nbrs[u]:
            if v in closed:
                continue
            ok, used_rev = edge_eval(u, v)
            if not ok:
                continue
            w = edge_cost_fn(nodes[u], nodes[v], used_rev)
            ng = g + w
            if (v not in gbest) or (ng < gbest[v] - 1e-9):
                gbest[v] = ng
                heapq.heappush(pq, (ng + h(v), ng, v, u))

    return None

# ============================================================
# GLOBAL STATE (will be overwritten by rerandomize)
# ============================================================
world_seed = BASE_SEED
gates = []
all_walls = []
WORLD_BOUNDS = None
A_pts = []
D_pts = []
H_req = []

nodes = []
nbrs = []

current_gate = 0
is_docked = True
rover_state = (0.0, 0.0, 0.0)

executed_xy = []
planned_playback = None
play_idx = 0
moving = False
goal_gate = None

# gate artists to remove/rebuild
gate_artists = []

gate_colors = [
    (0.2, 0.7, 0.9),
    (0.9, 0.55, 0.2),
    (0.55, 0.9, 0.35),
    (0.85, 0.35, 0.75),
    (0.95, 0.9, 0.25),
]

# ============================================================
# PLANNING
# ============================================================
def plan_free_space(start_pose, goal_pose):
    tmp_nodes = nodes + [start_pose, goal_pose]
    start_idx = len(tmp_nodes) - 2
    goal_idx  = len(tmp_nodes) - 1

    tmp_nbrs = [list(lst) for lst in nbrs] + [[], []]

    start_neighbors = connect_temp_neighbors(start_pose, nodes, TEMP_K)
    goal_neighbors  = connect_temp_neighbors(goal_pose, nodes, TEMP_K)

    tmp_nbrs[start_idx] = start_neighbors
    tmp_nbrs[goal_idx]  = goal_neighbors

    for j in start_neighbors:
        tmp_nbrs[j] = list(set(tmp_nbrs[j] + [start_idx]))
    for j in goal_neighbors:
        tmp_nbrs[j] = list(set(tmp_nbrs[j] + [goal_idx]))

    def edge_free(a, b):
        return se2_edge_free_with_reverse(a, b, S, all_walls, WORLD_BOUNDS, BOUND_MARGIN)

    result = astar_lazy(
        tmp_nodes, tmp_nbrs,
        start_idx, goal_idx,
        edge_free_fn=edge_free,
        edge_cost_fn=se2_edge_cost
    )
    if result is None:
        return None

    idx_path, rev_flags = result

    playback = []
    for (a_i, b_i), used_rev in zip(zip(idx_path[:-1], idx_path[1:]), rev_flags):
        a = tmp_nodes[a_i]
        b = tmp_nodes[b_i]
        seg = densify_edge(a, b, used_reverse=used_rev, ds=0.06, dth=0.12)
        if playback:
            playback.extend(seg[1:])
        else:
            playback.extend(seg)

    return playback

def plan_to_gate(goal_gate_idx):
    global current_gate, is_docked, rover_state

    cur = current_gate
    cur_th = H_req[cur]
    cur_dock = D_pts[cur]
    cur_app  = A_pts[cur]

    goal_th = H_req[goal_gate_idx]
    goal_dock = D_pts[goal_gate_idx]
    goal_app  = A_pts[goal_gate_idx]

    playback = []

    # UNDock
    if is_docked:
        undock = densify_straight_pose((cur_dock[0], cur_dock[1]), (cur_app[0], cur_app[1]), cur_th, ds=0.05)
        for x, y, th in undock:
            if square_collides_walls((x, y), th, S, all_walls):
                return None
        playback.extend(undock)
        start_pose = (cur_app[0], cur_app[1], cur_th)
    else:
        start_pose = rover_state

    # Free-space plan approach->approach
    goal_pose = (goal_app[0], goal_app[1], goal_th)
    free_path = plan_free_space(start_pose, goal_pose)
    if free_path is None:
        return None
    if playback:
        playback.extend(free_path[1:])
    else:
        playback.extend(free_path)

    # Dock straight
    dock = densify_straight_pose((goal_app[0], goal_app[1]), (goal_dock[0], goal_dock[1]), goal_th, ds=0.05)
    for x, y, th in dock:
        if square_collides_walls((x, y), th, S, all_walls):
            return None
    playback.extend(dock[1:])
    return playback

# ============================================================
# FIGURE SETUP
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 9.5))
plt.subplots_adjust(bottom=0.12)  # room for button

ax.set_aspect("equal", "box")
ax.set_facecolor((0.06, 0.07, 0.09))
fig.patch.set_facecolor((0.06, 0.07, 0.09))
for spine in ax.spines.values():
    spine.set_color((0.35, 0.35, 0.4))
ax.tick_params(colors=(0.75, 0.75, 0.8))
ax.grid(True, alpha=0.12)

plan_line, = ax.plot([], [], linestyle="--", linewidth=2.2, alpha=0.8,
                     color=(0.9, 0.9, 0.95))
trail_line, = ax.plot([], [], linewidth=2.0, alpha=0.55,
                      color=(0.7, 0.75, 0.9))

rover_poly = Polygon(square_corners((0, 0), 0.0, S),
                     closed=True, facecolor=(0.92, 0.92, 0.95),
                     edgecolor=(0.1, 0.1, 0.1), linewidth=2.2, alpha=0.95)
ax.add_patch(rover_poly)

# Button
btn_ax = fig.add_axes([0.12, 0.03, 0.22, 0.055])
btn = Button(btn_ax, "Rerandomize gates")

# ============================================================
# DRAW / RESET
# ============================================================
def clear_gate_artists():
    global gate_artists
    for a in gate_artists:
        try:
            a.remove()
        except Exception:
            pass
    gate_artists = []

def build_prm_for_current_world(seed):
    rng = np.random.default_rng(seed + 12345)
    xmin, xmax, ymin, ymax = WORLD_BOUNDS

    local_nodes = []
    while len(local_nodes) < PRM_SAMPLES:
        x = rng.uniform(xmin + BOUND_MARGIN, xmax - BOUND_MARGIN)
        y = rng.uniform(ymin + BOUND_MARGIN, ymax - BOUND_MARGIN)
        th = rng.uniform(-np.pi, np.pi)
        if square_collides_walls((x, y), th, S, all_walls):
            continue
        local_nodes.append((x, y, th))

    # Add approach poses as landmarks
    for i in range(NUM_GATES):
        local_nodes.append((A_pts[i][0], A_pts[i][1], H_req[i]))

    local_nbrs = build_knn_lists(local_nodes, PRM_K)
    return local_nodes, local_nbrs

def redraw_world():
    global rover_state, executed_xy, planned_playback, play_idx, moving, goal_gate

    xmin, xmax, ymin, ymax = WORLD_BOUNDS
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    clear_gate_artists()

    # Draw gates
    for i, g in enumerate(gates):
        col = gate_colors[i % len(gate_colors)]
        for (w0, w1) in g.walls_world():
            ln, = ax.plot([w0[0], w1[0]], [w0[1], w1[1]],
                          linewidth=3.2, alpha=0.9, color=col, solid_capstyle="round")
            gate_artists.append(ln)

        p1, p2 = g.opening_endpoints_world()
        sc1 = ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=24, color=col, alpha=0.95)
        gate_artists.append(sc1)

        scA = ax.scatter([A_pts[i][0]], [A_pts[i][1]], s=70, marker="x", color=(1,1,1), alpha=0.75)
        gate_artists.append(scA)

        scD = ax.scatter([D_pts[i][0]], [D_pts[i][1]], s=55, marker="o",
                         edgecolors=col, facecolors="none", linewidths=2)
        gate_artists.append(scD)

        txt = ax.text(g.center[0], g.center[1], f"G{i}",
                      color=(0.95,0.95,0.98), ha="center", va="center",
                      fontsize=11, alpha=0.95)
        gate_artists.append(txt)

    # Reset path visuals
    plan_line.set_data([], [])
    trail_line.set_data([], [])

    rover_poly.set_xy(square_corners((rover_state[0], rover_state[1]), rover_state[2], S))
    executed_xy = [np.array([rover_state[0], rover_state[1]])]

    planned_playback = None
    play_idx = 0
    moving = False
    goal_gate = None

    ax.set_title("Click a gate: UNDock → Lazy PRM (reverse allowed) → Dock",
                 color=(0.9, 0.9, 0.95), pad=12)

    fig.canvas.draw_idle()

def reset_world(new_seed):
    global world_seed, gates, all_walls, WORLD_BOUNDS, A_pts, D_pts, H_req
    global nodes, nbrs
    global current_gate, is_docked, rover_state, executed_xy

    world_seed = new_seed
    gates, all_walls, WORLD_BOUNDS = generate_world(NUM_GATES, world_seed)

    A_pts = [g.approach_center(APPROACH_L) for g in gates]
    D_pts = [g.dock_center()              for g in gates]
    H_req = [g.required_heading()         for g in gates]

    nodes, nbrs = build_prm_for_current_world(world_seed)

    current_gate = 0
    is_docked = True
    rover_state = (D_pts[current_gate][0], D_pts[current_gate][1], H_req[current_gate])

    redraw_world()

# ============================================================
# INPUT
# ============================================================
def nearest_gate_click(xy):
    d = [np.linalg.norm(D_pts[i] - xy) for i in range(NUM_GATES)]
    i = int(np.argmin(d))
    return i, float(d[i])

def on_click(event):
    global planned_playback, play_idx, moving, goal_gate
    if event.inaxes != ax or moving:
        return
    click = np.array([event.xdata, event.ydata], dtype=float)
    gidx, dist = nearest_gate_click(click)
    if dist > 1.6*S:
        return
    if gidx == current_gate:
        return

    ax.set_title(f"Planning: Gate {current_gate} → Gate {gidx} ...", color=(0.9,0.9,0.95))
    fig.canvas.draw_idle()

    plan = plan_to_gate(gidx)
    if plan is None:
        ax.set_title("No path found. If frequent: increase PRM_SAMPLES/PRM_K.", color=(0.95, 0.6, 0.6))
        fig.canvas.draw_idle()
        return

    planned_playback = plan
    play_idx = 0
    moving = True
    goal_gate = gidx

    xs = [p[0] for p in planned_playback]
    ys = [p[1] for p in planned_playback]
    plan_line.set_data(xs, ys)

    ax.set_title(f"Executing: Gate {current_gate} → Gate {gidx}", color=(0.9,0.9,0.95))
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)

def on_rerandomize(_event):
    # bump seed so you get a new world every click
    reset_world(world_seed + 1)

btn.on_clicked(on_rerandomize)

# ============================================================
# ANIMATION
# ============================================================
def animate(_):
    global rover_state, planned_playback, play_idx, moving, current_gate, goal_gate, is_docked, executed_xy
    if moving and planned_playback is not None:
        for _k in range(PLAYBACK_SKIP):
            if play_idx >= len(planned_playback):
                break
            rover_state = planned_playback[play_idx]
            play_idx += 1

        executed_xy.append(np.array([rover_state[0], rover_state[1]]))

        if play_idx >= len(planned_playback):
            rover_state = (D_pts[goal_gate][0], D_pts[goal_gate][1], H_req[goal_gate])
            executed_xy.append(np.array([rover_state[0], rover_state[1]]))
            current_gate = goal_gate
            goal_gate = None
            planned_playback = None
            moving = False
            is_docked = True
            plan_line.set_data([], [])
            ax.set_title(f"Arrived at Gate {current_gate}. Click the next gate.",
                         color=(0.9,0.9,0.95))

    rover_poly.set_xy(square_corners((rover_state[0], rover_state[1]), rover_state[2], S))
    t = np.array(executed_xy)
    trail_line.set_data(t[:, 0], t[:, 1])
    return rover_poly, plan_line, trail_line

# ============================================================
# BOOT
# ============================================================
reset_world(BASE_SEED)

ani = FuncAnimation(fig, animate, interval=FRAME_MS, blit=False)
plt.show()
