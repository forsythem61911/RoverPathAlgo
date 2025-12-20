import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import math, random
from dataclasses import dataclass
import heapq

# ============================================================
# CONFIG
# ============================================================
BASE_SEED = 7
NUM_GATES = 5
MIN_GATES = 2
MAX_GATES = 10

S = 1.0
DEPTH = 2.2 * S
APPROACH_L = 1.35 * S
MIN_GATE_SEP = 2.8 * S

WORLD_HALFSPAN = 9.0
BOUND_MARGIN = 1.5 * S

EPS = 2e-3

# Lazy PRM
PRM_SAMPLES = 900
PRM_K = 20
TEMP_K = 28
A_STAR_EXPANSION_LIMIT = 12000

# Collision sampling
POS_RES = 0.08
TH_RES  = 0.12

# Costs
W_TURN = 0.18
W_REVERSE = 0.05

# Animation
FRAME_MS = 20
PLAYBACK_SKIP = 2

# Smoothing controls
SMOOTH_ENABLED_DEFAULT = True
SHORTCUT_TRIES = 220          # higher => fewer turns (more checks)
CHAIKIN_ITERS_MAX = 2         # 0,1,2...
RESAMPLE_STEP = 0.06          # distance per playback step (controls speed)
END_LOCK_TOL = 0.08           # tolerance for detecting approach in playback

# ============================================================
# BASIC MATH
# ============================================================
def max_rovers_for_gates(num_gates):
    return max(1, num_gates // 2)

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

        # docking & undocking must work as straight segments
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
# LAZY PRM
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
# PATH SMOOTHING (shortcut + Chaikin + constant speed)
# ============================================================
def unique_xy(points, tol=1e-6):
    out = []
    last = None
    for p in points:
        if last is None or (abs(p[0]-last[0]) > tol or abs(p[1]-last[1]) > tol):
            out.append(np.array([p[0], p[1]], dtype=float))
            last = p
    return out

def chaikin(points_xy, iters):
    # Chaikin corner cutting on open polyline
    pts = [np.array(p, dtype=float) for p in points_xy]
    for _ in range(iters):
        if len(pts) < 3:
            return pts
        new_pts = [pts[0]]
        for i in range(len(pts)-1):
            p = pts[i]
            q = pts[i+1]
            Q = 0.75*p + 0.25*q
            R = 0.25*p + 0.75*q
            new_pts.extend([Q, R])
        new_pts[-1] = pts[-1]
        pts = new_pts
    return pts

def resample_polyline(points_xy, step):
    pts = [np.array(p, dtype=float) for p in points_xy]
    if len(pts) < 2:
        return pts

    # cumulative arc length
    segs = [np.linalg.norm(pts[i+1]-pts[i]) for i in range(len(pts)-1)]
    L = float(np.sum(segs))
    if L < 1e-9:
        return [pts[0]]

    n = max(2, int(L/step) + 1)
    targets = np.linspace(0.0, L, n)

    out = [pts[0]]
    cur_seg = 0
    cur_s = 0.0
    for tL in targets[1:]:
        while cur_seg < len(segs) and (cur_s + segs[cur_seg]) < tL - 1e-12:
            cur_s += segs[cur_seg]
            cur_seg += 1
        if cur_seg >= len(segs):
            out.append(pts[-1])
            continue
        a = pts[cur_seg]
        b = pts[cur_seg+1]
        segL = segs[cur_seg]
        if segL < 1e-12:
            out.append(a.copy())
            continue
        u = (tL - cur_s) / segL
        out.append(a + u*(b-a))
    return out

def tangent_headings(resampled_xy, th_start=None, th_end=None):
    xy = [np.array(p, dtype=float) for p in resampled_xy]
    n = len(xy)
    th = []
    for i in range(n):
        if i == n-1:
            v = xy[i] - xy[i-1]
        else:
            v = xy[i+1] - xy[i]
        ang = math.atan2(v[1], v[0]) if np.linalg.norm(v) > 1e-12 else (th[-1] if th else 0.0)
        th.append(ang)

    # softly lock endpoints (to match approach headings)
    if th_start is not None:
        th[0] = th_start
        if n > 1:
            th[1] = wrap_angle(0.7*th[1] + 0.3*th_start)
    if th_end is not None:
        th[-1] = th_end
        if n > 1:
            th[-2] = wrap_angle(0.7*th[-2] + 0.3*th_end)
    return th

def validate_states(states, walls, bounds, margin):
    for x, y, th in states:
        if not within_bounds((x, y), bounds, margin):
            return False
        if square_collides_walls((x, y), th, S, walls):
            return False
    return True

def smooth_free_segment(raw_states, th_start, th_end, walls, bounds, margin):
    """
    raw_states: list[(x,y,th)] for the middle free-space portion.
    Returns smoothed states with approx constant spatial step, or None.
    """
    raw_xy = unique_xy([(s[0], s[1]) for s in raw_states])
    if len(raw_xy) < 3:
        # no point smoothing
        out = [(p[0], p[1], th_start) for p in raw_xy]
        return out

    for iters in range(CHAIKIN_ITERS_MAX, -1, -1):
        pts = chaikin(raw_xy, iters)
        pts = resample_polyline(pts, RESAMPLE_STEP)
        ths = tangent_headings(pts, th_start=th_start, th_end=th_end)
        states = [(float(p[0]), float(p[1]), float(ths[i])) for i, p in enumerate(pts)]
        if validate_states(states, walls, bounds, margin):
            return states
    return None

def shortcut_path(states, walls, bounds, margin, tries=SHORTCUT_TRIES):
    """
    Shortcut the free-space portion using our exact edge checker (rotate/drive/reverse),
    then densify again.
    """
    if len(states) < 4:
        return states

    # pick "key poses" sparsely: use every ~6th point to reduce overhead
    key = states[::6]
    if np.linalg.norm(np.array([states[-1][0],states[-1][1]]) - np.array([key[-1][0],key[-1][1]])) > 1e-9:
        key.append(states[-1])

    key = [(k[0], k[1], k[2]) for k in key]

    for _ in range(tries):
        if len(key) < 3:
            break
        i = random.randint(0, len(key)-3)
        j = random.randint(i+2, len(key)-1)
        a = key[i]
        b = key[j]
        ok, used_rev = se2_edge_free_with_reverse(a, b, S, walls, bounds, margin)
        if not ok:
            continue
        # if valid, remove intermediate nodes i+1..j-1
        key = key[:i+1] + key[j:]

    # densify the shortcut keypath into a new state list
    out = []
    for a, b in zip(key[:-1], key[1:]):
        ok, used_rev = se2_edge_free_with_reverse(a, b, S, walls, bounds, margin)
        if not ok:
            # fallback: if shortcut produced something inconsistent, just bail out
            return states
        seg = densify_edge(a, b, used_reverse=used_rev, ds=0.06, dth=0.12)
        if out:
            out.extend(seg[1:])
        else:
            out.extend(seg)
    return out

# ============================================================
# GLOBAL STATE
# ============================================================
world_seed = BASE_SEED
num_rovers = max_rovers_for_gates(NUM_GATES)
gates = []
all_walls = []
WORLD_BOUNDS = None
A_pts = []
D_pts = []
H_req = []

nodes = []
nbrs = []

rover_current_gates = []
rover_last_gates = []
rover_docked = []
rover_states = []
rover_executed_xy = []
rover_plans = []
rover_play_idx = []
rover_moving = []
rover_goal_gates = []
rover_intersection_max_idx = []
rover_intersections = []
playing = False
intersections_dirty = True

gate_artists = []
rover_artists = []
rover_trails = []
rover_plan_lines = []
smooth_enabled = SMOOTH_ENABLED_DEFAULT

gate_colors = [
    (0.2, 0.7, 0.9),
    (0.9, 0.55, 0.2),
    (0.55, 0.9, 0.35),
    (0.85, 0.35, 0.75),
    (0.95, 0.9, 0.25),
]

rover_colors = [
    (0.92, 0.92, 0.95),
    (0.9, 0.6, 0.65),
    (0.55, 0.85, 0.75),
    (0.85, 0.75, 0.45),
    (0.65, 0.7, 0.95),
]

# ============================================================
# PLANNING
# ============================================================
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

    # add approach poses as landmarks
    for i in range(NUM_GATES):
        local_nodes.append((A_pts[i][0], A_pts[i][1], H_req[i]))

    local_nbrs = build_knn_lists(local_nodes, PRM_K)
    return local_nodes, local_nbrs

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

def smooth_playback(full_playback, cur_gate, goal_gate_idx):
    """
    Keep UNDock and Dock rigid.
    Smooth only the middle free-space segment.
    """
    if not smooth_enabled:
        return full_playback

    cur_app = A_pts[cur_gate]
    goal_app = A_pts[goal_gate_idx]
    cur_th = H_req[cur_gate]
    goal_th = H_req[goal_gate_idx]

    # find first index near cur_app (end of undock)
    end_undock = None
    for i, (x,y,th) in enumerate(full_playback):
        if (x-cur_app[0])**2 + (y-cur_app[1])**2 < END_LOCK_TOL**2:
            end_undock = i
            break
    # find last index near goal_app (start of dock)
    start_dock = None
    for i in range(len(full_playback)-1, -1, -1):
        x,y,th = full_playback[i]
        if (x-goal_app[0])**2 + (y-goal_app[1])**2 < END_LOCK_TOL**2:
            start_dock = i
            break

    if end_undock is None or start_dock is None or start_dock <= end_undock + 5:
        return full_playback

    prefix = full_playback[:end_undock+1]
    middle = full_playback[end_undock:start_dock+1]
    suffix = full_playback[start_dock:]

    # 1) shortcut middle to reduce stop/go
    middle2 = shortcut_path(middle, all_walls, WORLD_BOUNDS, BOUND_MARGIN, tries=SHORTCUT_TRIES)

    # 2) smooth XY and constant-speed resample; compute tangent headings
    sm = smooth_free_segment(middle2, th_start=cur_th, th_end=goal_th,
                             walls=all_walls, bounds=WORLD_BOUNDS, margin=BOUND_MARGIN)
    if sm is None:
        return full_playback

    # stitch, ensuring continuity without duplicate points
    out = prefix[:-1] + sm + suffix[1:]
    if not validate_states(out, all_walls, WORLD_BOUNDS, BOUND_MARGIN):
        return full_playback
    return out

def plan_to_gate(rover_idx, goal_gate_idx):
    cur = rover_current_gates[rover_idx]
    cur_th = H_req[cur]
    cur_dock = D_pts[cur]
    cur_app = A_pts[cur]

    goal_th = H_req[goal_gate_idx]
    goal_dock = D_pts[goal_gate_idx]
    goal_app = A_pts[goal_gate_idx]

    playback = []

    # UNDock
    if rover_docked[rover_idx]:
        undock = densify_straight_pose((cur_dock[0], cur_dock[1]), (cur_app[0], cur_app[1]), cur_th, ds=0.05)
        for x, y, th in undock:
            if square_collides_walls((x, y), th, S, all_walls):
                return None
        playback.extend(undock)
        start_pose = (cur_app[0], cur_app[1], cur_th)
    else:
        start_pose = rover_states[rover_idx]

    # Free-space plan approach->approach
    goal_pose = (goal_app[0], goal_app[1], goal_th)
    free_path = plan_free_space(start_pose, goal_pose)
    if free_path is None:
        return None
    playback.extend(free_path[1:])  # connect

    # Dock straight
    dock = densify_straight_pose((goal_app[0], goal_app[1]), (goal_dock[0], goal_dock[1]), goal_th, ds=0.05)
    for x, y, th in dock:
        if square_collides_walls((x, y), th, S, all_walls):
            return None
    playback.extend(dock[1:])

    # Smooth + constant-speed in the free-space segment
    playback2 = smooth_playback(playback, cur_gate=cur, goal_gate_idx=goal_gate_idx)
    return playback2

# ============================================================
# FIGURE / UI
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 9.5))
plt.subplots_adjust(bottom=0.14, right=0.82)

ax.set_aspect("equal", "box")
ax.set_facecolor((0.06, 0.07, 0.09))
fig.patch.set_facecolor((0.06, 0.07, 0.09))
for spine in ax.spines.values():
    spine.set_color((0.35, 0.35, 0.4))
ax.tick_params(colors=(0.75, 0.75, 0.8))
ax.grid(True, alpha=0.12)

# rover artists are initialized in reset_rovers()

# Buttons
btn_ax1 = fig.add_axes([0.1, 0.04, 0.22, 0.06])
btn_rerand = Button(btn_ax1, "Rerandomize gates")

btn_ax2 = fig.add_axes([0.34, 0.04, 0.18, 0.06])
btn_smooth = Button(btn_ax2, "Smooth: ON" if smooth_enabled else "Smooth: OFF")

btn_ax3 = fig.add_axes([0.54, 0.04, 0.14, 0.06])
btn_play = Button(btn_ax3, "Play")

slider_ax_gates = fig.add_axes([0.86, 0.2, 0.03, 0.68])
slider_ax_rovers = fig.add_axes([0.92, 0.2, 0.03, 0.68])

slider_gates = Slider(
    slider_ax_gates,
    "Docks",
    MIN_GATES,
    MAX_GATES,
    valinit=NUM_GATES,
    valstep=1,
    orientation="vertical",
)

slider_rovers = Slider(
    slider_ax_rovers,
    "Rovers",
    1,
    max_rovers_for_gates(NUM_GATES),
    valinit=num_rovers,
    valstep=1,
    orientation="vertical",
)

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

def clear_rover_artists():
    global rover_artists, rover_trails, rover_plan_lines
    for a in rover_artists + rover_trails + rover_plan_lines:
        try:
            a.remove()
        except Exception:
            pass
    rover_artists = []
    rover_trails = []
    rover_plan_lines = []

def reset_rovers(count):
    global rover_current_gates, rover_last_gates, rover_docked, rover_states
    global rover_executed_xy, rover_plans, rover_play_idx, rover_moving, rover_goal_gates
    global rover_intersection_max_idx, rover_intersections, playing, intersections_dirty

    clear_rover_artists()

    rover_current_gates = []
    rover_last_gates = []
    rover_docked = []
    rover_states = []
    rover_executed_xy = []
    rover_plans = []
    rover_play_idx = []
    rover_moving = []
    rover_goal_gates = []
    rover_intersection_max_idx = []
    rover_intersections = []

    for i in range(count):
        gate_idx = i % NUM_GATES
        rover_current_gates.append(gate_idx)
        rover_last_gates.append(None)
        rover_docked.append(True)
        rover_states.append((D_pts[gate_idx][0], D_pts[gate_idx][1], H_req[gate_idx]))
        rover_executed_xy.append([np.array([D_pts[gate_idx][0], D_pts[gate_idx][1]])])
        rover_plans.append(None)
        rover_play_idx.append(0)
        rover_moving.append(False)
        rover_goal_gates.append(None)
        rover_intersection_max_idx.append(-1)
        rover_intersections.append(set())

        col = rover_colors[i % len(rover_colors)]
        rover_poly = Polygon(square_corners((0, 0), 0.0, S),
                             closed=True, facecolor=col,
                             edgecolor=(0.1, 0.1, 0.1), linewidth=2.2, alpha=0.95)
        rover_poly.set_xy(square_corners((rover_states[i][0], rover_states[i][1]), rover_states[i][2], S))
        ax.add_patch(rover_poly)
        rover_artists.append(rover_poly)

        trail_line, = ax.plot([], [], linewidth=2.0, alpha=0.55, color=col)
        rover_trails.append(trail_line)

        plan_line, = ax.plot([], [], linestyle="--", linewidth=2.2, alpha=0.5, color=col)
        rover_plan_lines.append(plan_line)

    playing = False
    btn_play.label.set_text("Play")
    intersections_dirty = True

def pick_next_gate(rover_idx):
    cur = rover_current_gates[rover_idx]
    last = rover_last_gates[rover_idx]
    occupied = {
        rover_current_gates[i]
        for i in range(len(rover_current_gates))
        if i != rover_idx and rover_docked[i]
    }
    candidates = [
        i for i in range(NUM_GATES)
        if i != cur and i != last and i not in occupied
    ]
    if not candidates:
        candidates = [i for i in range(NUM_GATES) if i != cur and i not in occupied]
    if not candidates:
        return cur

    cur_dock = D_pts[cur]
    distances = [(i, np.linalg.norm(D_pts[i] - cur_dock)) for i in candidates]
    distances.sort(key=lambda x: (-x[1], x[0]))
    return distances[0][0]

def compute_intersections():
    n = len(rover_plans)
    max_idx = [-1 for _ in range(n)]
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        path_i = rover_plans[i]
        if not path_i or len(path_i) < 2:
            continue
        segs_i = list(zip(path_i[:-1], path_i[1:]))
        for j in range(i + 1, n):
            path_j = rover_plans[j]
            if not path_j or len(path_j) < 2:
                continue
            segs_j = list(zip(path_j[:-1], path_j[1:]))
            for idx_i, (a, b) in enumerate(segs_i):
                a_xy = (a[0], a[1])
                b_xy = (b[0], b[1])
                for idx_j, (c, d) in enumerate(segs_j):
                    c_xy = (c[0], c[1])
                    d_xy = (d[0], d[1])
                    if seg_intersect(a_xy, b_xy, c_xy, d_xy):
                        adjacency[i].add(j)
                        adjacency[j].add(i)
                        max_idx[i] = max(max_idx[i], idx_i + 1)
                        max_idx[j] = max(max_idx[j], idx_j + 1)
    return max_idx, adjacency

def plan_next_rover_path(rover_idx):
    global intersections_dirty
    goal = pick_next_gate(rover_idx)
    if goal == rover_current_gates[rover_idx]:
        return False
    plan = plan_to_gate(rover_idx, goal)
    if plan is None:
        return False
    rover_plans[rover_idx] = plan
    rover_play_idx[rover_idx] = 0
    rover_moving[rover_idx] = True
    rover_goal_gates[rover_idx] = goal
    rover_plan_lines[rover_idx].set_data([p[0] for p in plan], [p[1] for p in plan])
    intersections_dirty = True
    return True

def intersection_components(adjacency):
    n = len(adjacency)
    visited = set()
    comps = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for v in adjacency[u]:
                if v not in visited:
                    stack.append(v)
        comps.append(comp)
    return comps

def redraw_world():
    xmin, xmax, ymin, ymax = WORLD_BOUNDS
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    clear_gate_artists()

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

    for i in range(len(rover_states)):
        rover_plan_lines[i].set_data([], [])
        rover_trails[i].set_data([], [])
        rover_artists[i].set_xy(square_corners((rover_states[i][0], rover_states[i][1]),
                                               rover_states[i][2], S))

    ax.set_title("Play: rovers navigate between furthest docks",
                 color=(0.9, 0.9, 0.95), pad=12)
    fig.canvas.draw_idle()

def reset_world(new_seed):
    global world_seed, gates, all_walls, WORLD_BOUNDS, A_pts, D_pts, H_req
    global nodes, nbrs

    world_seed = new_seed
    gates, all_walls, WORLD_BOUNDS = generate_world(NUM_GATES, world_seed)

    A_pts = [g.approach_center(APPROACH_L) for g in gates]
    D_pts = [g.dock_center()              for g in gates]
    H_req = [g.required_heading()         for g in gates]

    nodes, nbrs = build_prm_for_current_world(world_seed)

    reset_rovers(num_rovers)
    redraw_world()

def on_rerandomize(_event):
    reset_world(world_seed + 1)

def on_toggle_smooth(_event):
    global smooth_enabled
    smooth_enabled = not smooth_enabled
    btn_smooth.label.set_text("Smooth: ON" if smooth_enabled else "Smooth: OFF")
    fig.canvas.draw_idle()

def on_toggle_play(_event):
    global playing
    playing = not playing
    btn_play.label.set_text("Pause" if playing else "Play")
    fig.canvas.draw_idle()

def on_gate_slider(val):
    global NUM_GATES, num_rovers
    new_gates = int(val)
    if new_gates == NUM_GATES:
        return
    NUM_GATES = new_gates
    max_rovers = max_rovers_for_gates(NUM_GATES)
    num_rovers = min(num_rovers, max_rovers)
    slider_rovers.eventson = False
    slider_rovers.valmax = max_rovers
    slider_rovers.ax.set_ylim(slider_rovers.valmin, max_rovers)
    slider_rovers.set_val(num_rovers)
    slider_rovers.eventson = True
    reset_world(world_seed)

def on_rover_slider(val):
    global num_rovers
    new_rovers = int(val)
    if new_rovers == num_rovers:
        return
    num_rovers = new_rovers
    reset_rovers(num_rovers)
    redraw_world()

btn_rerand.on_clicked(on_rerandomize)
btn_smooth.on_clicked(on_toggle_smooth)
btn_play.on_clicked(on_toggle_play)

slider_gates.on_changed(on_gate_slider)
slider_rovers.on_changed(on_rover_slider)

# ============================================================
# ANIMATION
# ============================================================
def animate(_):
    global intersections_dirty, rover_intersection_max_idx, rover_intersections

    if playing:
        for i in range(len(rover_states)):
            if not rover_moving[i] and rover_plans[i] is None:
                plan_next_rover_path(i)

    if intersections_dirty:
        rover_intersection_max_idx, rover_intersections = compute_intersections()
        intersections_dirty = False

    allowed_to_move = set(range(len(rover_states)))
    if rover_intersections:
        components = intersection_components(rover_intersections)
        allowed_to_move = set()
        for comp in components:
            pending = [
                i for i in comp
                if rover_moving[i]
                and rover_intersection_max_idx[i] >= 0
                and rover_play_idx[i] <= rover_intersection_max_idx[i]
            ]
            if pending:
                allowed_to_move.add(min(pending))
            else:
                for i in comp:
                    allowed_to_move.add(i)

    for i in range(len(rover_states)):
        if rover_moving[i] and rover_plans[i] is not None:
            if (i in allowed_to_move) or not rover_intersections[i]:
                for _k in range(PLAYBACK_SKIP):
                    if rover_play_idx[i] >= len(rover_plans[i]):
                        break
                    rover_states[i] = rover_plans[i][rover_play_idx[i]]
                    rover_play_idx[i] += 1
                rover_executed_xy[i].append(np.array([rover_states[i][0], rover_states[i][1]]))

                if rover_play_idx[i] >= len(rover_plans[i]):
                    goal_gate = rover_goal_gates[i]
                    rover_states[i] = (D_pts[goal_gate][0], D_pts[goal_gate][1], H_req[goal_gate])
                    rover_executed_xy[i].append(np.array([rover_states[i][0], rover_states[i][1]]))
                    rover_last_gates[i] = rover_current_gates[i]
                    rover_current_gates[i] = goal_gate
                    rover_goal_gates[i] = None
                    rover_plans[i] = None
                    rover_play_idx[i] = 0
                    rover_moving[i] = False
                    rover_docked[i] = True
                    rover_plan_lines[i].set_data([], [])
                    intersections_dirty = True

    for i in range(len(rover_states)):
        rover_artists[i].set_xy(square_corners((rover_states[i][0], rover_states[i][1]),
                                               rover_states[i][2], S))
        t = np.array(rover_executed_xy[i])
        rover_trails[i].set_data(t[:, 0], t[:, 1])

    return rover_artists + rover_plan_lines + rover_trails

# ============================================================
# BOOT
# ============================================================
reset_world(BASE_SEED)
ani = FuncAnimation(fig, animate, interval=FRAME_MS, blit=False)
plt.show()
