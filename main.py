import math, random, heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# ===================== CONFIG =====================
BASE_SEED = 7
NUM_GATES = 5

S = 1.0
DEPTH = 2.2 * S
APPROACH_L = 1.35 * S
MIN_GATE_SEP = 2.8 * S

WORLD_HALFSPAN = 9.0
BOUND_MARGIN = 1.5 * S

# PRM
PRM_SAMPLES = 900
PRM_K = 20
TEMP_K = 28
A_STAR_EXPANSION_LIMIT = 12000

# costs
W_TURN = 0.18

# animation
FRAME_MS = 20
PLAYBACK_SKIP = 2

# smoothing
SMOOTH_DEFAULT = True
SHORTCUT_TRIES = 220
CHAIKIN_ITERS_MAX = 2
RESAMPLE_STEP = 0.06
END_LOCK_TOL = 0.08

# perf / UI
TRAIL_MAX = 2500
TITLE_EVERY = 6

# rover collision geometry:
# wall collision can use slightly-inscribed circle (lets you fit gates tightly)
ROVER_R_WALL = (S / 2.0) * 0.95
# rover-rover must use circumscribed circle of square to prevent overlaps
ROVER_R_ROVER = (math.sqrt(2) / 2.0) * S * 0.99
ROVER_SEP = 2.0 * ROVER_R_ROVER * 1.03
ROVER_SEP2 = ROVER_SEP * ROVER_SEP


# ===================== MATH / GEOM =====================

def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def rot2(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]], float)

def square_corners(c, t, side):
    x, y = c
    h = side/2.0
    C, S_ = math.cos(t), math.sin(t)
    return np.array([
        [x + (-h)*C - (-h)*S_, y + (-h)*S_ + (-h)*C],
        [x + ( h)*C - (-h)*S_, y + ( h)*S_ + (-h)*C],
        [x + ( h)*C - ( h)*S_, y + ( h)*S_ + ( h)*C],
        [x + (-h)*C - ( h)*S_, y + (-h)*S_ + ( h)*C],
    ], float)

def within(xy, b, m):
    x, y = float(xy[0]), float(xy[1])
    xmin, xmax, ymin, ymax = b
    return xmin+m <= x <= xmax-m and ymin+m <= y <= ymax-m

def pt_seg_d2(p, a, b):
    ap = p-a
    ab = b-a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return float(np.dot(ap, ap))
    t = float(np.dot(ap, ab)/ab2)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    d = p - (a + t*ab)
    return float(np.dot(d, d))

def seg_seg_d2(a, b, c, d):
    # 2D segment–segment minimum distance squared
    return min(
        pt_seg_d2(a, c, d), pt_seg_d2(b, c, d),
        pt_seg_d2(c, a, b), pt_seg_d2(d, a, b)
    )

def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_seg(a, b, p, eps=1e-12):
    return (min(a[0], b[0])-eps <= p[0] <= max(a[0], b[0])+eps and
            min(a[1], b[1])-eps <= p[1] <= max(a[1], b[1])+eps)

def seg_intersect(a, b, c, d, eps=1e-12):
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if abs(o1) <= eps and on_seg(a, b, c): return True
    if abs(o2) <= eps and on_seg(a, b, d): return True
    if abs(o3) <= eps and on_seg(c, d, a): return True
    if abs(o4) <= eps and on_seg(c, d, b): return True
    return (o1*o2 < 0) and (o3*o4 < 0)


# ===================== WALL INDEX =====================
class WallIndex:
    def __init__(self, walls=None):
        self.set(walls or [])

    def set(self, walls):
        self.walls = walls
        self.p0 = np.array([w[0] for w in walls], float) if walls else np.zeros((0,2), float)
        self.p1 = np.array([w[1] for w in walls], float) if walls else np.zeros((0,2), float)
        self.d = self.p1 - self.p0
        self.len2 = np.sum(self.d*self.d, axis=1) if walls else np.zeros((0,), float)
        self.len2[self.len2 < 1e-12] = 1e-12

    def _min_d2_pts(self, pts):
        if self.p0.shape[0] == 0:
            return np.full((len(pts),), np.inf, float)
        ap = pts[:,None,:] - self.p0[None,:,:]
        t = np.sum(ap * self.d[None,:,:], axis=2) / self.len2[None,:]
        t = np.clip(t, 0.0, 1.0)[:,:,None]
        closest = self.p0[None,:,:] + t * self.d[None,:,:]
        diff = pts[:,None,:] - closest
        return np.min(np.sum(diff*diff, axis=2), axis=1)

    def collides_pose(self, x, y, bounds, margin, r=ROVER_R_WALL):
        if not within((x, y), bounds, margin):
            return True
        return self._min_d2_pts(np.array([[x, y]], float))[0] < r*r

    def collides_seg(self, a_xy, b_xy, bounds, margin, r=ROVER_R_WALL):
        a = np.array(a_xy, float)
        b = np.array(b_xy, float)
        if not within(a, bounds, margin) or not within(b, bounds, margin):
            return True
        rr = r*r
        for w0, w1 in self.walls:
            c = np.array(w0, float); d = np.array(w1, float)
            if seg_intersect(a, b, c, d):
                return True
            if min(pt_seg_d2(a, c, d), pt_seg_d2(b, c, d),
                   pt_seg_d2(c, a, b), pt_seg_d2(d, a, b)) < rr:
                return True
        return False


# ===================== GATE + ROVER =====================
@dataclass
class Gate:
    center: np.ndarray
    theta: float
    s: float = S
    depth: float = DEPTH

    walls: List[Tuple[np.ndarray, np.ndarray]] = field(init=False)
    open_p1: np.ndarray = field(init=False)
    open_p2: np.ndarray = field(init=False)
    dock: np.ndarray = field(init=False)
    approach: np.ndarray = field(init=False)
    heading: float = field(init=False)

    def __post_init__(self):
        R = rot2(self.theta)
        c = self.center
        s = self.s
        d = self.depth

        self.open_p1 = c + R @ np.array([-s/2, 0.0])
        self.open_p2 = c + R @ np.array([ s/2, 0.0])

        n = R @ np.array([0.0, 1.0])
        self.heading = math.atan2(n[1], n[0])

        mid = 0.5 * (self.open_p1 + self.open_p2)
        self.dock = mid + (s/2) * n
        self.approach = self.dock - APPROACH_L * n

        L0 = c + R @ np.array([-s/2, 0.0]); L1 = c + R @ np.array([-s/2, d])
        R0 = c + R @ np.array([ s/2, 0.0]); R1 = c + R @ np.array([ s/2, d])
        B0 = c + R @ np.array([-s/2, d]);  B1 = c + R @ np.array([ s/2, d])
        self.walls = [(L0, L1), (R0, R1), (B0, B1)]


@dataclass
class Rover:
    id: int
    current_gate: int
    previous_gate: int
    state: Tuple[float, float, float]

    goal_gate: int = -1
    path: Optional[List[Tuple[float, float, float]]] = None
    idx: int = 0
    moving: bool = False
    docked: bool = True

    stuck: int = 0
    reversing: bool = False
    reverse_target: int = 0
    blocked_by: int = -1

    plan_cd: int = 0  # cooldown frames to avoid replan spikes

    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_MAX))

    poly: Optional[Polygon] = None
    line: Optional[object] = None


# ===================== PRM / PATH =====================

def knn_lists(nodes, k):
    xy = np.array([(n[0], n[1]) for n in nodes], float)
    N = len(nodes)
    out = []
    for i in range(N):
        d2 = np.sum((xy - xy[i])**2, axis=1)
        idx = np.argpartition(d2, k+1)[:k+1]
        idx = idx[np.argsort(d2[idx])]
        out.append([j for j in idx if j != i][:k])
    return out

def knn_temp(pose, base_nodes, k):
    xy = np.array([(n[0], n[1]) for n in base_nodes], float)
    x, y, _ = pose
    d2 = np.sum((xy - np.array([x, y]))**2, axis=1)
    idx = np.argpartition(d2, k)[:k]
    return idx[np.argsort(d2[idx])].tolist()

def se2_cost(a, b):
    ax, ay, ath = a
    bx, by, bth = b
    dist = math.hypot(bx-ax, by-ay)
    if dist < 1e-12:
        return W_TURN * abs(wrap(bth-ath))
    tr = math.atan2(by-ay, bx-ax)
    return dist + W_TURN * (abs(wrap(tr-ath)) + abs(wrap(bth-tr)))

def densify_edge(a, b, ds=RESAMPLE_STEP, dth=0.12):
    ax, ay, ath = a
    bx, by, bth = b
    dx, dy = bx-ax, by-ay
    dist = math.hypot(dx, dy)
    out = [(ax, ay, ath)]

    if dist < 1e-12:
        d = wrap(bth-ath)
        n = max(1, int(abs(d)/dth))
        return out + [(ax, ay, wrap(ath + (i/n)*d)) for i in range(1, n+1)]

    tr = math.atan2(dy, dx)
    d0 = wrap(tr-ath); n0 = max(1, int(abs(d0)/dth))
    out += [(ax, ay, wrap(ath + (i/n0)*d0)) for i in range(1, n0+1)]

    n1 = max(1, int(dist/ds))
    out += [(ax + (i/n1)*dx, ay + (i/n1)*dy, tr) for i in range(1, n1+1)]

    d2 = wrap(bth-tr); n2 = max(1, int(abs(d2)/dth))
    out += [(bx, by, wrap(tr + (i/n2)*d2)) for i in range(1, n2+1)]
    return out

def densify_straight(a_xy, b_xy, th, ds=0.05):
    ax, ay = a_xy
    bx, by = b_xy
    dx, dy = bx-ax, by-ay
    dist = math.hypot(dx, dy)
    n = max(1, int(dist/ds))
    return [(ax + (i/n)*dx, ay + (i/n)*dy, th) for i in range(n+1)]

def astar_lazy(nodes, nbrs, s_idx, g_idx, edge_ok, edge_cost, limit=A_STAR_EXPANSION_LIMIT):
    gx, gy, _ = nodes[g_idx]
    def h(i):
        x, y, _ = nodes[i]
        return math.hypot(gx-x, gy-y)

    cache = {}
    def ok(i, j):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in cache:
            return cache[(a, b)]
        v = edge_ok(nodes[i], nodes[j])
        cache[(a, b)] = v
        return v

    pq = [(h(s_idx), 0.0, s_idx, -1)]
    came = {}
    gbest = {s_idx: 0.0}
    closed = set()
    exp = 0

    while pq:
        _, g, u, p = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        came[u] = p

        if u == g_idx:
            path = []
            cur = u
            while cur != -1:
                path.append(cur)
                cur = came[cur]
            return path[::-1]

        exp += 1
        if exp > limit:
            return None

        for v in nbrs[u]:
            if v in closed or not ok(u, v):
                continue
            ng = g + edge_cost(nodes[u], nodes[v])
            if v not in gbest or ng < gbest[v] - 1e-9:
                gbest[v] = ng
                heapq.heappush(pq, (ng + h(v), ng, v, u))
    return None


# ===================== SMOOTHING =====================

def uniq_xy(states, tol=1e-6):
    out = []
    last = None
    for x, y, _ in states:
        if last is None or abs(x-last[0]) > tol or abs(y-last[1]) > tol:
            out.append(np.array([x, y], float))
            last = (x, y)
    return out

def chaikin(pts, iters):
    pts = [np.array(p, float) for p in pts]
    for _ in range(iters):
        if len(pts) < 3:
            return pts
        n = [pts[0]]
        for i in range(len(pts)-1):
            p, q = pts[i], pts[i+1]
            n += [0.75*p + 0.25*q, 0.25*p + 0.75*q]
        n[-1] = pts[-1]
        pts = n
    return pts

def resample(pts, step):
    pts = [np.array(p, float) for p in pts]
    if len(pts) < 2:
        return pts
    seg = [np.linalg.norm(pts[i+1]-pts[i]) for i in range(len(pts)-1)]
    L = float(np.sum(seg))
    if L < 1e-9:
        return [pts[0]]
    n = max(2, int(L/step)+1)
    targets = np.linspace(0.0, L, n)

    out = [pts[0]]
    cur = 0
    s = 0.0
    for tL in targets[1:]:
        while cur < len(seg) and s + seg[cur] < tL - 1e-12:
            s += seg[cur]
            cur += 1
        if cur >= len(seg):
            out.append(pts[-1])
            continue
        a, b = pts[cur], pts[cur+1]
        Ls = seg[cur]
        out.append(a if Ls < 1e-12 else a + ((tL-s)/Ls) * (b-a))
    return out

def tan_heads(pts, th0=None, th1=None):
    pts = [np.array(p, float) for p in pts]
    n = len(pts)
    th = []
    for i in range(n):
        v = pts[i] - pts[i-1] if i == n-1 else pts[i+1] - pts[i]
        ang = math.atan2(v[1], v[0]) if float(np.dot(v, v)) > 1e-12 else (th[-1] if th else 0.0)
        th.append(ang)
    if th0 is not None:
        th[0] = th0
        if n > 1:
            th[1] = wrap(0.7*th[1] + 0.3*th0)
    if th1 is not None:
        th[-1] = th1
        if n > 1:
            th[-2] = wrap(0.7*th[-2] + 0.3*th1)
    return th

def validate(states, WI, bounds, margin):
    for x, y, _ in states:
        if WI.collides_pose(x, y, bounds, margin):
            return False
    return True

def shortcut(states, WI, bounds, margin, tries=SHORTCUT_TRIES):
    if len(states) < 4:
        return states
    key = states[::6]
    if (key[-1][0]-states[-1][0])**2 + (key[-1][1]-states[-1][1])**2 > 1e-12:
        key.append(states[-1])
    key = [(k[0], k[1], k[2]) for k in key]

    for _ in range(tries):
        if len(key) < 3:
            break
        i = random.randint(0, len(key)-3)
        j = random.randint(i+2, len(key)-1)
        a, b = key[i], key[j]
        if WI.collides_seg((a[0], a[1]), (b[0], b[1]), bounds, margin):
            continue
        key = key[:i+1] + key[j:]

    out = []
    for a, b in zip(key[:-1], key[1:]):
        if WI.collides_seg((a[0], a[1]), (b[0], b[1]), bounds, margin):
            return states
        seg = densify_edge(a, b)
        out += seg[1:] if out else seg
    return out

def smooth_mid(playback, cur_gate, goal_gate, gates, WI, bounds, margin, enabled):
    if (not enabled) or len(playback) < 12:
        return playback

    cur_app = gates[cur_gate].approach
    goal_app = gates[goal_gate].approach
    th0 = gates[cur_gate].heading
    th1 = gates[goal_gate].heading

    end_u = next((i for i, (x, y, _) in enumerate(playback)
                  if (x-cur_app[0])**2 + (y-cur_app[1])**2 < END_LOCK_TOL**2), None)
    start_d = next((i for i in range(len(playback)-1, -1, -1)
                    if (playback[i][0]-goal_app[0])**2 + (playback[i][1]-goal_app[1])**2 < END_LOCK_TOL**2), None)

    if end_u is None or start_d is None or start_d <= end_u + 5:
        return playback

    pre = playback[:end_u+1]
    mid = playback[end_u:start_d+1]
    suf = playback[start_d:]

    mid2 = shortcut(mid, WI, bounds, margin)
    raw = uniq_xy(mid2)
    if len(raw) < 3:
        return playback

    for it in range(CHAIKIN_ITERS_MAX, -1, -1):
        pts = resample(chaikin(raw, it), RESAMPLE_STEP)
        th = tan_heads(pts, th0, th1)
        sm = [(float(p[0]), float(p[1]), float(th[i])) for i, p in enumerate(pts)]
        out = pre[:-1] + sm + suf[1:]
        if validate(out, WI, bounds, margin):
            return out
    return playback


# ===================== SIM =====================
class Sim:
    gate_colors = [
        (0.2,0.7,0.9),(0.9,0.55,0.2),(0.55,0.9,0.35),(0.85,0.35,0.75),(0.95,0.9,0.25),
        (0.6,0.8,0.7),(0.9,0.4,0.4),(0.5,0.5,0.9),(0.9,0.75,0.5),(0.7,0.4,0.7)
    ]
    rover_colors = [
        (0.92,0.92,0.95),(0.95,0.6,0.6),(0.6,0.95,0.6),(0.6,0.6,0.95),(0.95,0.95,0.6)
    ]

    def __init__(self):
        self.seed = BASE_SEED
        self.num_docks = NUM_GATES
        self.num_rovers = 1
        self.smooth = SMOOTH_DEFAULT
        self.auto = False

        self.gates = []
        self.walls = []
        self.bounds = (-WORLD_HALFSPAN, WORLD_HALFSPAN, -WORLD_HALFSPAN, WORLD_HALFSPAN)
        self.WI = WallIndex([])

        self.nodes = []
        self.nbrs = []
        self.rovers: List[Rover] = []

        self.fig = None
        self.ax = None
        self.gate_art = []
        self.rover_art = []
        self.sld_d = None
        self.sld_r = None
        self.btn_play = None
        self.btn_sm = None
        self.btn_rerand = None
        self.ani = None

        self.frame = 0

    # ----- world gen helpers -----
    def _circle_hits_walls(self, x, y, walls, r=ROVER_R_WALL):
        p = np.array([x, y], float)
        rr = r*r
        for w0, w1 in walls:
            if pt_seg_d2(p, np.array(w0, float), np.array(w1, float)) < rr:
                return True
        return False

    def _seg_hits_walls(self, a, b, walls, r=ROVER_R_WALL):
        a = np.array(a, float)
        b = np.array(b, float)
        rr = r*r
        for w0, w1 in walls:
            c = np.array(w0, float)
            d = np.array(w1, float)
            if seg_intersect(a, b, c, d):
                return True
            if min(pt_seg_d2(a, c, d), pt_seg_d2(b, c, d),
                   pt_seg_d2(c, a, b), pt_seg_d2(d, a, b)) < rr:
                return True
        return False

    def generate_world(self, n, seed):
        rng = np.random.default_rng(seed)
        bounds = (-WORLD_HALFSPAN, WORLD_HALFSPAN, -WORLD_HALFSPAN, WORLD_HALFSPAN)
        gates = []
        tries = 0

        while len(gates) < n and tries < 50000:
            tries += 1
            c = rng.uniform(-0.7*WORLD_HALFSPAN, 0.7*WORLD_HALFSPAN, size=2)
            th = float(rng.uniform(-np.pi, np.pi))
            g = Gate(center=c, theta=th)

            if any(np.linalg.norm(g.center - gg.center) < MIN_GATE_SEP for gg in gates):
                continue

            other = [w for gg in gates for w in gg.walls]

            if not within(g.dock, bounds, BOUND_MARGIN) or not within(g.approach, bounds, BOUND_MARGIN):
                continue

            if self._circle_hits_walls(g.dock[0], g.dock[1], other): continue
            if self._circle_hits_walls(g.approach[0], g.approach[1], other): continue
            if self._circle_hits_walls(g.dock[0], g.dock[1], g.walls): continue
            if self._circle_hits_walls(g.approach[0], g.approach[1], g.walls): continue

            if self._seg_hits_walls(g.approach, g.dock, g.walls): continue
            if self._seg_hits_walls(g.approach, g.dock, other): continue

            gates.append(g)

        if len(gates) < n:
            raise RuntimeError("Failed to generate a dockable world. Try WORLD_HALFSPAN bigger or MIN_GATE_SEP smaller.")

        walls = [w for g in gates for w in g.walls]
        return gates, walls, bounds

    def build_prm(self, seed):
        rng = np.random.default_rng(seed + 12345)
        xmin, xmax, ymin, ymax = self.bounds
        nodes = []

        while len(nodes) < PRM_SAMPLES:
            x = float(rng.uniform(xmin + BOUND_MARGIN, xmax - BOUND_MARGIN))
            y = float(rng.uniform(ymin + BOUND_MARGIN, ymax - BOUND_MARGIN))
            th = float(rng.uniform(-np.pi, np.pi))
            if self.WI.collides_pose(x, y, self.bounds, BOUND_MARGIN, r=ROVER_R_WALL):
                continue
            nodes.append((x, y, th))

        for g in self.gates:
            nodes.append((float(g.approach[0]), float(g.approach[1]), float(g.heading)))

        return nodes, knn_lists(nodes, PRM_K)

    def reset(self, seed):
        self.auto = False
        self.seed = seed
        self.frame = 0

        self.gates, self.walls, self.bounds = self.generate_world(self.num_docks, seed)
        self.WI.set(self.walls)
        self.nodes, self.nbrs = self.build_prm(seed)

        self.rovers = []
        for i in range(self.num_rovers):
            gi = i % len(self.gates)
            g = self.gates[gi]
            r = Rover(
                i, gi, -1,
                (float(g.dock[0]), float(g.dock[1]), float(g.heading)),
                docked=True, moving=False
            )
            r.trail.append(np.array([g.dock[0], g.dock[1]], float))
            self.rovers.append(r)

        if self.ax is not None:
            self.redraw()

    # ----- dock logic -----
    def dock_free(self, dock, exclude=-1):
        for r in self.rovers:
            if r.id == exclude:
                continue
            if r.current_gate == dock or r.goal_gate == dock:
                return False
        return True

    def unavailable(self, exclude=-1):
        s = set()
        for r in self.rovers:
            if r.id == exclude:
                continue
            s.add(r.current_gate)
            if r.goal_gate >= 0:
                s.add(r.goal_gate)
        return s

    def furthest_dock(self, cur, prev, rid):
        curp = self.gates[cur].dock
        bad = self.unavailable(rid)
        best, bestd = -1, -1.0
        for i, g in enumerate(self.gates):
            if i == cur or i == prev or i in bad:
                continue
            d = float(np.linalg.norm(g.dock - curp))
            if d > bestd:
                bestd, best = d, i
        return best

    # ----- planning -----
    def plan_prm(self, start, goal, obstacle_pts):
        tmp_nodes = self.nodes + [start, goal]
        si = len(tmp_nodes) - 2
        gi = len(tmp_nodes) - 1

        tmp_nbrs = [lst[:] for lst in self.nbrs] + [[], []]
        sN = knn_temp(start, self.nodes, TEMP_K)
        gN = knn_temp(goal, self.nodes, TEMP_K)
        tmp_nbrs[si] = sN
        tmp_nbrs[gi] = gN

        for j in sN:
            if si not in tmp_nbrs[j]:
                tmp_nbrs[j].append(si)
        for j in gN:
            if gi not in tmp_nbrs[j]:
                tmp_nbrs[j].append(gi)

        def blocked_by_rovers_xy(x, y):
            for ox, oy in obstacle_pts:
                if (x-ox)**2 + (y-oy)**2 < ROVER_SEP2:
                    return True
            return False

        def edge_ok(a, b):
            ax, ay, _ = a
            bx, by, _ = b
            if self.WI.collides_pose(ax, ay, self.bounds, BOUND_MARGIN, r=ROVER_R_WALL): return False
            if self.WI.collides_pose(bx, by, self.bounds, BOUND_MARGIN, r=ROVER_R_WALL): return False
            if blocked_by_rovers_xy(ax, ay) or blocked_by_rovers_xy(bx, by): return False
            if self.WI.collides_seg((ax, ay), (bx, by), self.bounds, BOUND_MARGIN, r=ROVER_R_WALL): return False
            # extra rover clearance along edge (cheap sample)
            mx, my = 0.5*(ax+bx), 0.5*(ay+by)
            return not blocked_by_rovers_xy(mx, my)

        idx = astar_lazy(tmp_nodes, tmp_nbrs, si, gi, edge_ok, se2_cost)
        if idx is None:
            return None

        pb = []
        for a_i, b_i in zip(idx[:-1], idx[1:]):
            seg = densify_edge(tmp_nodes[a_i], tmp_nodes[b_i])
            pb += seg[1:] if pb else seg
        return pb

    def plan_rover(self, r: Rover, goal_gate: int):
        if not self.dock_free(goal_gate, r.id):
            return False

        cur = self.gates[r.current_gate]
        goal = self.gates[goal_gate]

        obstacle_pts = [(o.state[0], o.state[1]) for o in self.rovers if o.id != r.id]

        pb = []
        if r.docked:
            und = densify_straight(cur.dock, cur.approach, cur.heading)
            for x, y, _ in und:
                if self.WI.collides_pose(x, y, self.bounds, BOUND_MARGIN, r=ROVER_R_WALL):
                    return False
                for ox, oy in obstacle_pts:
                    if (x-ox)**2 + (y-oy)**2 < ROVER_SEP2:
                        return False
            pb += und
            start = (float(cur.approach[0]), float(cur.approach[1]), float(cur.heading))
        else:
            start = r.state

        goal_pose = (float(goal.approach[0]), float(goal.approach[1]), float(goal.heading))
        free = self.plan_prm(start, goal_pose, obstacle_pts)
        if free is None:
            return False
        pb += free[1:]

        dock = densify_straight(goal.approach, goal.dock, goal.heading)
        for x, y, _ in dock:
            if self.WI.collides_pose(x, y, self.bounds, BOUND_MARGIN, r=ROVER_R_WALL):
                return False
            for ox, oy in obstacle_pts:
                if (x-ox)**2 + (y-oy)**2 < ROVER_SEP2:
                    return False
        pb += dock[1:]

        pb = smooth_mid(pb, r.current_gate, goal_gate, self.gates, self.WI, self.bounds, BOUND_MARGIN, self.smooth)

        r.path = pb
        r.idx = 0
        r.goal_gate = goal_gate
        r.moving = True
        r.docked = False
        r.stuck = 0
        r.reversing = False
        r.blocked_by = -1
        return True

    # ----- traffic -----
    def resolve_traffic(self):
        active = [r for r in self.rovers if r.moving and r.path]
        if not active:
            return 0

        # propose indices
        desired = {}
        for r in active:
            r.blocked_by = -1
            if r.reversing:
                ni = r.idx - 1
                desired[r.id] = ni if (ni >= r.reverse_target and ni >= 0) else r.idx
            else:
                desired[r.id] = min(r.idx + PLAYBACK_SKIP, len(r.path) - 1)

        # commit sequentially by priority to avoid simultaneous collisions
        order = sorted(active, key=lambda rr: (0 if rr.reversing else 1, -rr.stuck, rr.id))

        committed_pos = {}     # rid -> np.array([x,y])
        committed_seg = []     # list of (a,b) segments in this frame (np arrays)
        moved = set()
        waiting = 0

        id_to_rover = {r.id: r for r in self.rovers}

        for r in order:
            ni = desired[r.id]
            if ni == r.idx:
                waiting += 1
                r.stuck += 1
                continue

            p0 = np.array([r.state[0], r.state[1]], float)
            p1 = np.array([r.path[ni][0], r.path[ni][1]], float)

            ok = True

            # vs committed new positions
            for pos in committed_pos.values():
                if np.sum((p1 - pos)**2) < ROVER_SEP2:
                    ok = False
                    break

            # vs committed motion segments (prevents clipping / crossings)
            if ok:
                for a, b in committed_seg:
                    if seg_seg_d2(p0, p1, a, b) < ROVER_SEP2:
                        ok = False
                        break

            # conservative vs not-yet-committed rovers' CURRENT positions
            if ok:
                for o in self.rovers:
                    if o.id == r.id or o.id in committed_pos:
                        continue
                    ox, oy, _ = o.state
                    d2 = (p1[0]-ox)**2 + (p1[1]-oy)**2
                    if d2 < ROVER_SEP2:
                        ok = False
                        r.blocked_by = o.id
                        break

            if ok:
                r.idx = ni
                r.state = r.path[r.idx]
                r.trail.append(np.array([r.state[0], r.state[1]], float))
                committed_pos[r.id] = p1
                committed_seg.append((p0, p1))
                moved.add(r.id)
                if not r.reversing:
                    r.stuck = max(0, r.stuck - 1)
            else:
                waiting += 1
                r.stuck += 1

        # cycle-breaker: if blocked graph has a cycle, force the highest-id rover in the cycle to reverse
        # (only if it has room to reverse)
        blocked_map = {r.id: r.blocked_by for r in active if (r.id not in moved and r.blocked_by != -1)}
        seen = set()
        for start in list(blocked_map.keys()):
            if start in seen:
                continue
            chain = []
            cur = start
            while cur in blocked_map and cur not in seen:
                seen.add(cur)
                chain.append(cur)
                cur = blocked_map[cur]
                if cur in chain:
                    cyc = chain[chain.index(cur):]
                    rid = max(cyc)
                    rr = id_to_rover.get(rid)
                    if rr and (not rr.reversing) and rr.idx > 10:
                        rr.reversing = True
                        rr.reverse_target = max(0, rr.idx - 60)
                        rr.stuck = 0
                    break

        # if reversing rover reached target or can’t reverse, stop reversing cleanly next frame
        for r in active:
            if r.reversing and (r.idx <= r.reverse_target or r.idx <= 0):
                r.reversing = False

        return waiting

    # ===================== VISUALS =====================
    def setup_vis(self):
        self.fig, self.ax = plt.subplots(figsize=(10.5, 9.5))
        plt.subplots_adjust(left=0.18, bottom=0.14)
        ax = self.ax
        ax.set_aspect("equal", "box")
        ax.set_facecolor((0.06, 0.07, 0.09))
        self.fig.patch.set_facecolor((0.06, 0.07, 0.09))
        for sp in ax.spines.values():
            sp.set_color((0.35, 0.35, 0.4))
        ax.tick_params(colors=(0.75, 0.75, 0.8))
        ax.grid(True, alpha=0.12)

        sd = self.fig.add_axes([0.02, 0.75, 0.10, 0.03])
        sd.set_facecolor((0.15, 0.15, 0.18))
        self.sld_d = Slider(sd, 'Docks', 2, 10, valinit=self.num_docks, valstep=1, color=(0.3, 0.6, 0.8))

        sr = self.fig.add_axes([0.02, 0.65, 0.10, 0.03])
        sr.set_facecolor((0.15, 0.15, 0.18))
        self.sld_r = Slider(sr, 'Rovers', 1, max(1, self.num_docks//2), valinit=self.num_rovers, valstep=1, color=(0.6, 0.8, 0.3))

        ap = self.fig.add_axes([0.02, 0.50, 0.10, 0.06])
        self.btn_play = Button(ap, 'Play', color=(0.15, 0.15, 0.18), hovercolor=(0.25, 0.4, 0.25))

        b1 = self.fig.add_axes([0.20, 0.04, 0.22, 0.06])
        self.btn_rerand = Button(b1, "Rerandomize gates")
        self.btn_rerand.on_clicked(lambda _ : self.reset(self.seed + 1))

        b2 = self.fig.add_axes([0.46, 0.04, 0.18, 0.06])
        self.btn_sm = Button(b2, "Smooth: ON" if self.smooth else "Smooth: OFF")

        self.sld_d.on_changed(self.on_docks)
        self.sld_r.on_changed(self.on_rovers)
        self.btn_play.on_clicked(self.on_play)
        self.btn_sm.on_clicked(self.on_smooth)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def clear_art(self):
        for a in self.gate_art:
            try: a.remove()
            except: pass
        self.gate_art = []
        for p, l in self.rover_art:
            try: p.remove()
            except: pass
            try: l.remove()
            except: pass
        self.rover_art = []

    def redraw(self):
        ax = self.ax
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        self.clear_art()

        for i, g in enumerate(self.gates):
            col = self.gate_colors[i % len(self.gate_colors)]
            for w0, w1 in g.walls:
                ln, = ax.plot([w0[0], w1[0]], [w0[1], w1[1]], lw=3.2, alpha=0.9, color=col, solid_capstyle="round")
                self.gate_art.append(ln)
            self.gate_art.append(ax.scatter([g.open_p1[0], g.open_p2[0]], [g.open_p1[1], g.open_p2[1]], s=24, color=col, alpha=0.95))
            self.gate_art.append(ax.scatter([g.approach[0]], [g.approach[1]], s=70, marker="x", color=(1,1,1), alpha=0.75))
            self.gate_art.append(ax.scatter([g.dock[0]], [g.dock[1]], s=55, marker="o", edgecolors=col, facecolors="none", linewidths=2))
            self.gate_art.append(ax.text(g.center[0], g.center[1], f"G{i}", color=(0.95,0.95,0.98), ha="center", va="center", fontsize=11, alpha=0.95))

        self.rover_art = []
        for r in self.rovers:
            col = self.rover_colors[r.id % len(self.rover_colors)]
            poly = Polygon(square_corners((r.state[0], r.state[1]), r.state[2], S),
                           closed=True, facecolor=col, edgecolor=(0.1,0.1,0.1), lw=2.2, alpha=0.95)
            ax.add_patch(poly)
            line, = ax.plot([], [], lw=2.0, alpha=0.55, color=col)
            r.poly, r.line = poly, line
            r.trail.clear()
            r.trail.append(np.array([r.state[0], r.state[1]], float))
            self.rover_art.append((poly, line))

        self.auto = False
        self.btn_play.label.set_text("Play")
        ax.set_title(f"Docks: {self.num_docks} | Rovers: {self.num_rovers} | Click gate or press Play",
                     color=(0.9,0.9,0.95), pad=12)
        self.fig.canvas.draw_idle()

    # ----- events -----
    def on_smooth(self, _):
        self.smooth = not self.smooth
        self.btn_sm.label.set_text("Smooth: ON" if self.smooth else "Smooth: OFF")
        self.fig.canvas.draw_idle()

    def on_docks(self, val):
        self.num_docks = int(val)
        maxr = max(1, self.num_docks // 2)
        self.sld_r.valmax = maxr
        self.sld_r.ax.set_xlim(1, maxr)
        if self.num_rovers > maxr:
            self.num_rovers = maxr
            self.sld_r.set_val(maxr)
        self.reset(self.seed)

    def on_rovers(self, val):
        self.num_rovers = int(val)
        self.reset(self.seed)

    def on_play(self, _):
        self.auto = not self.auto
        self.btn_play.label.set_text("Stop" if self.auto else "Play")
        if self.auto:
            for r in self.rovers:
                if r.docked and not r.moving and r.plan_cd == 0:
                    nxt = self.furthest_dock(r.current_gate, r.previous_gate, r.id)
                    if nxt >= 0:
                        if not self.plan_rover(r, nxt):
                            r.plan_cd = 30
            self.ax.set_title(f"Running | Docks: {self.num_docks} | Rovers: {self.num_rovers}", color=(0.6,0.95,0.6))
        else:
            self.ax.set_title(f"Paused | Docks: {self.num_docks} | Rovers: {self.num_rovers}", color=(0.9,0.9,0.95))
        self.fig.canvas.draw_idle()

    def nearest_gate(self, xy):
        d = [float(np.linalg.norm(g.dock - xy)) for g in self.gates]
        i = int(np.argmin(d))
        return i, d[i]

    def on_click(self, ev):
        if ev.inaxes != self.ax:
            return
        if (not self.auto) and any(r.moving for r in self.rovers):
            return
        click = np.array([ev.xdata, ev.ydata], float)
        gi, dist = self.nearest_gate(click)
        if dist > 1.6 * S:
            return
        r = next((rr for rr in self.rovers if rr.docked and not rr.moving and rr.current_gate != gi), None)
        if r is None:
            return

        self.ax.set_title(f"Planning: Rover {r.id} Gate {r.current_gate} → Gate {gi} ...", color=(0.9,0.9,0.95))
        self.fig.canvas.draw_idle()

        ok = self.plan_rover(r, gi)

        self.ax.set_title(
            f"Executing: Rover {r.id} Gate {r.current_gate} → Gate {gi}" if ok else "No path found. Try again.",
            color=(0.9,0.9,0.95) if ok else (0.95,0.6,0.6)
        )
        self.fig.canvas.draw_idle()

    # ----- animation -----
    def animate(self, _):
        self.frame += 1
        for r in self.rovers:
            if r.plan_cd > 0:
                r.plan_cd -= 1

        waiting = self.resolve_traffic()

        for r in self.rovers:
            if not (r.moving and r.path):
                continue

            # arrival
            if r.idx >= len(r.path) - 1:
                if self.dock_free(r.goal_gate, r.id):
                    g = self.gates[r.goal_gate]
                    r.state = (float(g.dock[0]), float(g.dock[1]), float(g.heading))
                    r.trail.append(np.array([r.state[0], r.state[1]], float))
                    r.previous_gate, r.current_gate = r.current_gate, r.goal_gate
                    r.goal_gate = -1
                    r.path = None
                    r.moving = False
                    r.docked = True
                    r.stuck = 0
                    r.reversing = False

                    if self.auto and r.plan_cd == 0:
                        nxt = self.furthest_dock(r.current_gate, r.previous_gate, r.id)
                        if nxt >= 0 and (not self.plan_rover(r, nxt)):
                            r.plan_cd = 30
                else:
                    new = self.furthest_dock(r.current_gate, r.goal_gate, r.id)
                    r.goal_gate = -1
                    r.path = None
                    r.moving = False
                    r.reversing = False
                    if new >= 0 and r.plan_cd == 0:
                        if not self.plan_rover(r, new):
                            r.plan_cd = 30

            # stuck fallback: replan occasionally (cooldown prevents spikes)
            if r.stuck > 140 and r.goal_gate >= 0 and r.plan_cd == 0:
                r.stuck = 0
                if not self.plan_rover(r, r.goal_gate):
                    r.plan_cd = 45

        # auto: retry idle rovers occasionally
        if self.auto:
            for r in self.rovers:
                if r.docked and not r.moving and r.goal_gate == -1 and r.plan_cd == 0:
                    nxt = self.furthest_dock(r.current_gate, r.previous_gate, r.id)
                    if nxt >= 0 and (not self.plan_rover(r, nxt)):
                        r.plan_cd = 30

        # draw
        for r in self.rovers:
            if r.poly:
                r.poly.set_xy(square_corners((r.state[0], r.state[1]), r.state[2], S))
            if r.line and len(r.trail) > 0:
                t = np.array(r.trail, float)
                r.line.set_data(t[:,0], t[:,1])

        # throttle title updates (big perf win)
        if self.frame % TITLE_EVERY == 0:
            moving = sum(1 for r in self.rovers if r.moving)
            if moving or waiting:
                self.ax.set_title(("Running" if self.auto else "Manual") + f" | Moving: {moving} | Waiting: {waiting}",
                                  color=(0.6,0.95,0.6))
            elif not self.auto:
                self.ax.set_title(f"Docks: {self.num_docks} | Rovers: {self.num_rovers} | Click gate or press Play",
                                  color=(0.9,0.9,0.95))
        return []

    def run(self):
        self.setup_vis()
        self.reset(BASE_SEED)
        self.ani = FuncAnimation(self.fig, self.animate, interval=FRAME_MS, blit=False)
        plt.show()


if __name__ == "__main__":
    Sim().run()
