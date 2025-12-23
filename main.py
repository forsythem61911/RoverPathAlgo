import math, random, heapq, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, FancyBboxPatch, Rectangle
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
GATE_CLEARANCE_PENALTY = 15.0
GATE_CLEARANCE_R = 1.6 * S

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
TRAIL_MAX = 500
TRAIL_DRAW_EVERY = 3
FAST_VIS = True
TITLE_EVERY = 999999  # avoid frequent title redraw (expensive)

# background planner
PLANNER_WORKERS = 1

# rover collision geometry
ROVER_R_WALL = (S / 2.0) * 0.95
ROVER_R_ROVER = (math.sqrt(2) / 2.0) * S * 0.99
ROVER_SEP = 2.0 * ROVER_R_ROVER * 1.03
ROVER_SEP2 = ROVER_SEP * ROVER_SEP

# ===================== COLLISION POLICY (REFINED) =====================
WAIT_BEFORE_YIELD = 10          # frames to just wait if blocked
YIELD_COOLDOWN = 45             # don't yield repeatedly
BACKOFF_INDICES = 80            # reverse this many path indices (short distance; path is dense)
BACKOFF_MIN_IDX = 10            # only reverse if idx >= this
REPLAN_TRIGGER = 70             # stuck frames before requesting background replan
REPLAN_COOLDOWN = 35            # frames between replans

# if reversing is impossible (near path start), inject a tiny backoff segment
BACKOFF_STEP = 0.04             # world units per injected point
BACKOFF_PTS = 12                # number of injected points
FORCE_STEP_FRAMES = 22          # force step-by-step (no skipping) for a bit after backoff insert

# ===================== HUMANS + CAMERA FOV =====================
HUMAN_R = 0.42 * S
HUMAN_FOV_DEG = 45.0              # full angle of view
HUMAN_FOV_HALF = math.radians(HUMAN_FOV_DEG / 2.0)
HUMAN_RANGE = 3.2 * S             # detection range from camera center
HUMAN_RESUME_DELAY_S = 3.0        # resume this long after humans are out of view
HUMAN_CLICK_TOL = 0.75 * HUMAN_R  # click distance to select/delete human

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

class ToggleSwitch:
    def __init__(self, ax, label, initial_state=False, on_callback=None):
        self.ax = ax
        self.label = label
        self.state = initial_state
        self.on_callback = on_callback
        
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.axis('off')
        
        # Track (Pill shape)
        self.track = FancyBboxPatch((0.0, 0.25), 1.0, 0.5, boxstyle="round,pad=0,rounding_size=0.25", 
                                    facecolor='#39393D', edgecolor='#555555', lw=0.5)
        self.ax.add_patch(self.track)
        
        # Shadow (Subtle offset)
        self.shadow = Circle((0.25 if not self.state else 0.75, 0.48), radius=0.22, 
                             facecolor='#111111', alpha=0.4, zorder=2)
        self.ax.add_patch(self.shadow)
        
        # Thumb
        self.thumb = Circle((0.25 if not self.state else 0.75, 0.5), radius=0.22, 
                            facecolor='#FFFFFF', edgecolor='#DDDDDD', lw=0.5, zorder=3)
        self.ax.add_patch(self.thumb)
        
        # Label text (Aesthetic update)
        self.text = self.ax.text(-0.15, 0.5, label, va='center', ha='right', color='#E0E0E0', 
                                fontsize=10, weight='medium')
        
        self._update_ui()
        self.ax.figure.canvas.mpl_connect('button_press_event', self._on_click)

    def _update_ui(self):
        # iOS-style colors: Green for ON, Dark Gray for OFF
        if self.state:
            self.track.set_facecolor('#34C759')
            pos = 0.75
        else:
            self.track.set_facecolor('#39393D')
            pos = 0.25
        
        self.thumb.center = (pos, 0.5)
        self.shadow.center = (pos, 0.48)
        self.ax.figure.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes == self.ax:
            self.state = not self.state
            self._update_ui()
            if self.on_callback:
                self.on_callback(self.state)

    def set_state(self, state):
        self.state = state
        self._update_ui()


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

    plan_cd: int = 0

    # background planning
    plan_future: Optional[object] = None
    plan_token: int = 0
    plan_replaces_existing: bool = False
    replan_cd: int = 0

    # refined collision behavior
    yield_cd: int = 0         # cooldown to avoid thrashing
    force_step: int = 0       # force step-by-step movement briefly (no skipping)

    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_MAX))

    poly: Optional[Polygon] = None
    line: Optional[object] = None

    # FOV visualization (front/back)
    fov_front: Optional[Wedge] = None
    fov_back: Optional[Wedge] = None


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

def knn_temp_xy(pose, xy, k):
    x, y, _ = pose
    d2 = (xy[:,0]-x)**2 + (xy[:,1]-y)**2
    idx = np.argpartition(d2, k)[:k]
    idx = idx[np.argsort(d2[idx])]
    return idx.tolist()

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

def smooth_mid_points(playback, cur_app, goal_app, th0, th1, WI, bounds, margin, enabled):
    if (not enabled) or len(playback) < 12:
        return playback

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


# ===================== BACKGROUND PLANNER (PROCESS) =====================
_W_NODES = None
_W_NBRS = None
_W_NODE_XY = None
_W_WI = None
_W_BOUNDS = None
_W_GATES = None

def _worker_init(nodes, nbrs, walls, bounds, gates):
    global _W_NODES, _W_NBRS, _W_NODE_XY, _W_WI, _W_BOUNDS, _W_GATES
    _W_NODES = nodes
    _W_NBRS = nbrs
    _W_NODE_XY = np.array([(n[0], n[1]) for n in nodes], float)
    _W_WI = WallIndex(walls)
    _W_BOUNDS = bounds
    _W_GATES = gates

def _plan_worker(cur_state, docked,
                 cur_dock, cur_app, cur_heading,
                 goal_app, goal_dock, goal_heading,
                 obstacle_pts, smooth_enabled,
                 cur_gate_id, goal_gate_id):
    def blocked_by_rovers_xy(x, y):
        for ox, oy in obstacle_pts:
            if (x-ox)**2 + (y-oy)**2 < ROVER_SEP2:
                return True
        return False

    pb = []
    if docked:
        und = densify_straight(cur_dock, cur_app, cur_heading)
        for x, y, _ in und:
            if _W_WI.collides_pose(x, y, _W_BOUNDS, BOUND_MARGIN, r=ROVER_R_WALL):
                return None
            if blocked_by_rovers_xy(x, y):
                return None
        pb += und
        start = (float(cur_app[0]), float(cur_app[1]), float(cur_heading))
    else:
        start = cur_state

    goal_pose = (float(goal_app[0]), float(goal_app[1]), float(goal_heading))

    tmp_nodes = _W_NODES + [start, goal_pose]
    si = len(tmp_nodes) - 2
    gi = len(tmp_nodes) - 1

    tmp_nbrs = [lst[:] for lst in _W_NBRS] + [[], []]
    sN = knn_temp_xy(start, _W_NODE_XY, TEMP_K)
    gN = knn_temp_xy(goal_pose, _W_NODE_XY, TEMP_K)
    tmp_nbrs[si] = sN
    tmp_nbrs[gi] = gN
    for j in sN:
        tmp_nbrs[j].append(si)
    for j in gN:
        tmp_nbrs[j].append(gi)

    def edge_ok(a, b):
        ax, ay, _ = a
        bx, by, _ = b
        if _W_WI.collides_pose(ax, ay, _W_BOUNDS, BOUND_MARGIN, r=ROVER_R_WALL): return False
        if _W_WI.collides_pose(bx, by, _W_BOUNDS, BOUND_MARGIN, r=ROVER_R_WALL): return False
        if blocked_by_rovers_xy(ax, ay) or blocked_by_rovers_xy(bx, by): return False
        if _W_WI.collides_seg((ax, ay), (bx, by), _W_BOUNDS, BOUND_MARGIN, r=ROVER_R_WALL): return False
        mx, my = 0.5*(ax+bx), 0.5*(ay+by)
        return not blocked_by_rovers_xy(mx, my)

    def edge_cost(a, b):
        base = se2_cost(a, b)
        
        # Penalize if edge or midpoint is near a gate that isn't start/goal
        penalty = 0.0
        pts_to_check = [a, b, (0.5*(a[0]+b[0]), 0.5*(a[1]+b[1]), a[2])]
        for px, py, _ in pts_to_check:
            for gi, gate in enumerate(_W_GATES):
                if gi == cur_gate_id or gi == goal_gate_id:
                    continue
                # distance to gate center (or approach point might be better, but user said gate opening)
                # Let's use gate center as a proxy for the opening/mouth
                dist2 = (px-gate.center[0])**2 + (py-gate.center[1])**2
                if dist2 < GATE_CLEARANCE_R**2:
                    penalty += GATE_CLEARANCE_PENALTY
                    break # only penalize once per point
        return base + penalty

    idx = astar_lazy(tmp_nodes, tmp_nbrs, si, gi, edge_ok, edge_cost)
    if idx is None:
        return None

    pb_free = []
    for a_i, b_i in zip(idx[:-1], idx[1:]):
        seg = densify_edge(tmp_nodes[a_i], tmp_nodes[b_i])
        pb_free += seg[1:] if pb_free else seg
    pb += pb_free[1:] if pb else pb_free

    dock_seg = densify_straight(goal_app, goal_dock, goal_heading)
    for x, y, _ in dock_seg:
        if _W_WI.collides_pose(x, y, _W_BOUNDS, BOUND_MARGIN, r=ROVER_R_WALL):
            return None
        if blocked_by_rovers_xy(x, y):
            return None
    pb += dock_seg[1:]

    pb = smooth_mid_points(pb, cur_app, goal_app, cur_heading, goal_heading,
                           _W_WI, _W_BOUNDS, BOUND_MARGIN, smooth_enabled)
    return pb


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
        self.node_xy = None

        self.rovers: List[Rover] = []

        # humans
        self.humans: List[Tuple[float, float]] = []
        self.human_art: List[Tuple[Circle, object]] = []  # (circle, text)

        # pause / safety stop
        self.paused = False
        self.pause_clear_deadline: Optional[float] = None
        self.pause_text = None

        # FOV and Trails overlay
        self.show_fov = False
        self.show_trails = True
        self.btn_fov = None
        self.btn_trails = None

        self.fig = None
        self.ax = None
        self.gate_art = []
        self.rover_art = []
        self.sld_d = None
        self.sld_r = None
        self.btn_play = None
        self.sw_sm = None
        self.sw_fov = None
        self.sw_tr = None
        self.btn_rerand = None

        self.frame = 0
        self.pool: Optional[ProcessPoolExecutor] = None

        # small HUD text
        self.hud = None

    # ---------- pool ----------
    def _shutdown_pool(self):
        if self.pool is not None:
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.pool = None

    def _ensure_pool(self):
        self._shutdown_pool()
        self.pool = ProcessPoolExecutor(
            max_workers=PLANNER_WORKERS,
            initializer=_worker_init,
            initargs=(self.nodes, self.nbrs, self.walls, self.bounds, self.gates),
        )

    # ---------- world gen ----------
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

        nbrs = knn_lists(nodes, PRM_K)
        return nodes, nbrs

    def reset(self, seed):
        self.auto = False
        self.seed = seed
        self.frame = 0

        # keep humans (don’t wipe) so you can rerand gates and still test stops
        self.paused = False
        self.pause_clear_deadline = None

        self.gates, self.walls, self.bounds = self.generate_world(self.num_docks, seed)
        self.WI.set(self.walls)
        self.nodes, self.nbrs = self.build_prm(seed)
        self.node_xy = np.array([(n[0], n[1]) for n in self.nodes], float)

        self.rovers = []
        for i in range(self.num_rovers):
            gi = i % len(self.gates)
            g = self.gates[gi]
            r = Rover(
                i, gi, -1,
                (float(g.dock[0]), float(g.dock[1]), float(g.heading)),
                docked=True, moving=False
            )
            r.trail.append((float(g.dock[0]), float(g.dock[1])))
            self.rovers.append(r)

        self._ensure_pool()

        if self.ax is not None:
            self.redraw()

    # ---------- humans ----------
    def _human_nearest(self, x, y):
        if not self.humans:
            return -1, 1e30
        best_i, best_d = -1, 1e30
        for i, (hx, hy) in enumerate(self.humans):
            d = math.hypot(hx - x, hy - y)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i, best_d

    def _humans_clear_art(self):
        for c, t in self.human_art:
            try: c.remove()
            except: pass
            try: t.remove()
            except: pass
        self.human_art = []

    def _humans_redraw_art(self):
        if self.ax is None:
            return
        self._humans_clear_art()
        ax = self.ax
        for (hx, hy) in self.humans:
            circ = Circle((hx, hy), radius=HUMAN_R, facecolor=(0.9, 0.1, 0.1, 1.0),
                          edgecolor=(1, 1, 1, 0.9), lw=1.5)
            ax.add_patch(circ)
            txt = ax.text(hx, hy, "H", color=(1, 1, 1, 1.0), ha="center", va="center",
                          fontsize=12, weight="bold")
            self.human_art.append((circ, txt))
        self.fig.canvas.draw_idle()

    def toggle_human_at(self, x, y):
        if x is None or y is None:
            return
        if not within((x, y), self.bounds, 0.0):
            return
        i, d = self._human_nearest(x, y)
        if i >= 0 and d <= HUMAN_CLICK_TOL:
            self.humans.pop(i)
        else:
            self.humans.append((float(x), float(y)))
        self._humans_redraw_art()

    def delete_human_at(self, x, y):
        if x is None or y is None:
            return
        i, d = self._human_nearest(x, y)
        if i >= 0 and d <= HUMAN_CLICK_TOL:
            self.humans.pop(i)
            self._humans_redraw_art()

    # ---------- dock logic ----------
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
                bestd, best = d
        return best

    # ---------- safety ----------
    def pose_safe(self, x, y, exclude_id=-1):
        if self.WI.collides_pose(x, y, self.bounds, BOUND_MARGIN, r=ROVER_R_WALL):
            return False
        for o in self.rovers:
            if o.id == exclude_id:
                continue
            ox, oy, _ = o.state
            if (x-ox)**2 + (y-oy)**2 < ROVER_SEP2:
                return False
        return True

    # ---------- background planning ----------
    def start_plan_job(self, r: Rover, goal_gate: int, keep_moving: bool):
        if r.plan_future is not None:
            return False
        if not self.dock_free(goal_gate, r.id):
            return False

        cur = self.gates[r.current_gate]
        goal = self.gates[goal_gate]
        obstacle_pts = [(o.state[0], o.state[1]) for o in self.rovers if o.id != r.id]

        r.goal_gate = goal_gate
        r.plan_replaces_existing = bool(r.moving and r.path)
        r.plan_token += 1
        tok = r.plan_token

        if not keep_moving:
            r.moving = False

        r.plan_future = self.pool.submit(
            _plan_worker,
            r.state, r.docked,
            (float(cur.dock[0]), float(cur.dock[1])),
            (float(cur.approach[0]), float(cur.approach[1])),
            float(cur.heading),
            (float(goal.approach[0]), float(goal.approach[1])),
            (float(goal.dock[0]), float(goal.dock[1])),
            float(goal.heading),
            obstacle_pts,
            bool(self.smooth),
            r.current_gate,
            goal_gate,
        )
        r.plan_future._tok = tok  # type: ignore[attr-defined]
        return True

    def _nearest_idx(self, path, x, y, max_scan=160):
        if not path:
            return 0
        n = min(len(path), max_scan)
        best_i = 0
        best = 1e30
        for i in range(n):
            px, py, _ = path[i]
            d = (px-x)*(px-x) + (py-y)*(py-y)
            if d < best:
                best = d
                best_i = i
        return best_i

    def poll_plan_jobs(self):
        for r in self.rovers:
            f = r.plan_future
            if f is None or (not f.done()):
                continue
            r.plan_future = None
            tok = getattr(f, "_tok", None)
            if tok is not None and tok != r.plan_token:
                continue

            try:
                pb = f.result()
            except Exception:
                pb = None

            if pb is None or len(pb) < 2:
                if not r.plan_replaces_existing and r.docked:
                    r.goal_gate = -1
                    r.plan_cd = 40
                r.replan_cd = REPLAN_COOLDOWN
                continue

            x, y, _ = r.state
            r.path = pb
            r.idx = self._nearest_idx(pb, x, y)
            r.moving = True
            r.docked = False
            r.stuck = 0
            r.reversing = False
            r.blocked_by = -1
            r.replan_cd = REPLAN_COOLDOWN

            r.force_step = 0
            r.yield_cd = max(r.yield_cd, 12)

    # ---------- smooth backoff segment (NO TELEPORT) ----------
    def inject_backoff_segment(self, r: Rover):
        x, y, th = r.state
        dx, dy = -math.cos(th), -math.sin(th)

        pts = []
        for i in range(1, BACKOFF_PTS + 1):
            nx = x + i * BACKOFF_STEP * dx
            ny = y + i * BACKOFF_STEP * dy
            if not self.pose_safe(nx, ny, exclude_id=r.id):
                break
            pts.append((float(nx), float(ny), float(th)))

        if not pts:
            return False

        if r.path is None:
            r.path = [r.state] + pts
            r.idx = 0
            r.moving = True
        else:
            ins = min(r.idx + 1, len(r.path))
            r.path[ins:ins] = pts

        r.force_step = max(r.force_step, FORCE_STEP_FRAMES)
        r.yield_cd = max(r.yield_cd, YIELD_COOLDOWN)
        r.stuck = 0
        return True

    # ---------- traffic ----------
    def resolve_traffic(self):
        active = [r for r in self.rovers if r.moving and r.path]
        if not active:
            return 0

        desired = {}
        for r in active:
            r.blocked_by = -1
            if r.reversing:
                ni = r.idx - 1
                desired[r.id] = ni if (ni >= r.reverse_target and ni >= 0) else r.idx
            else:
                if r.force_step > 0:
                    desired[r.id] = min(r.idx + 1, len(r.path) - 1)
                else:
                    desired[r.id] = min(r.idx + PLAYBACK_SKIP, len(r.path) - 1)

        def remain(rr: Rover):
            return (len(rr.path) - 1 - rr.idx) if rr.path else 10**9

        order = sorted(active, key=lambda rr: (0 if rr.reversing else 1, remain(rr), rr.id))

        committed_pos = {}
        committed_seg = []
        moved = set()
        waiting = 0
        id_to_rover = {r.id: r for r in self.rovers}

        def move_ok(r: Rover, p0, p1):
            for rid, pos in committed_pos.items():
                if np.sum((p1 - pos)**2) < ROVER_SEP2:
                    r.blocked_by = rid
                    return False

            for a, b in committed_seg:
                if seg_seg_d2(p0, p1, a, b) < ROVER_SEP2:
                    return False

            for o in self.rovers:
                if o.id == r.id or o.id in committed_pos:
                    continue
                ox, oy, _ = o.state
                d2 = (p1[0]-ox)**2 + (p1[1]-oy)**2
                if d2 < ROVER_SEP2:
                    r.blocked_by = o.id
                    return False
            return True

        for r in order:
            ni = desired[r.id]
            if ni == r.idx:
                waiting += 1
                r.stuck += 1
                continue

            p0 = np.array([r.state[0], r.state[1]], float)
            candidates = [ni]
            if (not r.reversing) and (ni != r.idx + 1):
                candidates.append(min(r.idx + 1, len(r.path) - 1))

            committed = False
            for cand in candidates:
                if cand == r.idx:
                    continue
                p1 = np.array([r.path[cand][0], r.path[cand][1]], float)
                if move_ok(r, p0, p1):
                    r.idx = cand
                    r.state = r.path[r.idx]
                    r.trail.append((float(r.state[0]), float(r.state[1])))
                    committed_pos[r.id] = p1
                    committed_seg.append((p0, p1))
                    moved.add(r.id)
                    committed = True
                    if r.force_step > 0:
                        r.force_step -= 1
                    if not r.reversing:
                        r.stuck = max(0, r.stuck - 1)
                    break

            if committed:
                continue

            waiting += 1
            r.stuck += 1

        for r in active:
            if r.id in moved:
                continue
            if r.reversing:
                continue
            if r.yield_cd > 0:
                continue
            if r.blocked_by == -1:
                continue
            if r.stuck < WAIT_BEFORE_YIELD:
                continue

            if r.id > r.blocked_by:
                if r.idx >= BACKOFF_MIN_IDX:
                    r.reversing = True
                    r.reverse_target = max(0, r.idx - BACKOFF_INDICES)
                    r.stuck = 0
                    r.yield_cd = YIELD_COOLDOWN
                    r.force_step = max(r.force_step, FORCE_STEP_FRAMES)
                else:
                    if self.inject_backoff_segment(r):
                        r.force_step = max(r.force_step, FORCE_STEP_FRAMES)

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
                    if rr and (not rr.reversing) and rr.idx > 2 and rr.yield_cd == 0:
                        rr.reversing = True
                        rr.reverse_target = max(0, rr.idx - BACKOFF_INDICES)
                        rr.stuck = 0
                        rr.yield_cd = YIELD_COOLDOWN
                        rr.force_step = max(rr.force_step, FORCE_STEP_FRAMES)
                    break

        for r in active:
            if r.reversing and (r.idx <= r.reverse_target or r.idx <= 0):
                r.reversing = False

        return waiting

    # ===================== HUMAN DETECTION =====================
    def _cam_positions(self, r: Rover):
        x, y, th = r.state
        fx = x + (S/2.0) * math.cos(th)
        fy = y + (S/2.0) * math.sin(th)
        bx = x - (S/2.0) * math.cos(th)
        by = y - (S/2.0) * math.sin(th)
        return (fx, fy, th), (bx, by, wrap(th + math.pi))

    def _human_in_cam_fov(self, cam, hx, hy):
        cx, cy, axis = cam
        dx = hx - cx
        dy = hy - cy
        d2 = dx*dx + dy*dy
        if d2 > HUMAN_RANGE * HUMAN_RANGE:
            return False
        ang = math.atan2(dy, dx)
        if abs(wrap(ang - axis)) <= HUMAN_FOV_HALF:
            return True
        return False

    def any_human_detected(self):
        if not self.humans:
            return False
        for r in self.rovers:
            cam_f, cam_b = self._cam_positions(r)
            for (hx, hy) in self.humans:
                # treat human as a disk: if center in view OR edge in view (cheap approx: inflate range by HUMAN_R)
                if self._human_in_cam_fov(cam_f, hx, hy) or self._human_in_cam_fov(cam_b, hx, hy):
                    return True
                # edge-in-range fallback (helps when human is just outside range by radius)
                cx, cy, axis = cam_f
                dx, dy = hx - cx, hy - cy
                if dx*dx + dy*dy <= (HUMAN_RANGE + HUMAN_R) ** 2:
                    ang = math.atan2(dy, dx)
                    if abs(wrap(ang - axis)) <= HUMAN_FOV_HALF:
                        return True
                cx, cy, axis = cam_b
                dx, dy = hx - cx, hy - cy
                if dx*dx + dy*dy <= (HUMAN_RANGE + HUMAN_R) ** 2:
                    ang = math.atan2(dy, dx)
                    if abs(wrap(ang - axis)) <= HUMAN_FOV_HALF:
                        return True
        return False

    def update_pause_state(self, now_s: float):
        detected = self.any_human_detected()
        if detected:
            self.paused = True
            self.pause_clear_deadline = None
            return

        if self.paused:
            if self.pause_clear_deadline is None:
                self.pause_clear_deadline = now_s + HUMAN_RESUME_DELAY_S
            if now_s >= self.pause_clear_deadline:
                self.paused = False
                self.pause_clear_deadline = None

    # ===================== VISUALS =====================
    def setup_vis(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 9.5))
        plt.subplots_adjust(left=0.24, bottom=0.10, right=0.96, top=0.94)
        ax = self.ax
        ax.set_aspect("equal", "box")
        ax.set_facecolor((0.06, 0.07, 0.09))
        self.fig.patch.set_facecolor((0.06, 0.07, 0.09))
        for sp in ax.spines.values():
            sp.set_color((0.35, 0.35, 0.4))
        ax.tick_params(colors=(0.75, 0.75, 0.8))
        ax.grid(False if FAST_VIS else True, alpha=0.12)

        # Control Panel Layout Parameters
        px = 0.015; pw = 0.21; py = 0.05; ph = 0.90
        
        # Bordered Backpanel (Glassmorphism inspired)
        panel_bg = Rectangle((px, py), pw, ph, facecolor='#1A1C22', edgecolor='#4A4E5A', 
                             lw=1.2, transform=self.fig.transFigure, zorder=-1, alpha=0.98)
        self.fig.patches.append(panel_bg)
        
        # Panel Title
        self.fig.text(px + pw/2, py + ph - 0.045, "CONTROL PANEL", color='#F0F0F0', 
                      va='center', ha='center', fontsize=13, weight='bold')

        # Items position
        item_top = py + ph - 0.12
        spacing = 0.065
        item_w = pw - 0.04
        item_x = px + 0.02

        # SLIDERS
        sd_ax = self.fig.add_axes([item_x + 0.07, item_top, item_w - 0.07, 0.025])
        sd_ax.set_facecolor((0.08, 0.08, 0.1))
        self.sld_d = Slider(sd_ax, 'Docks', 2, 10, valinit=self.num_docks, valstep=1, 
                            color='#007AFF', track_color='#2C2C2E')
        self.sld_d.label.set_color('#EBEBF5')
        self.sld_d.label.set_fontsize(9)

        sr_ax = self.fig.add_axes([item_x + 0.07, item_top - spacing, item_w - 0.07, 0.025])
        sr_ax.set_facecolor((0.08, 0.08, 0.1))
        self.sld_r = Slider(sr_ax, 'Rovers', 1, max(1, self.num_docks//2), valinit=self.num_rovers, valstep=1, 
                            color='#32D74B', track_color='#2C2C2E')
        self.sld_r.label.set_color('#EBEBF5')
        self.sld_r.label.set_fontsize(9)

        # BUTTONS
        btn_y = item_top - 2.8 * spacing
        btn_h = 0.05
        
        ap = self.fig.add_axes([item_x, btn_y, item_w, btn_h])
        self.btn_play = Button(ap, 'PLAY SIMULATION', color='#2C2C2E', hovercolor='#3A3A3C')
        self.btn_play.label.set_color('#FFFFFF')
        self.btn_play.label.set_weight('bold')
        self.btn_play.label.set_fontsize(10)

        b1 = self.fig.add_axes([item_x, btn_y - btn_h - 0.02, item_w, btn_h])
        self.btn_rerand = Button(b1, "RERANDOMIZE GATES", color='#2C2C2E', hovercolor='#3A3A3C')
        self.btn_rerand.label.set_color('#FFFFFF')
        self.btn_rerand.label.set_fontsize(10)

        # TOGGLES
        toggle_y = btn_y - 2.8 * btn_h - 0.03
        tw = 0.045; th = 0.022
        
        sm_ax = self.fig.add_axes([item_x + item_w - tw, toggle_y, tw, th])
        self.sw_sm = ToggleSwitch(sm_ax, "Smoothing Algorithm", initial_state=self.smooth, 
                                  on_callback=self.on_smooth)
        
        fov_ax = self.fig.add_axes([item_x + item_w - tw, toggle_y - spacing, tw, th])
        self.sw_fov = ToggleSwitch(fov_ax, "FOV Visualization", initial_state=self.show_fov, 
                                   on_callback=self.on_fov)
        
        tr_ax = self.fig.add_axes([item_x + item_w - tw, toggle_y - 2*spacing, tw, th])
        self.sw_tr = ToggleSwitch(tr_ax, "Rover Path Trails", initial_state=self.show_trails, 
                                  on_callback=self.on_trails)

        self.sld_d.on_changed(self.on_docks)
        self.sld_r.on_changed(self.on_rovers)
        self.btn_rerand.on_clicked(lambda _ : self.reset(self.seed + 1))
        self.btn_play.on_clicked(self.on_play)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # pause banner + controls HUD
        self.pause_text = ax.text(0.5, 0.965, "", transform=ax.transAxes,
                                  ha="center", va="top", fontsize=15,
                                  color=(1, 1, 1, 0.92),
                                  bbox=dict(boxstyle="round,pad=0.35", fc=(0, 0, 0, 0.35), ec=(1, 1, 1, 0.18)))
        self.pause_text.set_visible(False)

        self.hud = ax.text(0.01, 0.01,
                           "Click: add/remove Human (H) or dispatch Rover   |   Right-click: delete nearest H",
                           transform=ax.transAxes, ha="left", va="bottom",
                           fontsize=10, color=(0.95, 0.95, 0.98, 0.65))
        self.hud.set_clip_on(False)

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

        # rover FOV wedges are per-rover patches (remove via rover refs)
        for r in self.rovers:
            if r.fov_front is not None:
                try: r.fov_front.remove()
                except: pass
                r.fov_front = None
            if r.fov_back is not None:
                try: r.fov_back.remove()
                except: pass
                r.fov_back = None

        self._humans_clear_art()

    def redraw(self):
        ax = self.ax
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        self.clear_art()

        for i, g in enumerate(self.gates):
            col = self.gate_colors[i % len(self.gate_colors)]
            for w0, w1 in g.walls:
                ln, = ax.plot([w0[0], w1[0]], [w0[1], w1[1]], lw=3.2, alpha=0.9,
                              color=col, solid_capstyle="round")
                self.gate_art.append(ln)
            self.gate_art.append(ax.scatter([g.open_p1[0], g.open_p2[0]],
                                            [g.open_p1[1], g.open_p2[1]],
                                            s=24, color=col, alpha=0.95))
            self.gate_art.append(ax.scatter([g.approach[0]], [g.approach[1]],
                                            s=70, marker="x", color=(1,1,1), alpha=0.75))
            self.gate_art.append(ax.scatter([g.dock[0]], [g.dock[1]],
                                            s=55, marker="o", edgecolors=col, facecolors="none", linewidths=2))
            self.gate_art.append(ax.text(g.center[0], g.center[1], f"G{i}",
                                         color=(0.95,0.95,0.98), ha="center", va="center",
                                         fontsize=11, alpha=0.95))

        self.rover_art = []
        for r in self.rovers:
            col = self.rover_colors[r.id % len(self.rover_colors)]
            poly = Polygon(square_corners((r.state[0], r.state[1]), r.state[2], S),
                           closed=True, facecolor=col, edgecolor=(0.1,0.1,0.1),
                           lw=2.2, alpha=0.95)
            ax.add_patch(poly)
            line, = ax.plot([], [], lw=2.0, alpha=0.55, color=col)
            r.poly, r.line = poly, line
            r.line.set_visible(self.show_trails)
            r.trail.clear()
            r.trail.append((float(r.state[0]), float(r.state[1])))
            self.rover_art.append((poly, line))

            # FOV wedges (front/back)
            cam_f, cam_b = self._cam_positions(r)
            deg_half = HUMAN_FOV_DEG / 2.0
            theta_f = math.degrees(cam_f[2])
            theta_b = math.degrees(cam_b[2])

            wf = Wedge((cam_f[0], cam_f[1]), r=HUMAN_RANGE,
                       theta1=theta_f - deg_half, theta2=theta_f + deg_half,
                       width=None, facecolor=col, edgecolor=None, alpha=0.12)
            wb = Wedge((cam_b[0], cam_b[1]), r=HUMAN_RANGE,
                       theta1=theta_b - deg_half, theta2=theta_b + deg_half,
                       width=None, facecolor=col, edgecolor=None, alpha=0.12)
            wf.set_visible(self.show_fov)
            wb.set_visible(self.show_fov)
            ax.add_patch(wf); ax.add_patch(wb)
            r.fov_front = wf
            r.fov_back = wb

        # humans
        self._humans_redraw_art()

        self.auto = False
        self.btn_play.label.set_text("Play")
        ax.set_title("")
        self.fig.canvas.draw_idle()

    def update_fov_art(self):
        if not self.rovers:
            return
        deg_half = HUMAN_FOV_DEG / 2.0
        for r in self.rovers:
            if r.fov_front is None or r.fov_back is None:
                continue
            cam_f, cam_b = self._cam_positions(r)
            theta_f = math.degrees(cam_f[2])
            theta_b = math.degrees(cam_b[2])

            r.fov_front.set_center((cam_f[0], cam_f[1]))
            r.fov_front.set_theta1(theta_f - deg_half)
            r.fov_front.set_theta2(theta_f + deg_half)
            r.fov_front.set_radius(HUMAN_RANGE)
            r.fov_front.set_visible(self.show_fov)

            r.fov_back.set_center((cam_b[0], cam_b[1]))
            r.fov_back.set_theta1(theta_b - deg_half)
            r.fov_back.set_theta2(theta_b + deg_half)
            r.fov_back.set_radius(HUMAN_RANGE)
            r.fov_back.set_visible(self.show_fov)

    # ----- events -----
    def on_smooth(self, state):
        self.smooth = state
        self.fig.canvas.draw_idle()

    def on_fov(self, state):
        self.show_fov = state
        self.update_fov_art()
        self.fig.canvas.draw_idle()

    def on_trails(self, state):
        self.show_trails = state
        for r in self.rovers:
            if r.line:
                r.line.set_visible(self.show_trails)
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
        self.btn_play.label.set_text("STOP SIMULATION" if self.auto else "PLAY SIMULATION")
        if self.auto and (not self.paused):
            for r in self.rovers:
                if r.docked and (not r.moving) and r.plan_cd == 0 and r.plan_future is None:
                    nxt = self.furthest_dock(r.current_gate, r.previous_gate, r.id)
                    if nxt >= 0:
                        if not self.start_plan_job(r, nxt, keep_moving=False):
                            r.plan_cd = 30
        self.fig.canvas.draw_idle()

    def nearest_gate(self, xy):
        d = [float(np.linalg.norm(g.dock - xy)) for g in self.gates]
        i = int(np.argmin(d))
        return i, d[i]

    def on_click(self, ev):
        if ev.inaxes != self.ax:
            return

        is_right = (ev.button == 3)
        if is_right:
            self.delete_human_at(ev.xdata, ev.ydata)
            return

        if ev.button == 1:
            # Check if clicking an existing human
            hi, hd = self._human_nearest(ev.xdata, ev.ydata)
            if hi >= 0 and hd <= HUMAN_CLICK_TOL:
                self.humans.pop(hi)
                self._humans_redraw_art()
                return

            # Check if clicking a gate for dispatch
            click = np.array([ev.xdata, ev.ydata], float)
            gi, g_dist = self.nearest_gate(click)
            if g_dist <= 1.6 * S:
                # Dispatch logic
                if (not self.auto) and any(rr.moving for rr in self.rovers):
                    return
                r = next((rr for rr in self.rovers
                          if rr.docked and (not rr.moving) and rr.plan_future is None and rr.current_gate != gi), None)
                if r is not None:
                    self.start_plan_job(r, gi, keep_moving=False)
                return

            # Otherwise, add a new human
            self.toggle_human_at(ev.xdata, ev.ydata)

    # ----- animation -----
    def animate(self, _):
        self.frame += 1
        now = time.monotonic()

        # pause logic (human in FOV)
        self.update_pause_state(now)

        # update pause title
        if self.paused:
            title_msg = "HUMAN DETECTED"
            if self.pause_clear_deadline is not None:
                rem = max(0.0, self.pause_clear_deadline - now)
                title_msg = f"RESUMING IN {rem:0.1f}s"
            self.ax.set_title(title_msg, color='yellow', fontsize=18, weight='bold')
        else:
            self.ax.set_title("")

        # allow background plans to complete even while paused (no motion though)
        self.poll_plan_jobs()

        # freeze motion + auto scheduling while paused
        if not self.paused:
            for r in self.rovers:
                if r.plan_cd > 0:
                    r.plan_cd -= 1
                if r.replan_cd > 0:
                    r.replan_cd -= 1
                if r.yield_cd > 0:
                    r.yield_cd -= 1

            waiting = self.resolve_traffic()

            # background replan when truly stuck (async, never blocks others)
            for r in self.rovers:
                if not (r.moving and r.path):
                    continue
                if r.goal_gate >= 0 and r.stuck >= REPLAN_TRIGGER and r.replan_cd == 0 and r.plan_future is None:
                    self.start_plan_job(r, r.goal_gate, keep_moving=True)
                    r.replan_cd = REPLAN_COOLDOWN

            # arrivals + autoplan
            for r in self.rovers:
                if not (r.moving and r.path):
                    continue

                if r.idx >= len(r.path) - 1:
                    if self.dock_free(r.goal_gate, r.id):
                        g = self.gates[r.goal_gate]
                        r.state = (float(g.dock[0]), float(g.dock[1]), float(g.heading))
                        r.trail.append((float(r.state[0]), float(r.state[1])))

                        r.previous_gate, r.current_gate = r.current_gate, r.goal_gate
                        r.goal_gate = -1
                        r.path = None
                        r.moving = False
                        r.docked = True
                        r.stuck = 0
                        r.reversing = False
                        r.force_step = 0
                        r.yield_cd = 0

                        if self.auto and r.plan_cd == 0 and r.plan_future is None:
                            nxt = self.furthest_dock(r.current_gate, r.previous_gate, r.id)
                            if nxt >= 0 and (not self.start_plan_job(r, nxt, keep_moving=False)):
                                r.plan_cd = 30
                    else:
                        r.goal_gate = -1
                        r.path = None
                        r.moving = False
                        r.reversing = False
                        r.force_step = 0

            if self.auto:
                for r in self.rovers:
                    if r.docked and (not r.moving) and r.goal_gate == -1 and r.plan_cd == 0 and r.plan_future is None:
                        nxt = self.furthest_dock(r.current_gate, r.previous_gate, r.id)
                        if nxt >= 0 and (not self.start_plan_job(r, nxt, keep_moving=False)):
                            r.plan_cd = 30

        # draw rovers + trails + FOV
        for r in self.rovers:
            if r.poly:
                r.poly.set_xy(square_corners((r.state[0], r.state[1]), r.state[2], S))
            if r.line and self.show_trails:
                lx = [p[0] for p in r.trail]
                ly = [p[1] for p in r.trail]
                r.line.set_data(lx, ly)
                r.line.set_visible(True)
            elif r.line:
                r.line.set_visible(False)

        self.update_fov_art()
        return []

    def run(self):
        self.setup_vis()
        self.reset(BASE_SEED)
        self.ani = FuncAnimation(self.fig, self.animate, interval=FRAME_MS, blit=False)
        try:
            plt.show()
        finally:
            self._shutdown_pool()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    Sim().run()
