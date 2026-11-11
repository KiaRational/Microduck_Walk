"""Procedural Microduck obstacle course, after NVIDIA's Microduck blog course (wood ramps / steps / platforms,
rubber mats and metal plates to walk over; wood boxes, flight cases, C-stands with sandbags and tape-roll
stacks to walk around). Sizes are scaled to a 25 cm robot:

* walk-over props: ramps <= ~12 % slope; stairs of 2-4 steps up and down with risers <= ``max_riser``
  (3.5 cm at full difficulty); low platforms; mats and plates <= 1 cm;
* avoid props: >= 15 cm tall and >= 6 cm wide, so they show up in a 32x24 depth image.

Every prop is part of the one terrain mesh (physics and the ray-cast depth camera read only ``/World/ground``);
materials are vertex colours, which the terrain importer keeps as ``displayColor``.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import FlatPatchSamplingCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass

FLOOR = (196, 205, 214)
WOOD = (186, 132, 78)
WOOD_DARK = (140, 92, 52)
RUBBER = (34, 34, 36)
METAL = (190, 194, 200)
CASE = (28, 28, 30)
SILVER = (160, 162, 168)
SANDBAG = (20, 20, 20)
TAPE = (40, 36, 34)


def _paint(mesh: trimesh.Trimesh, rgb) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.array([*rgb, 255], np.uint8), (len(mesh.vertices), 1))
    return mesh


def _box(size, center, yaw, rgb):
    m = trimesh.creation.box(extents=size)
    tf = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
    tf[:3, 3] = center
    m.apply_transform(tf)
    return _paint(m, rgb)


def _cyl(radius, height, center, rgb, sections=10):
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_translation(center)
    return _paint(m, rgb)


def _hull(points, xy, yaw, rgb):
    """Convex prism from local points (x, y, z), placed at xy with yaw."""
    m = trimesh.convex.convex_hull(np.asarray(points, float))
    tf = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
    tf[:2, 3] = xy
    m.apply_transform(tf)
    return _paint(m, rgb)


# ---------------------------------------------------------------- walk-over props (local frame, centred at 0)

def wood_ramp_platform(rng, d, cfg):
    """Ramp up - flat top - ramp down (a trapezoid prism)."""
    h = rng.uniform(0.3, 1.0) * cfg.max_ramp * d + 0.004
    ramp = h / rng.uniform(0.06, cfg.max_slope)
    top, w = rng.uniform(0.15, 0.45), rng.uniform(0.3, 0.6)
    x0, x1 = top / 2 + ramp, top / 2
    pts = [(sx, sy * w / 2, 0.0) for sx in (-x0, x0) for sy in (-1, 1)]
    pts += [(sx, sy * w / 2, h) for sx in (-x1, x1) for sy in (-1, 1)]
    return pts, 2 * x0, w


def wedge_ramp(rng, d, cfg):
    """One-sided wedge that drops off at the high end (drop <= max_riser)."""
    h = rng.uniform(0.3, 1.0) * cfg.max_riser * d + 0.004
    ramp, w = h / rng.uniform(0.06, cfg.max_slope), rng.uniform(0.25, 0.45)
    pts = [(-ramp / 2, sy * w / 2, 0.0) for sy in (-1, 1)]
    pts += [(ramp / 2, sy * w / 2, z) for sy in (-1, 1) for z in (0.0, h)]
    return pts, ramp, w


def place_walk_over(rng, d, cfg, xy, yaw):
    kind = rng.choice(["ramp_platform", "wedge", "steps", "platform", "mat", "plate"],
                      p=[0.2, 0.1, 0.4, 0.1, 0.1, 0.1])
    if kind in ("ramp_platform", "wedge"):
        pts, lx, ly = (wood_ramp_platform if kind == "ramp_platform" else wedge_ramp)(rng, d, cfg)
        return [_hull(pts, xy, yaw, WOOD)], (lx, ly)
    c, s = np.cos(yaw), np.sin(yaw)
    to_w = lambda lx, ly: (xy[0] + c * lx - s * ly, xy[1] + s * lx + c * ly)  # noqa: E731
    if kind == "steps":  # wooden stairs: 2-4 steps up, a landing, and the same steps down
        n = rng.integers(2, 5)
        rise = 0.005 + rng.uniform(0.5, 1.0) * (cfg.max_riser - 0.005) * d
        run, w, landing = rng.uniform(0.12, 0.18), rng.uniform(0.35, 0.6), rng.uniform(0.2, 0.35)
        meshes = []
        for i in range(n):  # stacked slabs, each shorter by one tread on both ends
            length = landing + 2 * run * (n - 1 - i)
            meshes.append(_box((length, w, rise), (*xy, rise * (i + 0.5)), yaw, WOOD if i % 2 else WOOD_DARK))
        return meshes, (landing + 2 * run * (n - 1), w)
    if kind == "platform":  # low wooden platform
        h = 0.005 + rng.uniform(0.5, 1.0) * (cfg.max_riser - 0.005) * d
        lx, ly = rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6)
        return [_box((lx, ly, h), (*xy, h / 2), yaw, WOOD)], (lx, ly)
    if kind == "mat":  # studded rubber mat
        lx, ly, t, stud = rng.uniform(0.35, 0.6), rng.uniform(0.3, 0.5), 0.004, 0.004 + 0.004 * d
        meshes = [_box((lx, ly, t), (*xy, t / 2), yaw, RUBBER)]
        for sx in np.arange(-lx / 2 + 0.03, lx / 2 - 0.02, 0.06):
            for sy in np.arange(-ly / 2 + 0.03, ly / 2 - 0.02, 0.06):
                meshes.append(_cyl(0.012, stud, (*to_w(sx, sy), t + stud / 2), RUBBER, sections=6))
        return meshes, (lx, ly)
    lx, ly, t = rng.uniform(0.35, 0.6), rng.uniform(0.3, 0.5), 0.003 + 0.004 * d  # metal plate
    return [_box((lx, ly, t), (*xy, t / 2), yaw, METAL)], (lx, ly)


# ---------------------------------------------------------------- avoid props (tall, solid)

def place_avoid(rng, d, xy, yaw):
    kind = rng.choice(["wood_box", "flight_case", "c_stand", "tape_stack", "box_stack"], p=[0.3, 0.2, 0.2, 0.15, 0.15])
    if kind == "wood_box":
        s = rng.uniform(0.15, 0.3)
        return [_box((s, s * rng.uniform(0.8, 1.2), max(0.15, s)), (*xy, max(0.15, s) / 2), yaw, WOOD_DARK)], s * 0.75
    if kind == "flight_case":
        lx, ly, h = rng.uniform(0.3, 0.5), rng.uniform(0.2, 0.3), rng.uniform(0.22, 0.35)
        meshes = [_box((lx, ly, h), (*xy, h / 2 + 0.02), yaw, CASE)]
        meshes.append(_box((lx * 0.9, ly * 0.9, 0.02), (*xy, 0.01), yaw, SILVER))  # castor base
        return meshes, max(lx, ly) / 2 + 0.03
    if kind == "c_stand":  # tripod base + sandbag + pole (the pole is wide enough to be seen in depth)
        meshes, r = [], rng.uniform(0.14, 0.2)
        for k in range(3):
            a = yaw + k * 2 * np.pi / 3
            meshes.append(_box((r, 0.025, 0.025), (xy[0] + np.cos(a) * r / 2, xy[1] + np.sin(a) * r / 2, 0.0125), a, SILVER))
        meshes.append(_box((0.06, 0.06, 0.9), (*xy, 0.45), yaw, SILVER))
        meshes.append(_box((0.16, 0.1, 0.08), (xy[0] + np.cos(yaw) * 0.08, xy[1] + np.sin(yaw) * 0.08, 0.065), yaw, SANDBAG))
        meshes.append(_box((0.12, 0.05, 0.1), (*xy, 0.95), yaw, (230, 120, 40)))  # LED light head
        return meshes, r + 0.03
    if kind == "tape_stack":
        n = rng.integers(3, 5)
        return [_cyl(0.06, 0.05, (*xy, 0.025 + 0.05 * i), TAPE if i % 2 == 0 else (150, 120, 90)) for i in range(n)], 0.08
    s = rng.uniform(0.18, 0.25)  # two stacked boxes
    return [_box((s, s, s), (*xy, s / 2), yaw, WOOD_DARK),
            _box((s * 0.8, s * 0.8, s * 0.8), (*xy, s + s * 0.4), yaw + 0.3, WOOD)], s * 0.75


# ---------------------------------------------------------------- tile

def obstacle_course_terrain(difficulty: float, cfg: "MicroduckCourseCfg"):
    rng = np.random.default_rng(np.random.randint(0, 2**31))
    sx, sy = cfg.size
    centre = np.array([sx / 2, sy / 2])
    meshes = [_paint(trimesh.creation.box(extents=(sx, sy, 0.02)).apply_translation((sx / 2, sy / 2, -0.01)), FLOOR)]
    # thin dark outline (1 mm high) so each robot's 3x3 m tile is visible in the viewer
    for (cx, cy, lx, ly) in ((sx / 2, 0.015, sx, 0.03), (sx / 2, sy - 0.015, sx, 0.03),
                             (0.015, sy / 2, 0.03, sy), (sx - 0.015, sy / 2, 0.03, sy)):
        meshes.append(_box((lx, ly, 0.001), (cx, cy, 0.0005), 0.0, (90, 96, 104)))
    taken = [(centre, cfg.spawn_clear_radius)]

    def free(p, r):
        if np.any(p - r < 0.05) or p[0] + r > sx - 0.05 or p[1] + r > sy - 0.05:
            return False
        return all(np.linalg.norm(p - q) > r + rq + 0.05 for q, rq in taken)

    n_walk = int(round(cfg.num_walk_over[0] + difficulty * (cfg.num_walk_over[1] - cfg.num_walk_over[0])))
    n_avoid = int(round(cfg.num_avoid[0] + difficulty * (cfg.num_avoid[1] - cfg.num_avoid[0])))
    for _ in range(n_walk):
        for _try in range(30):
            p, yaw = rng.uniform(0.3, [sx - 0.3, sy - 0.3]), rng.uniform(-np.pi, np.pi)
            ms, (lx, ly) = place_walk_over(rng, difficulty, cfg, p, yaw)
            r = 0.5 * np.hypot(lx, ly)
            if free(p, r):
                meshes += ms
                taken.append((p, r))
                break
    for _ in range(n_avoid):
        for _try in range(30):
            p, yaw = rng.uniform(0.2, [sx - 0.2, sy - 0.2]), rng.uniform(-np.pi, np.pi)
            ms, r = place_avoid(rng, difficulty, p, yaw)
            if free(p, r):
                meshes += ms
                taken.append((p, r))
                break
    return meshes, np.array([sx / 2, sy / 2, 0.0])


@configclass
class MicroduckCourseCfg(SubTerrainBaseCfg):
    function = obstacle_course_terrain
    max_riser: float = 0.035
    """Max single step height (stair riser, platform, wedge drop) at difficulty 1 [m]."""
    max_ramp: float = 0.05
    """Max ramp-platform height at difficulty 1 [m] (reached at <= max_slope)."""
    max_slope: float = 0.12
    num_walk_over: tuple[int, int] = (3, 7)
    num_avoid: tuple[int, int] = (0, 0)
    spawn_clear_radius: float = 0.3


def walk_over_course(**kw) -> MicroduckCourseCfg:
    """Walking-policy terrain: only props the robot should walk over (the walker is blind)."""
    return MicroduckCourseCfg(num_walk_over=(5, 11), num_avoid=(0, 0), spawn_clear_radius=0.3, **kw)


def goal_patches(radius: float = 1.3) -> dict[str, FlatPatchSamplingCfg]:
    """Goal positions on floor or low props, at least ~12 cm away from anything tall."""
    return {"target": FlatPatchSamplingCfg(num_patches=32, patch_radius=[0.06, 0.12], max_height_diff=0.03,
                                           x_range=(-radius, radius), y_range=(-radius, radius), z_range=(-0.02, 0.05))}


def full_course(**kw) -> MicroduckCourseCfg:
    """Navigation terrain: walk-over props plus tall props to avoid."""
    return MicroduckCourseCfg(num_walk_over=(4, 9), num_avoid=(4, 12), spawn_clear_radius=0.3,
                              flat_patch_sampling=goal_patches(), **kw)



def one_robot_per_tile(env, env_ids):
    """Startup event: put env i on its own terrain tile (Isaac Lab otherwise assigns tiles randomly, so several
    robots share a tile: they don't collide with each other, but they overlap). Needs >= num_envs tiles."""
    import torch

    t = env.scene.terrain
    rows, cols = t.terrain_origins.shape[:2]
    idx = torch.arange(env.num_envs, device=t.env_origins.device) % (rows * cols)
    t.terrain_levels = idx % rows
    t.terrain_types = idx // rows
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
