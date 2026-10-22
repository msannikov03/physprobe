"""MuJoCo scene: grey floor plane, one cube on a free joint, fixed top-down camera `cam`.

Friction: floor contacts are frictionless (condim=1); Coulomb friction mu*m*g acts on the cube's x and y
degrees of freedom as MuJoCo dry friction (dof_frictionloss), with mu = max(cube, floor) as in MuJoCo's
pair mixing. MuJoCo's soft friction cones inflate the normal force while sliding fast (the cube hops and
pitches up to 11 deg at mu=0.8), the decoupled form gives N = m*g and deceleration mu*g.

Pixel convention (project, origin_px): u = column, v = row, pixel centres at integer indices,
row 0 at the top. World +x maps to +u (right), world +y maps to -v (up).

Render quantisation (known limitation of sim-v1, kept so existing data stays reproducible): with
offsamples=4 MSAA an edge's coverage moves in steps of about 1/4 px (measured on compute: rendered left
edge at 17.749 / 27.246 / 34.749 px vs projected 17.771 / 27.270 / 34.765), i.e. ~1.4 mm at the cube centre.
Position differences below that can render byte-identically (poscheck: 22 pairs of identical clips with
dx0 of 0.016-0.82 mm). The shortest push slides (~15.7 mm, ~2.7 px) span ~11 levels. New datasets that
need finer sub-pixel motion (push_nuis, push_ood) should raise offsamples (8-16) or render at 512 px and
downsample; that changes the pixels, so it needs a SIM_VERSION bump and regeneration.
"""
import functools
import math

import mujoco
import numpy as np

DT = 1.0 / 1280
SUBSTEPS = 20
FPS = 64
T = 64
H = W = 256
G = 9.81
CUBE_HALF = 0.05
FLOOR_FRICTION = 0.01
CUBE_TORSION = 0.005
CUBE_ROLL = 0.0001
FRICTION_SOLREF = (0.002, 1.0)
PATH_M = 1.2
VIEW_M = 1.5
START_FRAC = 0.12
FOVY_DEG = 30.0
F_PX = (H / 2) / math.tan(math.radians(FOVY_DEG / 2))
CAM_Z = F_PX * VIEW_M / W
CAM_X = (0.5 - START_FRAC) * VIEW_M
CAM_POS = (CAM_X, 0.0, CAM_Z)
CAM_XYAXES = (1, 0, 0, 0, 1, 0)
M_PER_PX_FLOOR = VIEW_M / W
CUBE_RGBA = (0.90, 0.15, 0.10, 1.0)
FLOOR_RGBA = (0.50, 0.50, 0.50, 1.0)
SETTLE_S = 1.0


def _f(v):
    return " ".join(f"{x:.10g}" for x in v)


XML = f"""
<mujoco model="physprobe">
  <option timestep="{DT:.10g}" gravity="0 0 {-G:.10g}"/>
  <statistic extent="2" center="{CAM_X:.10g} 0 0"/>
  <visual>
    <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="4096" offsamples="4"/>
    <global offwidth="{W}" offheight="{H}"/>
  </visual>
  <asset>
    <material name="floor" rgba="{_f(FLOOR_RGBA)}" specular="0" shininess="0" reflectance="0"/>
    <material name="cube" rgba="{_f(CUBE_RGBA)}" specular="0.1" shininess="0.1" reflectance="0"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="{CAM_X:.10g} 0 4" dir="0.25 0.35 -1"
           diffuse="0.75 0.75 0.75" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="floor" condim="1"
          friction="{FLOOR_FRICTION:.10g} 0.001 0.0001"/>
    <camera name="cam" mode="fixed" pos="{_f(CAM_POS)}" xyaxes="{_f(CAM_XYAXES)}" fovy="{FOVY_DEG:.10g}"/>
    <body name="cube" pos="0 0 {CUBE_HALF:.10g}">
      <joint name="cube" type="free" solreffriction="{_f(FRICTION_SOLREF)}"/>
      <inertial pos="0 0 0" mass="1" diaginertia="{_f([(2 * CUBE_HALF) ** 2 / 6] * 3)}"/>
      <geom name="cube" type="box" size="{_f([CUBE_HALF] * 3)}" material="cube" condim="1"
            friction="0.5 {CUBE_TORSION:.10g} {CUBE_ROLL:.10g}"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_model():
    return mujoco.MjModel.from_xml_string(XML)


def ids(model):
    """-> (cube body id, cube geom id, floor geom id, first dof of the cube's free joint)"""
    b = model.body("cube").id
    return b, model.geom("cube").id, model.geom("floor").id, int(model.jnt_dofadr[model.body_jntadr[b]])


def set_params(model, m, mu, data=None):
    """Cube mass m, inertia m(2a)^2/6, geom friction [mu, 0.005, 0.0001], dry friction mu*m*g on x, y.
    With `data`, also refresh MuJoCo's compile-time constants (invweight0), which set constraint softness:
    without this the resting penetration and the friction response depend on m. Render statistics are kept."""
    b, g, fl, dof = ids(model)
    model.body_mass[b] = m
    model.body_inertia[b] = m * (2 * CUBE_HALF) ** 2 / 6
    model.geom_friction[g] = [mu, CUBE_TORSION, CUBE_ROLL]
    mu_pair = max(float(model.geom_friction[g, 0]), float(model.geom_friction[fl, 0]))
    model.dof_frictionloss[dof:dof + 6] = [mu_pair * m * G, mu_pair * m * G, 0, 0, 0, 0]
    if data is not None:
        extent, center = float(model.stat.extent), model.stat.center.copy()
        mujoco.mj_setConst(model, data)
        model.stat.extent = extent
        model.stat.center[:] = center
        mujoco.mj_resetData(model, data)


def place(model, data, x0, z):
    """Reset and put the cube at (x0, 0, z), unrotated, at rest."""
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = [x0, 0.0, z, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0


@functools.lru_cache(maxsize=None)
def rest_z():
    """COM height of the cube resting on the floor after SETTLE_S of settling (mass-independent)."""
    model = build_model()
    data = mujoco.MjData(model)
    set_params(model, 1.0, 0.5, data)
    place(model, data, 0.0, CUBE_HALF)
    for _ in range(int(round(SETTLE_S / DT))):
        mujoco.mj_step(model, data)
    return float(data.qpos[2])


def project(p):
    """World points (..., 3) -> pixel coords (..., 2) [u, v] for camera `cam`."""
    p = np.asarray(p, dtype=np.float64)
    depth = CAM_Z - p[..., 2]
    u = W / 2 + F_PX * (p[..., 0] - CAM_POS[0]) / depth - 0.5
    v = H / 2 - F_PX * (p[..., 1] - CAM_POS[1]) / depth - 0.5
    return np.stack([u, v], axis=-1)


def m_per_px(z):
    """Metres per pixel on the horizontal plane at height z."""
    return (CAM_Z - z) / F_PX


def camera_info():
    z = rest_z()
    return dict(pos=[float(c) for c in CAM_POS], xyaxes=list(CAM_XYAXES), fovy=FOVY_DEG,
                m_per_px_floor=float(M_PER_PX_FLOOR), origin_px=[float(c) for c in project([0.0, 0.0, 0.0])],
                m_per_px_com=float(m_per_px(z)), origin_px_com=[float(c) for c in project([0.0, 0.0, z])],
                m_per_px_top=float(m_per_px(z + CUBE_HALF)),
                origin_px_top=[float(c) for c in project([0.0, 0.0, z + CUBE_HALF])],
                image_wh=[W, H], view_m=VIEW_M, start_frac=START_FRAC,
                px_convention="u=col, v=row, pixel centres at integer indices, row 0 top; "
                              "world +x -> +u, world +y -> -v; u_px = origin_px[0] + x / m_per_px")


def quat_mat(q):
    """Quaternions (..., 4) [w, x, y, z] -> rotation matrices (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1).reshape(q.shape[:-1] + (3, 3))


def tilt_yaw_deg(q):
    """Tilt (angle of the cube's z axis from vertical) and |yaw| in degrees for quaternions (..., 4)."""
    R = quat_mat(q)
    tilt = np.degrees(np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0)))
    yaw = np.abs(np.degrees(np.arctan2(R[..., 1, 0], R[..., 0, 0])))
    return tilt, yaw


_SIGNS = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)


def cube_corners(pos, quat):
    """Corner positions (..., 8, 3) for COM positions (..., 3) and quaternions (..., 4)."""
    R = quat_mat(quat)
    return np.asarray(pos, dtype=np.float64)[..., None, :] + np.einsum("...ab,jb->...ja", R, _SIGNS * CUBE_HALF)
