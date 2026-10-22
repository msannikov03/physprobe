"""Generate push / slide / poscheck clips per CONTRACT v1.

  $SIM_PY -m physprobe.sim.generate --scenario push --n 2000 --dataset push --workers 10

Per clip: mj_forward, render frame 0 at rest, set qvel[0] (J/m for push, v0 for slide, 0 for poscheck),
then 64 frames x 20 mj_step. Frame k and state row k are taken at t = k/64 (state row 0 already holds the
post-impulse velocity); contact row k is the mean over the 20 substeps of [k/64, (k+1)/64).
Resumable: a clip is done when its three .npy files exist and its record in .parts/ carries the dataset's
parts tag (a hash of the config and of physprobe/sim/*.py, fixed at first generation).

Provenance: the first generation of a dataset stores git_commit (short HEAD, '-dirty' if the tree is dirty),
sim_code_sha256 and created in .parts/config.json under "provenance"; later runs copy them into dataset.json
unchanged. Resuming with clips left to generate is refused when the sim code hash differs (use --overwrite).
Datasets whose config.json predates this (no "provenance") can be re-assembled but not extended.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np

from physprobe import io
from physprobe.sim import scene as S

SIM_VERSION = "sim-v1"
V_STOP = 1e-3
TILT_MAX_DEG = 5.0
YAW_MAX_DEG = 5.0
EXTRA_S = 3.0
BASE_SEEDS = {"push": 0, "slide": 100000, "poscheck": 300000}
DATASET_SEEDS = {"push_ood": 200000, "push_nuis": 400000}   # by dataset name, so they never reuse push seeds
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = int(os.environ.get("PHYSPROBE_MAX_WORKERS", "10"))
NICE = 10
_BAD = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)
_FRICTION_DOF = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF


def sim_code_sha256():
    """sha256 over physprobe/sim/*.py (sorted by name; name and bytes of each file)."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(SIM_DIR, "*.py"))):
        h.update(os.path.basename(p).encode() + b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def sample_params(seed, m_range, mu_range, scenario):
    """m log-uniform, then mu uniform, then (poscheck only) x0 uniform on the path."""
    rng = np.random.default_rng(seed)
    m = float(np.exp(rng.uniform(np.log(m_range[0]), np.log(m_range[1]))))
    mu = float(rng.uniform(mu_range[0], mu_range[1]))
    x0 = float(rng.uniform(0.0, S.PATH_M)) if scenario == "poscheck" else 0.0
    return m, mu, x0


class Sim:
    """One model, data and (optionally) renderer; reused across clips."""

    def __init__(self, render=True):
        self.model = S.build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, S.H, S.W) if render else None
        self.body, self.geom, self.floor, self.dof = S.ids(self.model)
        self._pair = {self.geom, self.floor}
        self._f6 = np.zeros(6)

    def _render(self, out):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="cam")
        self.renderer.render(out=out)

    def _forces(self):
        """-> (sum of normal forces over cube-floor contacts, |floor tangential reaction|, max penetration)"""
        d = self.data
        fn = pen = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if {int(con.geom[0]), int(con.geom[1])} == self._pair:
                mujoco.mj_contactForce(self.model, d, c, self._f6)
                fn += self._f6[0]
                pen = max(pen, -float(con.dist))
        n = d.nefc
        sel = (d.efc_type[:n] == _FRICTION_DOF) & ((d.efc_id[:n] == self.dof) | (d.efc_id[:n] == self.dof + 1))
        f = d.efc_force[:n][sel]
        ft = float(math.sqrt(float(np.dot(f, f))))
        return fn, ft, pen

    def run(self, m, mu, vx0, x0=0.0, render=True):
        """Simulate one clip. -> dict with clip (or None), contact, state and per-clip diagnostics."""
        model, d = self.model, self.data
        S.set_params(model, m, mu, d)
        S.place(model, d, x0, S.rest_z())
        mujoco.mj_forward(model, d)
        T, K = S.T, S.SUBSTEPS
        clip = np.empty((T, S.H, S.W, 3), np.uint8) if (render and self.renderer is not None) else None
        if clip is not None:
            self._render(clip[0])
        d.qvel[0] = vx0
        state = np.empty((T, 6))
        quat = np.empty((T, 4))
        contact = np.empty((T, 3))
        qsub = np.empty((T * K, 4))
        vsub = np.empty(T * K)
        fsub = np.empty((T * K, 2))
        zpen = 0.0
        for k in range(T):
            if k > 0 and clip is not None:
                self._render(clip[k])
            state[k, :3] = d.qpos[:3]
            state[k, 3:] = d.qvel[:3]
            quat[k] = d.qpos[3:7]
            for j in range(K):
                i = k * K + j
                vsub[i] = math.hypot(d.qvel[0], d.qvel[1])
                mujoco.mj_step(model, d)
                fn, ft, pen = self._forces()
                fsub[i] = fn, ft
                zpen = max(zpen, pen)
                qsub[i] = d.qpos[3:7]
            sl = slice(k * K, (k + 1) * K)
            contact[k] = [fsub[sl, 0].mean(), fsub[sl, 1].mean(), vsub[sl].mean()]
        t_stop = 0.0 if vx0 == 0 else None
        if t_stop is None:
            below = np.nonzero(vsub[1:] < V_STOP)[0]
            if below.size:
                t_stop = (below[0] + 1) * S.DT
            else:
                t_end = T / S.FPS + EXTRA_S
                while d.time < t_end:
                    if math.hypot(d.qvel[0], d.qvel[1]) < V_STOP:
                        t_stop = float(d.time)
                        break
                    mujoco.mj_step(model, d)
        bad = any(d.warning[w].number > 0 for w in _BAD)
        disp_stop = float(math.hypot(d.qpos[0] - x0, d.qpos[1])) if t_stop is not None else None
        return dict(clip=clip, contact=contact, state=state, quat=quat, qsub=qsub, z_pen=zpen,
                    t_stop=t_stop, disp_stop=disp_stop, bad=bad)


def clip_meta(res, x0):
    """Visibility and stability fields from a Sim.run result."""
    state, quat = res["state"], res["quat"]
    corners = S.cube_corners(state[:, :3], quat)
    uv = S.project(corners)
    in_frame = bool(((uv >= -0.5) & (uv <= np.array([S.W, S.H]) - 0.5)).all())
    com_px = S.project(state[:, :3])
    tilt, yaw = S.tilt_yaw_deg(np.concatenate([quat, res["qsub"]]))
    speed = np.hypot(state[:, 3], state[:, 4])
    finite = all(np.isfinite(a).all() for a in (res["contact"], state, res["qsub"])) and not res["bad"]
    if res["clip"] is not None:
        frame0_hash = hashlib.sha1(np.ascontiguousarray(res["clip"][0]).tobytes()).hexdigest()[:16]
    else:
        frame0_hash = None
    max_tilt = float(np.nanmax(tilt)) if finite else float("nan")
    max_yaw = float(np.nanmax(yaw)) if finite else float("nan")
    ok = bool(in_frame and finite and max_tilt < TILT_MAX_DEG and max_yaw < YAW_MAX_DEG)
    t_stop = res["t_stop"]
    return dict(in_frame=in_frame,
                disp_m=round(float(np.hypot(*(state[-1, :2] - state[0, :2]))), 7),
                moved_px=round(float(np.hypot(*(com_px[-1] - com_px[0]))), 4),
                frames_moving=int((speed[1:] > V_STOP).sum()),
                t_stop_s=None if t_stop is None else round(float(t_stop), 6),
                max_tilt_deg=round(max_tilt, 5) if finite else None,
                max_yaw_deg=round(max_yaw, 5) if finite else None,
                z_pen_max_m=round(float(res["z_pen"]), 9),
                ok=ok, moving_at_end=bool(speed[-1] > V_STOP), frame0_hash=frame0_hash)


def make_clip(sim, cfg, index, render=True):
    """-> (meta row, result dict) for clip `index` of a dataset config."""
    seed = cfg["base_seed"] + index
    scenario = cfg["scenario"]
    m, mu, x0 = sample_params(seed, cfg["m_range"], cfg["mu_range"], scenario)
    if scenario == "push":
        J, v0 = cfg["J"], cfg["J"] / m
    elif scenario == "slide":
        J, v0 = None, cfg["v0"]
    else:
        J, v0 = None, 0.0
    res = sim.run(m, mu, v0, x0, render=render)
    split = cfg["split"] if cfg["split"] in ("smoke", "ood") else io.canonical_split(seed)
    row = dict(id=f"{index:06d}", scenario=scenario, m=m, mu=mu, J=J, v0=float(v0), x0_m=x0, seed=int(seed), split=split)
    row.update(clip_meta(res, x0))
    return row, res


# --------------------------------------------------------------------------- multiprocessing workers

_W = {}


def _renice():
    try:
        cur = os.nice(0)
        if cur < NICE:
            os.nice(NICE - cur)
    except OSError:
        pass


def _init(cfg):
    _renice()
    _W["cfg"] = cfg
    _W["sim"] = Sim(render=True)


def _work(index):
    cfg = _W["cfg"]
    t0 = time.time()
    row, res = make_clip(_W["sim"], cfg, index)
    io.write_clip(cfg["ds_dir"], row["id"], res["clip"], res["contact"].astype(np.float32), res["state"].astype(np.float32))
    part = dict(row, _gen_s=round(time.time() - t0, 4), _tag=cfg["_tag"])
    _write_json(os.path.join(cfg["ds_dir"], ".parts", row["id"] + ".json"), part)
    return row["id"], part["_gen_s"]


def _write_json(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, sort_keys=True)
    os.replace(tmp, path)


def _done(ds_dir, id, tag):
    """Clip `id` is finished: its .npy files exist and its .parts record carries `tag` (None = legacy record)."""
    p = os.path.join(ds_dir, ".parts", id + ".json")
    if not os.path.exists(p) or not all(os.path.exists(os.path.join(ds_dir, f"{k}_{id}.npy"))
                                        for k in ("clip", "contact", "state")):
        return False
    try:
        with open(p) as f:
            return json.load(f).get("_tag") == tag
    except (OSError, ValueError):
        return False


def _provenance(cfg, code):
    tag = hashlib.sha256((json.dumps(cfg, sort_keys=True) + code).encode()).hexdigest()[:16]
    return dict(git_commit=io.git_commit(), sim_code_sha256=code, created=io.now_iso(), parts_tag=tag)


# --------------------------------------------------------------------------- dataset level

def dataset_info(cfg, n):
    ranges = dict(m=[cfg["m_range"][0], cfg["m_range"][1], "loguniform"],
                  mu=[cfg["mu_range"][0], cfg["mu_range"][1], "uniform"])
    if cfg["scenario"] == "poscheck":
        ranges["x0_m"] = [0.0, S.PATH_M, "uniform"]
    return dict(name=cfg["name"], scenario=cfg["scenario"], n=n, T=S.T, H=S.H, W=S.W, fps=S.FPS, dt=S.DT,
                substeps=S.SUBSTEPS, ranges=ranges,
                J=cfg["J"] if cfg["scenario"] == "push" else None,
                v0_fixed=cfg["v0"] if cfg["scenario"] == "slide" else (0.0 if cfg["scenario"] == "poscheck" else None),
                cube_half=S.CUBE_HALF, floor_friction=S.FLOOR_FRICTION, camera=S.camera_info(),
                mujoco_version=mujoco.__version__, sim_version=SIM_VERSION, base_seed=cfg["base_seed"],
                split_mode=cfg["split"], g=S.G, path_m=S.PATH_M, rest_z_m=S.rest_z(), v_stop_mps=V_STOP,
                cube_rgba=list(S.CUBE_RGBA), floor_rgba=list(S.FLOOR_RGBA),
                friction_model="Coulomb mu*m*g as MuJoCo dof frictionloss on the cube's x,y dofs "
                               f"(solreffriction {S.FRICTION_SOLREF[0]} {S.FRICTION_SOLREF[1]}); "
                               "floor contacts frictionless (condim=1); ft_N = |dof friction force|",
                frame0="rendered at rest before the velocity is set; state row 0 holds the post-impulse velocity")


def config_from_args(a):
    base = a.base_seed if a.base_seed is not None else BASE_SEEDS[a.scenario]
    if a.base_seed is None and a.dataset in DATASET_SEEDS:
        base = DATASET_SEEDS[a.dataset]
    split = a.split
    if split == "auto":
        split = "smoke" if a.dataset.startswith("smoke") else ("ood" if a.dataset.endswith("_ood") else "canonical")
    return dict(sim_version=SIM_VERSION, name=a.dataset, scenario=a.scenario, base_seed=int(base), J=float(a.J),
                v0=float(a.v0), m_range=[float(x) for x in a.m_range], mu_range=[float(x) for x in a.mu_range],
                split=split)


def generate(cfg, n, workers=8, overwrite=False, log=print):
    """Generate ids 0..n-1 (skipping done ones), then write meta.jsonl and dataset.json. -> list of meta rows"""
    ds_dir = io.dataset_dir(cfg["name"])
    parts = os.path.join(ds_dir, ".parts")
    os.makedirs(parts, exist_ok=True)
    cfg_path = os.path.join(parts, "config.json")
    code = sim_code_sha256()
    prov = None
    if os.path.exists(cfg_path) and not overwrite:
        with open(cfg_path) as f:
            old = json.load(f)
        prov = old.pop("provenance", None)
        if old != cfg:
            raise SystemExit(f"{cfg_path} differs from this run's config:\n  old {old}\n  new {cfg}\n"
                             "use another --dataset or --overwrite")
        legacy = prov is None
    else:
        prov = _provenance(cfg, code)
        _write_json(cfg_path, dict(cfg, provenance=prov))
        legacy = False
    tag = None if legacy else prov["parts_tag"]
    todo = [i for i in range(n) if overwrite or not _done(ds_dir, f"{i:06d}", tag)]
    if todo and legacy:
        raise SystemExit(f"{cfg_path} has no provenance (generated before code hashing); {len(todo)} clips are "
                         "missing and cannot be added with verified code. Use --overwrite or another --dataset.")
    if todo and prov["sim_code_sha256"] != code:
        raise SystemExit(f"physprobe/sim code changed since {cfg['name']} was first generated "
                         f"(sha256 {prov['sim_code_sha256'][:12]} -> {code[:12]}); {len(todo)} clips left. "
                         "Use --overwrite to regenerate everything, or another --dataset.")
    if not legacy and prov["sim_code_sha256"] != code:
        log(f"[{cfg['name']}] note: sim code differs from the generating code; provenance kept as first recorded")
    wcfg = dict(cfg, ds_dir=ds_dir, _tag=tag)
    workers = max(1, min(int(workers), MAX_WORKERS, len(todo) or 1))
    log(f"[{cfg['name']}] {n} clips, {n - len(todo)} done, {len(todo)} to generate, workers={workers}")
    t0 = time.time()
    if todo:
        if workers == 1:
            _init(wcfg)
            it = map(_work, todo)
        else:
            pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(wcfg,))
            it = pool.imap_unordered(_work, todo)
        times = []
        for k, (_, gs) in enumerate(it, 1):
            times.append(gs)
            if k % max(1, len(todo) // 20) == 0 or k == len(todo):
                el = time.time() - t0
                log(f"[{cfg['name']}] {k}/{len(todo)}  {el:.0f}s elapsed, {el / k:.3f} s/clip wall, "
                    f"{np.mean(times):.3f} s/clip per worker, eta {el / k * (len(todo) - k):.0f}s")
        if workers > 1:
            pool.close()
            pool.join()
    rows = []
    for i in range(n):
        with open(os.path.join(parts, f"{i:06d}.json")) as f:
            rows.append({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    io.write_meta(ds_dir, rows)
    info = dataset_info(cfg, n)
    if legacy:
        try:
            old_dsj = io.load_dataset_json(cfg["name"])
        except (OSError, ValueError):
            old_dsj = {}
        info.update(git_commit=old_dsj.get("git_commit", "unknown"), created=old_dsj.get("created"),
                    sim_code_sha256=None, provenance="legacy: config.json predates provenance; git_commit and "
                                                     "created are kept from the existing dataset.json")
    else:
        info.update(git_commit=prov["git_commit"], created=prov["created"], sim_code_sha256=prov["sim_code_sha256"])
    io.write_dataset_json(ds_dir, info)
    log(f"[{cfg['name']}] wrote meta.jsonl and dataset.json ({time.time() - t0:.0f}s)")
    return rows


# --------------------------------------------------------------------------- previews

def _label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 7 * len(text) + 6, 14], fill=(0, 0, 0))
    dr.text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def write_preview(name, k=16, fps=16, out_dir=None, log=print):
    """previews/<name>/grid.mp4 (first k clips tiled, fps 16 = 4x slow motion) and sheet.png
    (4 clips spanning the displacement range x frames 0, 16, 32, 63). Needs pillow; without it the previews
    are skipped with a log line, so a finished generation is not failed by a missing preview dependency."""
    try:
        from PIL import Image
    except ImportError:
        log("preview: pillow not installed, previews skipped (pip install -r requirements-sim.txt)")
        return None, None
    meta = io.load_meta(name)[:k]
    out_dir = out_dir or os.path.join(io.root(), "previews", name)
    os.makedirs(out_dir, exist_ok=True)
    cols = int(math.ceil(math.sqrt(len(meta))))
    rows_n = int(math.ceil(len(meta) / cols))
    clips = [np.asarray(io.load_clip(name, r["id"])) for r in meta]
    labels = [f"{r['id']} m={r['m']:.2f} mu={r['mu']:.2f}" for r in meta]
    grid = np.zeros((S.T, rows_n * S.H, cols * S.W, 3), np.uint8)
    for j, (c, lab) in enumerate(zip(clips, labels)):
        y, x = divmod(j, cols)
        for t in range(S.T):
            grid[t, y * S.H:(y + 1) * S.H, x * S.W:(x + 1) * S.W] = _label(c[t], lab)
    mp4 = os.path.join(out_dir, "grid.mp4")
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{grid.shape[2]}x{grid.shape[1]}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", mp4]
        subprocess.run(cmd, input=grid.tobytes(), check=True)
        log(f"preview: {mp4}")
    else:
        log("preview: ffmpeg not found, grid.mp4 skipped")
    order = np.argsort([r["disp_m"] for r in meta])
    pick = order[np.linspace(0, len(order) - 1, min(4, len(order))).round().astype(int)]
    frames = (0, 16, 32, S.T - 1)
    sheet = np.zeros((len(pick) * S.H, len(frames) * S.W, 3), np.uint8)
    for a, j in enumerate(pick):
        for b, t in enumerate(frames):
            sheet[a * S.H:(a + 1) * S.H, b * S.W:(b + 1) * S.W] = _label(clips[j][t], f"{labels[j]} f{t}")
    png = os.path.join(out_dir, "sheet.png")
    Image.fromarray(sheet).save(png)
    log(f"preview: {png}")
    return mp4, png


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenario", choices=("push", "slide", "poscheck"), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--base-seed", type=int, default=None,
                   help="default: push 0, slide 100000, poscheck 300000; datasets push_ood 200000, push_nuis 400000")
    p.add_argument("--workers", type=int, default=8, help=f"capped at PHYSPROBE_MAX_WORKERS={MAX_WORKERS}")
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--v0", type=float, default=1.2)
    p.add_argument("--m-range", type=float, nargs=2, default=(0.5, 2.0))
    p.add_argument("--mu-range", type=float, nargs=2, default=(0.2, 0.8))
    p.add_argument("--split", choices=("auto", "canonical", "smoke", "ood"), default="auto",
                   help="auto: smoke for datasets named smoke*, ood for *_ood, else canonical (seed %% 20)")
    p.add_argument("--preview", type=int, default=16, help="clips in previews/<dataset>/grid.mp4, 0 = none")
    p.add_argument("--overwrite", action="store_true", help="regenerate all ids and restamp provenance")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    _renice()
    cfg = config_from_args(a)
    rows = generate(cfg, a.n, workers=a.workers, overwrite=a.overwrite)
    if a.preview > 0:
        write_preview(a.dataset, a.preview)
    from physprobe.sim import stats
    print(json.dumps(stats.summarize(rows, stats.gen_times(a.dataset)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
