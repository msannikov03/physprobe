"""Simulation tests (compute, $SIM_PY): stopping distance, tilt, frame-0 identity, normal force, generate+validate,
resume / refusal / provenance. Run: $SIM_PY tests/test_sim.py   (or pytest tests/test_sim.py; skipped without mujoco)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import mujoco  # noqa: F401
except ImportError:
    import pytest
    pytest.skip("needs SIM_PY (mujoco)", allow_module_level=True)

from physprobe.sim import generate as gen  # noqa: E402
from physprobe.sim import scene as S  # noqa: E402
from test_io import tmp_root  # noqa: E402

_SIM = {}
QUIET = lambda *a: None  # noqa: E731


def _sim(render=False):
    if render not in _SIM:
        _SIM[render] = gen.Sim(render=render)
    return _SIM[render]


def test_stopping_distance():
    """Final displacement within 5 % of v0^2 / (2 mu g) for five (m, mu, v0) cases, push and slide."""
    for m, mu, v0 in [(0.5, 0.2, 2.0), (0.5, 0.8, 2.0), (1.0, 0.5, 1.0), (2.0, 0.8, 0.5), (1.3, 0.35, 1.2)]:
        r = _sim().run(m, mu, v0, render=False)
        d_th = v0 ** 2 / (2 * mu * S.G)
        assert r["disp_stop"] is not None, (m, mu, v0)
        err = r["disp_stop"] / d_th - 1
        assert abs(err) < 0.05, f"m={m} mu={mu} v0={v0}: d={r['disp_stop']:.4f} vs {d_th:.4f} ({err:+.2%})"
        t_th = v0 / (mu * S.G)
        assert abs(r["t_stop"] - t_th) < 0.01, f"t_stop {r['t_stop']:.4f} vs {t_th:.4f}"


def test_tilt_mu08():
    """At mu = 0.8 and v0 = 2 m/s (where soft friction cones made the cube hop) the cube slides flat: tilt and yaw
    computed from the quaternion at every substep stay < 5 deg, z stays at rest height, vz ~ 0."""
    z0 = S.rest_z()
    for m in (0.5, 2.0):
        r = _sim().run(m, 0.8, 2.0, render=False)
        tilt, yaw = S.tilt_yaw_deg(np.concatenate([r["quat"], r["qsub"]]))
        assert np.isfinite(tilt).all() and float(tilt.max()) < 5.0, (m, float(np.nanmax(tilt)))
        assert float(yaw.max()) < 5.0, (m, float(yaw.max()))
        assert np.abs(r["state"][:, 2] - z0).max() < 1e-4, (m, np.abs(r["state"][:, 2] - z0).max())
        assert np.abs(r["state"][:, 5]).max() < 1e-3, (m, np.abs(r["state"][:, 5]).max())
        assert gen.clip_meta(r, 0.0)["ok"]


def test_frame0_identical():
    """Frame 0 at m = 0.5 and m = 2.0 (different mu) is pixel-identical; later frames differ."""
    sim = _sim(render=True)
    a = sim.run(0.5, 0.2, 2.0)["clip"].copy()
    b = sim.run(2.0, 0.8, 0.5)["clip"].copy()
    assert np.array_equal(a[0], b[0]), int(np.abs(a[0].astype(int) - b[0].astype(int)).max())
    assert not np.array_equal(a[S.T - 1], b[S.T - 1])
    assert not np.array_equal(a[0], a[S.T - 1])


def test_normal_force():
    """Every row: fn = m g (rel 1e-4). Rows sliding for the whole interval: ft = mu m g (rel 1e-4). Rows after the
    stop: ft < 1e-3 m g. slip equals the substep mean of |v_xy| from state: with constant deceleration the 20
    substep speeds of row k are v_k - (v_k - v_{k+1}) j / 20, j = 0..19, mean (10.5 v_k + 9.5 v_{k+1}) / 20;
    after the stop slip ~ state speed ~ 0. Tolerance 1e-3 m/s."""
    for m, mu in [(0.5, 0.8), (2.0, 0.2), (1.0, 0.5)]:
        v0 = 1.0 / m
        r = _sim().run(m, mu, v0, render=False)
        c, st = r["contact"], r["state"]
        mg = m * S.G
        fn_err = np.abs(c[:, 0] / mg - 1).max()
        assert fn_err < 1e-4, f"m={m} mu={mu}: max |fn/mg - 1| = {fn_err:.2e}"
        v = np.hypot(st[:, 3], st[:, 4])
        sliding = np.nonzero(v[1:] > 0.05)[0]
        assert sliding.size > 0, (m, mu)
        ft_err = np.abs(c[sliding, 1] / (mu * mg) - 1).max()
        assert ft_err < 1e-4, f"m={m} mu={mu}: max |ft/(mu m g) - 1| = {ft_err:.2e} while sliding"
        slip_pred = (10.5 * v[sliding] + 9.5 * v[sliding + 1]) / 20
        slip_err = np.abs(c[sliding, 2] - slip_pred).max()
        assert slip_err < 1e-3, f"m={m} mu={mu}: sliding slip vs state speed max err {slip_err:.2e}"
        k_stop = int(np.ceil(r["t_stop"] * S.FPS)) + 1
        assert k_stop < S.T - 1, (m, mu, r["t_stop"])
        stopped = np.arange(k_stop, S.T)
        assert np.abs(c[stopped, 2] - v[stopped]).max() < 1e-3, (m, mu, c[stopped, 2].max())
        assert c[stopped, 1].max() < 1e-3 * mg, (m, mu, c[stopped, 1].max() / mg)


def test_slide_mass_invariant():
    """Slide with fixed v0: trajectory and penetration do not depend on m (no mass leak)."""
    a = _sim().run(0.5, 0.5, 1.2, render=False)
    b = _sim().run(2.0, 0.5, 1.2, render=False)
    assert np.allclose(a["state"], b["state"], atol=1e-9)
    assert abs(a["z_pen"] - b["z_pen"]) < 1e-9


def _cfg(*extra):
    return gen.config_from_args(gen.parse_args(["--scenario", "push", "--n", "3", "--dataset", "t_push", *extra]))


def _raises_exit(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e)
    raise AssertionError("expected SystemExit")


def test_generate_validate():
    """generate n = 3 into a temporary root with 2 spawn workers; io.validate has no problems; dataset.json has the
    v1.1 keys and provenance; meta and state agree; a rerun is a no-op that keeps the provenance; a deleted part
    record regenerates exactly that id."""
    with tmp_root():
        from physprobe import io
        cfg = _cfg()
        rows = gen.generate(cfg, 3, workers=2, log=QUIET)
        assert [r["id"] for r in rows] == ["000000", "000001", "000002"]
        rep = io.validate("t_push", verbose=False)
        assert not rep["problems"], rep["problems"]
        assert rep["ok_frac"] == 1.0
        dsj = io.load_dataset_json("t_push")
        for k in ("camera", "ranges", "J", "v0_fixed", "cube_half", "floor_friction", "mujoco_version", "git_commit",
                  "created", "friction_model", "sim_version", "base_seed", "split_mode", "g", "path_m", "rest_z_m",
                  "v_stop_mps", "cube_rgba", "floor_rgba", "frame0", "sim_code_sha256"):
            assert k in dsj, k
        assert dsj["sim_code_sha256"] == gen.sim_code_sha256()
        for k in ("m_per_px_floor", "origin_px", "pos", "fovy", "m_per_px_com", "origin_px_com", "m_per_px_top",
                  "origin_px_top"):
            assert k in dsj["camera"], k
        need = {"id", "scenario", "m", "mu", "J", "v0", "x0_m", "seed", "split", "in_frame", "disp_m", "moved_px",
                "frames_moving", "t_stop_s", "max_tilt_deg", "max_yaw_deg", "z_pen_max_m", "ok"}
        assert need <= set(rows[0]), need - set(rows[0])
        assert not any(k.startswith("_") for k in rows[0])
        m, mu, _ = gen.sample_params(1, (0.5, 2.0), (0.2, 0.8), "push")
        assert rows[1]["m"] == m and rows[1]["mu"] == mu and rows[1]["split"] == io.canonical_split(1)
        for r in rows:
            assert r["v0"] == cfg["J"] / r["m"]
            st = io.load_state("t_push", r["id"])
            assert np.allclose(st[0], [0, 0, dsj["rest_z_m"], r["v0"], 0, 0], atol=1e-6), st[0]
        mtimes = {i: os.path.getmtime(os.path.join(io.dataset_dir("t_push"), f"clip_{i}.npy")) for i in
                  ("000000", "000001", "000002")}
        again = gen.generate(cfg, 3, workers=2, log=QUIET)
        assert again == rows
        dsj2 = io.load_dataset_json("t_push")
        assert (dsj2["git_commit"], dsj2["created"]) == (dsj["git_commit"], dsj["created"])
        os.remove(os.path.join(io.dataset_dir("t_push"), ".parts", "000001.json"))
        log = []
        third = gen.generate(cfg, 3, workers=1, log=log.append)
        assert third == rows and any("1 to generate" in s for s in log), log
        new_mtimes = {i: os.path.getmtime(os.path.join(io.dataset_dir("t_push"), f"clip_{i}.npy")) for i in mtimes}
        assert [i for i in mtimes if new_mtimes[i] != mtimes[i]] == ["000001"], (mtimes, new_mtimes)


def test_generate_refusals():
    """A changed config is refused; changed sim code with clips left is refused (unless --overwrite); a legacy
    config (no provenance) is re-assembled without restamping but cannot be extended; parts from an interrupted
    --overwrite run with another config are not reused."""
    with tmp_root():
        from physprobe import io
        cfg = _cfg()
        gen.generate(cfg, 2, workers=1, log=QUIET)
        ds = io.dataset_dir("t_push")
        cfg_path = os.path.join(ds, ".parts", "config.json")
        msg = _raises_exit(lambda: gen.generate(_cfg("--J", "1.5"), 2, workers=1, log=QUIET))
        assert "differs" in msg, msg
        with open(cfg_path) as f:
            stored = json.load(f)
        stored["provenance"]["sim_code_sha256"] = "0" * 64
        with open(cfg_path, "w") as f:
            json.dump(stored, f)
        gen.generate(cfg, 2, workers=1, log=QUIET)          # nothing to do: allowed, provenance kept
        msg = _raises_exit(lambda: gen.generate(cfg, 3, workers=1, log=QUIET))
        assert "sim code changed" in msg, msg
        rows = gen.generate(cfg, 3, workers=1, overwrite=True, log=QUIET)
        assert len(rows) == 3 and io.load_dataset_json("t_push")["sim_code_sha256"] == gen.sim_code_sha256()
        # legacy: strip provenance and part tags, as in datasets generated before provenance existed
        with open(cfg_path) as f:
            stored = json.load(f)
        stored.pop("provenance")
        with open(cfg_path, "w") as f:
            json.dump(stored, f)
        for i in ("000000", "000001", "000002"):
            p = os.path.join(ds, ".parts", i + ".json")
            with open(p) as f:
                rec = json.load(f)
            rec.pop("_tag")
            with open(p, "w") as f:
                json.dump(rec, f)
        dsj = io.load_dataset_json("t_push")
        dsj["git_commit"], dsj["created"] = "abc1234", "2026-01-01T00:00:00+00:00"
        io.write_dataset_json(ds, dsj)
        assert gen.generate(cfg, 3, workers=1, log=QUIET) == rows
        dsj = io.load_dataset_json("t_push")
        assert dsj["git_commit"] == "abc1234" and dsj["sim_code_sha256"] is None and "legacy" in dsj["provenance"]
        msg = _raises_exit(lambda: gen.generate(cfg, 4, workers=1, log=QUIET))
        assert "no provenance" in msg, msg
        # interrupted --overwrite with a new config: only parts carrying the new tag count as done
        cfg2 = _cfg("--J", "1.5")
        stored2 = dict(cfg2, provenance=gen._provenance(cfg2, gen.sim_code_sha256()))
        with open(cfg_path, "w") as f:
            json.dump(stored2, f)
        log = []
        rows2 = gen.generate(cfg2, 3, workers=1, log=log.append)
        assert any("3 to generate" in s for s in log), log
        assert [r["v0"] for r in rows2] == [1.5 / r["m"] for r in rows2]


if __name__ == "__main__":
    fails = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
