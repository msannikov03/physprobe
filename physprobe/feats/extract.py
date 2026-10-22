"""Extract frozen-encoder features into features/<dataset>/<rep>/<mode>.npz (CONTRACT v1).

python -m physprobe.feats.extract --dataset D --rep R [--modes M ...] [--batch 4] [--limit N]

Order follows meta.jsonl. Resumable: a mode is skipped when its file already holds exactly the requested ids
AND its info has the current dtype, model revision and code_hash (sha1 of encoders.py + extract.py); otherwise
it is re-extracted (--ignore-code-hash relaxes only the code_hash part, e.g. after a comment-only edit).
Finished chunks of clips are kept in features/<D>/<R>/.partial/ until every requested mode is merged and saved;
each chunk carries the same (dtype, revision, code_hash) tag and chunks with another tag are ignored.
static encodes each distinct frame 0 once (sha1 of the frame) and replicates it.
--dtype auto (default) = bf16 for pretrained reps, fp32 for *_random; fp32 also turns TF32 off.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import transformers

from physprobe import io, paths
from physprobe.feats import encoders

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
TAG_KEYS = ("dtype", "revision", "code_hash")


def code_hash():
    h = hashlib.sha1()
    d = os.path.dirname(os.path.abspath(__file__))
    for f in ("encoders.py", "extract.py"):
        with open(os.path.join(d, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def rep_dir(dataset, rep):
    return os.path.dirname(paths.features_path(dataset, rep, "x"))


def partial_dir(dataset, rep):
    return os.path.join(rep_dir(dataset, rep), ".partial")


def _savez_atomic(path, **arrays):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_tag(enc):
    """(dtype, revision, code_hash) of this run; stored in feature info and in every partial chunk."""
    return dict(dtype=enc.base_info()["dtype"], revision=enc.revision, code_hash=code_hash())


def _read_partials(pdir, tag=None, log=None):
    """-> list of (path, ids, {mode: X}, timing dict). With `tag`, chunks whose timing["tag"] differs (or that
    have none, i.e. were written before tagging) are skipped and left on disk (--overwrite removes them)."""
    out, skipped = [], 0
    for p in sorted(glob.glob(os.path.join(pdir, "chunk_*.npz"))):
        with np.load(p) as z:
            timing = json.loads(str(z["timing"]))
            if tag is not None and timing.get("tag") != tag:
                skipped += 1
                continue
            ids = [str(i) for i in z["ids"]]
            xs = {k[3:]: np.asarray(z[k]) for k in z.files if k.startswith("X__")}
        out.append((p, ids, xs, timing))
    if skipped and log:
        log(f"ignored {skipped} partial chunk(s) in {pdir} with another dtype/revision/code_hash tag")
    return out


def _existing_ok(dataset, rep, mode, ids, tag=None, ignore_code_hash=False):
    """True when features/<dataset>/<rep>/<mode>.npz holds exactly `ids` and (with `tag`) its info matches the
    tag's dtype, revision and (unless ignore_code_hash) code_hash. -> (ok, reason)"""
    p = paths.features_path(dataset, rep, mode)
    if not os.path.exists(p):
        return False, "missing"
    with np.load(p) as z:
        if [str(i) for i in z["ids"]] != list(ids):
            return False, "different ids"
        info = json.loads(str(z["info"]))
    for k in (tag or {}):
        if k == "code_hash" and ignore_code_hash:
            continue
        if info.get(k) != tag[k]:
            return False, f"{k} {info.get(k)} != {tag[k]}"
    return True, "complete"


def _load_batch(dataset, bids):
    return torch.from_numpy(np.stack([np.asarray(io.load_clip(dataset, i)) for i in bids]))


def frame0_hashes(dataset, ids):
    return [hashlib.sha1(np.ascontiguousarray(io.load_clip(dataset, i)[0]).tobytes()).hexdigest() for i in ids]


def fp32_exact():
    """Turn TF32 off for fp32 matmuls and cuDNN convs (new fp32_precision API when present: mixing it with the
    legacy allow_tf32 flags raises in torch >= 2.9)."""
    mm = torch.backends.cuda.matmul
    if hasattr(mm, "fp32_precision"):
        mm.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        mm.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _tf32_info():
    mm = torch.backends.cuda.matmul
    if hasattr(mm, "fp32_precision"):
        return dict(matmul=mm.fp32_precision, cudnn_conv=torch.backends.cudnn.conv.fp32_precision)
    return dict(matmul="tf32" if mm.allow_tf32 else "ieee",
                cudnn_conv="tf32" if torch.backends.cudnn.allow_tf32 else "ieee")


def _env_info(enc, batch):
    dev = torch.cuda.get_device_name(enc.device) if enc.device.type == "cuda" else platform.processor() or "cpu"
    return dict(batch=batch, device=dev, torch=torch.__version__, transformers=transformers.__version__,
                code_hash=code_hash(), fp32_precision=_tf32_info())


def _peak_gib(enc):
    if enc.device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(enc.device) / 2 ** 30, 2)


def _fmt_eta(sec):
    return f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.0f} s"


def run_static(dataset, enc, ids, batch, log):
    t0 = time.time()
    hashes = frame0_hashes(dataset, ids)
    uniq = list(dict.fromkeys(hashes))
    rep_id = {h: ids[hashes.index(h)] for h in uniq}
    rows = {}
    if enc.device.type == "cuda":
        torch.cuda.synchronize(enc.device)
    t1 = time.time()
    for k in range(0, len(uniq), batch):
        hs = uniq[k:k + batch]
        frames = torch.from_numpy(np.stack([np.asarray(io.load_clip(dataset, rep_id[h])[0]) for h in hs]))
        X = enc.encode_static(frames)
        rows.update(zip(hs, X))
    gpu = time.time() - t1
    X = np.stack([rows[h] for h in hashes])
    wall = time.time() - t0
    info = enc.mode_info("static")
    info.update(_env_info(enc, batch), n_unique_frame0=len(uniq), sec_per_clip=round(wall / len(ids), 5),
                sec_per_unique_frame0=round(gpu / len(uniq), 5), peak_gpu_mem_gib=_peak_gib(enc))
    p = io.save_features(dataset, enc.name, "static", ids, X, info)
    log(f"[{dataset}/{enc.name}] static: {len(uniq)} unique frame 0 for {len(ids)} clips, "
        f"{info['sec_per_unique_frame0']:.3f} s/unique frame, shape {X.shape} -> {p}")


def run_clips(dataset, enc, ids, modes, batch, chunk, keep_partial, log):
    pdir = partial_dir(dataset, enc.name)
    tag = run_tag(enc)
    parts = _read_partials(pdir, tag, log)
    done = {m: set() for m in modes}
    for _, pids, xs, _ in parts:
        for m in modes:
            if m in xs:
                done[m].update(pids)
    work = [i for i in ids if any(i not in done[m] for m in modes)]
    log(f"[{dataset}/{enc.name}] modes {modes}: {len(ids) - len(work)}/{len(ids)} clips found in partials, "
        f"{len(work)} to encode, batch {batch}")
    if work:
        _encode_work(dataset, enc, work, modes, batch, chunk, pdir, log, tag)
        parts = _read_partials(pdir, tag)
    tot_wall = sum(t["wall"] for *_, t in parts)
    tot_gpu = sum(t["gpu"] for *_, t in parts)
    tot_n = sum(t["n"] for *_, t in parts)
    peak = max([t.get("peak_gpu_mem_gib") or 0 for *_, t in parts] or [0]) or None
    for m in modes:
        rows = {}
        for _, pids, xs, _ in parts:
            if m in xs:
                rows.update(zip(pids, xs[m]))
        missing = [i for i in ids if i not in rows]
        if missing:
            raise RuntimeError(f"{m}: {len(missing)} ids missing after encoding, e.g. {missing[:3]}")
        X = np.stack([rows[i] for i in ids])
        info = enc.mode_info(m)
        info.update(_env_info(enc, batch), sec_per_clip=round(tot_wall / tot_n, 5),
                    gpu_sec_per_clip=round(tot_gpu / tot_n, 5), modes_in_pass=list(modes),
                    peak_gpu_mem_gib=peak)
        p = io.save_features(dataset, enc.name, m, ids, X, info)
        log(f"[{dataset}/{enc.name}] {m}: shape {X.shape} {X.dtype} -> {p}")
    if not keep_partial:
        for p, *_ in parts:
            os.remove(p)
        if os.path.isdir(pdir) and not os.listdir(pdir):
            os.rmdir(pdir)


def _encode_work(dataset, enc, work, modes, batch, chunk, pdir, log, tag):
    batches = [work[k:k + batch] for k in range(0, len(work), batch)]
    acc_ids, acc_x = [], collections.defaultdict(list)
    acc_wall = acc_gpu = 0.0
    n_done, t_start, t_last_log = 0, time.time(), 0.0
    cuda = enc.device.type == "cuda"
    with ThreadPoolExecutor(2) as pool:
        queue = collections.deque()
        it = iter(batches)
        for b in [next(it, None), next(it, None)]:
            if b:
                queue.append((b, pool.submit(_load_batch, dataset, b)))
        t_prev = time.time()
        while queue:
            bids, fut = queue.popleft()
            clips = fut.result()
            nxt = next(it, None)
            if nxt:
                queue.append((nxt, pool.submit(_load_batch, dataset, nxt)))
            if cuda:
                torch.cuda.synchronize(enc.device)
            tg = time.time()
            feats = enc.encode(clips, modes)
            acc_gpu += time.time() - tg
            acc_ids += bids
            for m in modes:
                acc_x[m].append(feats[m])
            now = time.time()
            acc_wall += now - t_prev
            t_prev = now
            n_done += len(bids)
            if len(acc_ids) >= chunk or not queue:
                timing = dict(n=len(acc_ids), wall=acc_wall, gpu=acc_gpu, peak_gpu_mem_gib=_peak_gib(enc), tag=tag)
                arrays = {f"X__{m}": np.concatenate(acc_x[m]) for m in modes}
                name = f"chunk_{acc_ids[0]}_{acc_ids[-1]}_{os.getpid()}_{int(now * 1000)}.npz"
                _savez_atomic(os.path.join(pdir, name), ids=np.asarray(acc_ids, dtype="<U16"),
                              timing=np.array(json.dumps(timing)), **arrays)
                acc_ids, acc_x = [], collections.defaultdict(list)
                acc_wall = acc_gpu = 0.0
            if now - t_last_log > 10 or not queue:
                rate = (now - t_start) / n_done
                log(f"[{dataset}/{enc.name}] {n_done}/{len(work)} clips, {rate:.3f} s/clip, "
                    f"ETA {_fmt_eta(rate * (len(work) - n_done))}, peak {_peak_gib(enc)} GiB")
                t_last_log = now


def run(dataset, enc, modes=None, batch=4, limit=None, chunk=64, overwrite=False, keep_partial=False, log=None,
        ignore_code_hash=False):
    """Extract `modes` (default: all of enc.modes) for the first `limit` clips of `dataset` in meta order.
    -> list of modes (re-)extracted."""
    log = log or (lambda s: print(s, flush=True))
    ids = [r["id"] for r in io.load_meta(dataset)]
    if limit:
        ids = ids[:limit]
    modes = list(modes or enc.modes)
    bad = [m for m in modes if m not in enc.modes]
    if bad:
        raise ValueError(f"{enc.name}: unknown modes {bad}; available {list(enc.modes)}")
    tag = run_tag(enc)
    todo = []
    for m in modes:
        if overwrite:
            ok, why = False, "--overwrite"
        else:
            ok, why = _existing_ok(dataset, enc.name, m, ids, tag, ignore_code_hash)
        if not ok:
            todo.append(m)
            if why not in ("missing", "--overwrite"):
                log(f"[{dataset}/{enc.name}] {m}: existing file is stale ({why}), re-extracting")
    skipped = [m for m in modes if m not in todo]
    if skipped:
        log(f"[{dataset}/{enc.name}] already complete, skipped: {skipped}")
    if overwrite and os.path.isdir(partial_dir(dataset, enc.name)):
        for p in glob.glob(os.path.join(partial_dir(dataset, enc.name), "chunk_*.npz")):
            os.remove(p)
    if "static" in todo:
        run_static(dataset, enc, ids, batch, log)
    clip_modes = [m for m in todo if m != "static"]
    if clip_modes:
        run_clips(dataset, enc, ids, clip_modes, batch, chunk, keep_partial, log)
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rep", required=True, choices=sorted(encoders.REGISTRY))
    ap.add_argument("--modes", nargs="*", default=None, help="default: all modes of the rep")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=64, help="clips per partial file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto", choices=["auto"] + sorted(DTYPES),
                    help="auto: bf16 for pretrained reps, fp32 for *_random (fp32 also disables TF32)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-partial", action="store_true")
    ap.add_argument("--ignore-code-hash", action="store_true",
                    help="treat files with matching ids, dtype and revision as complete despite another code_hash")
    a = ap.parse_args(argv)
    t0 = time.time()
    dtype = encoders.default_dtype(a.rep) if a.dtype == "auto" else DTYPES[a.dtype]
    if dtype == torch.float32:
        fp32_exact()
    print(f"[{a.dataset}/{a.rep}] loading encoder on {a.device} ({dtype})", flush=True)
    enc = encoders.get(a.rep, device=a.device, dtype=dtype)
    print(f"[{a.dataset}/{a.rep}] encoder ready in {time.time() - t0:.1f} s, modes {list(enc.modes)}", flush=True)
    run(a.dataset, enc, a.modes, a.batch, a.limit, a.chunk, a.overwrite, a.keep_partial,
        ignore_code_hash=a.ignore_code_hash)
    print(f"[{a.dataset}/{a.rep}] done in {_fmt_eta(time.time() - t0)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
