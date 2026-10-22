"""Frozen encoder registry (CONTRACT v1, features section). Imports: torch + transformers, no mujoco.

Every encoder maps uint8 clips (B, T, H, W, 3) to pooled features per mode. Preprocessing is manual:
no processor resize or crop. V-JEPA 2 sees native 256x256; DINOv2 gets a bilinear 256->224 resize. The Cosmos tokenizer encoder
(TorchScript, no transformers) sees native 256x256 in [-1, 1] with the last frame repeated to 1 + 8k frames.

dtype: extract.py defaults to bf16 for pretrained reps and fp32 for random-init reps (default_dtype): a random
network's between-clip variance is tiny, so bf16 rounding noise is of the same order as the signal.

Flatten orders of the sequence modes (row-major, documented in info["seq_shape"]; kept as generated):
  vjepa2_l*  clip_seq*   (slot, channel)              32 x 1024, slot k = frames 2k, 2k+1
  dinov2_b*  clip_seq    (frame, feature)             16 x 1536, frames 0, 4, ..., 60
  cosmos_cv8 clip_seq    (slot, channel)              9 x 16
  cosmos_cv8 clip_grid4  (channel, slot, row, col)    16 x 9 x 4 x 4  <- channel-major, unlike the others
"""
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"
DINOV2_ID = "facebook/dinov2-base"
COSMOS_CV8_ID = "nvidia/Cosmos-Tokenize1-CV8x8x8-720p"
REVISIONS = {
    VJEPA2_ID: "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
    DINOV2_ID: "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    COSMOS_CV8_ID: "b6af495317c76f287a4131e9299936b1533f5f9f",
}
RANDOM_SEED = 0
FP16_MAX = 65000.0


def vjepa_pool(tokens, n_t):
    """tokens (B, n_t*S, D) in (T, H, W) order -> (clip_mean (B, D), clip_seq (B, n_t, D)), float32."""
    b, n, d = tokens.shape
    slots = tokens.float().reshape(b, n_t, n // n_t, d).mean(2)
    return slots.mean(1), slots


def dino_frame_feat(last_hidden_state):
    """(N, 1+P, D) -> (N, 2D) float32: CLS || mean of patch tokens."""
    h = last_hidden_state.float()
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)


def default_dtype(name):
    """bf16 for pretrained reps, fp32 for random-init reps (see module doc)."""
    return torch.float32 if REGISTRY[name][1].get("random_init") else torch.bfloat16


def _out(x, mode):
    x = x.detach().float().cpu().numpy()
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{mode}: non-finite features")
    if mode.startswith(("clip_seq", "clip_grid")):
        if np.abs(x).max() > FP16_MAX:
            raise OverflowError(f"{mode}: |x| max {np.abs(x).max():.1f} does not fit float16")
        return x.astype(np.float16)
    return x.astype(np.float32)


class Encoder:
    """Base class. Subclasses set hf_id, all_modes, input_res and implement encode / encode_static."""
    hf_id = ""
    all_modes = ()
    input_res = 0

    def __init__(self, name, device="cuda", dtype=torch.bfloat16, random_init=False, config=None, modes=None):
        self.name = name
        self.device = torch.device(device)
        self.dtype = dtype
        self.random_init = random_init or config is not None
        self.modes = tuple(modes or self.all_modes)
        self.revision = REVISIONS.get(self.hf_id) if config is None else "tiny-config"
        if config is None:
            config = AutoConfig.from_pretrained(self.hf_id, revision=self.revision)
        self.config = config
        if self.random_init:
            torch.manual_seed(RANDOM_SEED)
            model = AutoModel.from_config(config, attn_implementation="sdpa")
        else:
            model = AutoModel.from_pretrained(self.hf_id, revision=self.revision, dtype=dtype,
                                              attn_implementation="sdpa")
            got = getattr(model.config, "_commit_hash", None)
            if got and got != self.revision:
                raise RuntimeError(f"{self.hf_id}: loaded revision {got} != pinned {self.revision}")
        self.model = model.to(self.device, dtype).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(3, 1, 1)

    def _norm(self, x_u8, size=None):
        """uint8 (..., H, W, 3) -> normalised (..., 3, H', W') in self.dtype; optional bilinear resize (4D input)."""
        x = x_u8.to(self.device, non_blocking=True).movedim(-1, -3).float().div_(255.0)
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
        return ((x - self.mean) / self.std).to(self.dtype)

    def base_info(self):
        init = (f"random (torch.manual_seed({RANDOM_SEED}); AutoModel.from_config)" if self.random_init
                else "pretrained")
        return dict(hf_id=self.hf_id, revision=self.revision, init=init, input_res=self.input_res,
                    dtype=str(self.dtype).replace("torch.", ""), normalize="imagenet mean/std")

    def encode(self, clips, modes):
        raise NotImplementedError

    def encode_static(self, frames):
        raise NotImplementedError

    def mode_info(self, mode):
        raise NotImplementedError


class VJEPA2(Encoder):
    """V-JEPA 2 ViT-L encoder (predictor skipped). Tokens are (T/2, 16, 16) in (T, H, W) order.

    Layer modes: the suffix-free modes read last_hidden_state (block 24 + the encoder's final LayerNorm), while
    _L08 / _L16 read hidden_states[8] / [16], the residual stream after that block WITHOUT the final LayerNorm.
    A layer comparison therefore mixes depth with the LayerNorm (info["layer_source"] says which). LayerNorm'd
    L08/L16 variants would be new modes and need a new extraction (deferred)."""
    hf_id = VJEPA2_ID
    input_res = 256
    all_modes = ("static", "clip_mean", "clip_seq", "clip_mean_L08", "clip_mean_L16", "clip_seq_L08", "clip_seq_L16")
    suffix_layer = {"": None, "_L08": 8, "_L16": 16}
    static_frames = 64

    @property
    def n_layers(self):
        return self.config.num_hidden_layers

    def _layer(self, suffix):
        return self.suffix_layer[suffix] or self.n_layers

    def _forward(self, clips, hidden):
        _, t, h, w, _ = clips.shape
        if h != self.config.crop_size or w != self.config.crop_size:
            raise ValueError(f"V-JEPA 2 expects {self.config.crop_size}x{self.config.crop_size}, got {h}x{w}")
        x = self._norm(clips)
        return self.model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=hidden), t

    @torch.inference_mode()
    def encode(self, clips, modes):
        modes = [m for m in modes if m != "static"]
        hidden = any(m.endswith(("_L08", "_L16")) for m in modes)
        out, t = self._forward(clips, hidden)
        n_t = t // self.config.tubelet_size
        res = {}
        for sfx in self.suffix_layer:
            if f"clip_mean{sfx}" not in modes and f"clip_seq{sfx}" not in modes:
                continue
            layer = self._layer(sfx)
            h = out.last_hidden_state if layer == self.n_layers else out.hidden_states[layer]
            mean, seq = vjepa_pool(h, n_t)
            res[f"clip_mean{sfx}"] = mean
            res[f"clip_seq{sfx}"] = seq.flatten(1)
        return {m: _out(res[m], m) for m in modes}

    @torch.inference_mode()
    def encode_static(self, frames):
        clips = frames[:, None].expand(-1, self.static_frames, -1, -1, -1)
        out, _ = self._forward(clips, False)
        return _out(out.last_hidden_state.float().mean(1), "static")

    def mode_info(self, mode):
        info = self.base_info()
        if mode == "static":
            layer, pooling, n_frames = self.n_layers, "mean over all tokens", self.static_frames
            info["frames"] = f"frame 0 repeated {self.static_frames}x"
        else:
            kind, sfx = ("clip_seq", mode[8:]) if mode.startswith("clip_seq") else ("clip_mean", mode[9:])
            layer, n_frames = self._layer(sfx), 64
            n_t = n_frames // self.config.tubelet_size
            if kind == "clip_mean":
                pooling = "mean over all tokens (T, H, W)"
            else:
                pooling = f"spatial mean per temporal slot, ({n_t}, {self.config.hidden_size}) flattened row-major"
                ts = self.config.tubelet_size
                info["seq_shape"] = [n_t, self.config.hidden_size]
                info["slot_frames"] = f"slot k covers frames {ts}k..{ts}k+{ts - 1}"
        info["layer_source"] = ("last_hidden_state (block 24 + final LayerNorm)" if layer == self.n_layers
                                else f"hidden_states[{layer}] (output of block {layer}, no final LayerNorm)")
        info.update(layer=layer, pooling=pooling, n_frames=n_frames)
        return info


class DINOv2(Encoder):
    """DINOv2 ViT-B/14 per frame. Frame feature = CLS || mean(patch tokens), 2 x 768 = 1536.

    Every forward pass sees exactly frame_chunk = 64 frames (the last chunk is padded by repeating its last
    frame, padding outputs are dropped), so a frame's feature does not depend on --batch or on which modes are
    in the pass (in bf16 the GEMM kernels, and so the rounding, change with the batch shape).

    clip_seq samples single frames 0, 4, ..., 60 (kept as generated). On push and slide frame 0 is the same
    in every clip, so slot 0 is constant across clips, frames 61-63 are never seen, and unlike V-JEPA slots
    and pixel bins the slots are not temporal means. Means over 16 bins of 4 frames would be a new mode."""
    hf_id = DINOV2_ID
    input_res = 224
    all_modes = ("static", "last", "clip_mean", "clip_seq")
    seq_frames = tuple(range(0, 64, 4))
    frame_chunk = 64

    @torch.inference_mode()
    def _frames(self, frames):
        """uint8 (N, H, W, 3) -> (N, 2D) float32 tensor, in fixed-size chunks of frame_chunk frames."""
        feats = []
        c = self.frame_chunk
        for i in range(0, frames.shape[0], c):
            x = self._norm(frames[i:i + c], size=(self.input_res, self.input_res))
            k = x.shape[0]
            if k < c:
                x = torch.cat([x, x[-1:].expand(c - k, -1, -1, -1)])
            feats.append(dino_frame_feat(self.model(pixel_values=x).last_hidden_state)[:k])
        return torch.cat(feats)

    @torch.inference_mode()
    def encode(self, clips, modes):
        modes = [m for m in modes if m != "static"]
        b, t = clips.shape[:2]
        if "clip_mean" in modes:
            idx = list(range(t))
        else:
            idx = sorted(set(([t - 1] if "last" in modes else []) + (list(self.seq_frames) if "clip_seq" in modes else [])))
        f = self._frames(clips[:, idx].reshape(-1, *clips.shape[2:])).reshape(b, len(idx), -1)
        pos = {k: j for j, k in enumerate(idx)}
        res = {}
        if "last" in modes:
            res["last"] = f[:, pos[t - 1]]
        if "clip_mean" in modes:
            res["clip_mean"] = f.mean(1)
        if "clip_seq" in modes:
            res["clip_seq"] = f[:, [pos[k] for k in self.seq_frames]].flatten(1)
        return {m: _out(res[m], m) for m in modes}

    def encode_static(self, frames):
        return _out(self._frames(frames), "static")

    def mode_info(self, mode):
        info = self.base_info()
        d = 2 * self.config.hidden_size
        info["resize"] = f"bilinear antialias 256->{self.input_res}, no crop"
        info["frame_batch"] = f"{self.frame_chunk} frames per forward (padded), independent of --batch"
        info["layer"] = self.config.num_hidden_layers
        info["layer_source"] = "last_hidden_state (final LayerNorm)"
        frame = "[CLS || mean(patch tokens)]"
        if mode == "static":
            info.update(pooling=f"frame 0 {frame}", n_frames=1, frame_idx=[0])
        elif mode == "last":
            info.update(pooling=f"frame 63 {frame}", n_frames=1, frame_idx=[63])
        elif mode == "clip_mean":
            info.update(pooling=f"mean over 64 frames of {frame}", n_frames=64)
        else:
            info.update(pooling=f"{len(self.seq_frames)} frames x {frame}, ({len(self.seq_frames)}, {d}) flattened row-major",
                        n_frames=len(self.seq_frames), frame_idx=list(self.seq_frames),
                        seq_shape=[len(self.seq_frames), d])
        return info


class CosmosCV8(Encoder):
    """NVIDIA Cosmos-Tokenize1 CV8x8x8 continuous video tokenizer, encoder.jit only (TorchScript, causal).
    Input (B, 3, 1+8k, H, W) in [-1, 1]; latent (B, 16, 1+k, H/8, W/8). For 64 frames: last frame repeated once
    -> 65 frames -> latent (B, 16, 9, 32, 32). Slot 0 encodes frame 0 alone; slot j >= 1 ends at frame 8j
    (frame 64 is a copy of frame 63); temporal convs are causal, so slot j sees frames <= 8j.

    Flatten orders (kept as generated): clip_seq is (slot, channel) = z.mean((3, 4)).transpose(1, 2), 9 x 16;
    clip_grid4 is (channel, slot, row, col) = adaptive_avg_pool3d(z, (9, 4, 4)), 16 x 9 x 4 x 4. Per-slot
    slicing of clip_grid4 must reshape to (16, 9, 4, 4) and index axis 1. `model` (tests) replaces the download."""
    hf_id = COSMOS_CV8_ID
    input_res = 256
    all_modes = ("clip_mean", "clip_seq", "clip_grid4")
    t_stride = 8
    grid = 4

    def __init__(self, name, device="cuda", dtype=torch.bfloat16, modes=None, model=None):
        self.name = name
        self.device = torch.device(device)
        self.dtype = dtype
        self.random_init = False
        self.modes = tuple(modes or self.all_modes)
        self.revision = REVISIONS[self.hf_id]
        if model is None:
            path = hf_hub_download(self.hf_id, "encoder.jit", revision=self.revision)
            model = torch.jit.load(path, map_location=self.device)
        self.model = model.to(self.device).eval().to(dtype)

    def base_info(self):
        return dict(hf_id=self.hf_id, revision=self.revision, file="encoder.jit (TorchScript)", init="pretrained",
                    input_res=self.input_res, dtype=str(self.dtype).replace("torch.", ""),
                    normalize="x/127.5 - 1 (range [-1, 1])", trained_res="720p and above (we feed 256x256)")

    def _pad_t(self, t):
        return 1 + self.t_stride * -(-(t - 1) // self.t_stride)

    def _latent(self, clips):
        """uint8 (B, T, H, W, 3) -> float32 latent (B, 16, 1+k, H/8, W/8)."""
        _, t, h, w, _ = clips.shape
        if h != self.input_res or w != self.input_res:
            raise ValueError(f"Cosmos CV8 path expects {self.input_res}x{self.input_res}, got {h}x{w}")
        x = clips.to(self.device, non_blocking=True).permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1.0)
        pad = self._pad_t(t) - t
        if pad:
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        out = self.model(x.to(self.dtype))
        return (out[0] if isinstance(out, (tuple, list)) else out).float()

    @torch.inference_mode()
    def encode(self, clips, modes):
        z = self._latent(clips)
        res = {}
        if "clip_mean" in modes:
            res["clip_mean"] = z.mean((2, 3, 4))
        if "clip_seq" in modes:
            res["clip_seq"] = z.mean((3, 4)).transpose(1, 2).flatten(1)
        if "clip_grid4" in modes:
            res["clip_grid4"] = F.adaptive_avg_pool3d(z, (z.shape[2], self.grid, self.grid)).flatten(1)
        return {m: _out(res[m], m) for m in modes}

    def mode_info(self, mode):
        info = self.base_info()
        n_t, c, g = 1 + 64 // self.t_stride, 16, self.grid
        info.update(n_frames=64, frames_fed=f"64 frames + frame 63 repeated once = {self._pad_t(64)}",
                    latent_shape=[c, n_t, self.input_res // 8, self.input_res // 8], layer="encoder output",
                    layer_source="encoder output latent",
                    slot_frames="slot 0 = frame 0 alone; slot j>=1 = frames 8j-7..8j (causal: sees frames <= 8j)")
        if mode == "clip_mean":
            info["pooling"] = "mean over latent (T, H, W)"
        elif mode == "clip_seq":
            info.update(pooling=f"spatial mean per latent slot, ({n_t}, {c}) flattened row-major", seq_shape=[n_t, c])
        else:
            info.update(pooling=f"adaptive avg pool of each latent frame to {g}x{g}, ({c}, {n_t}, {g}, {g}) flattened "
                        "row-major (channel, slot, row, col)", seq_shape=[c, n_t, g, g])
        return info


REGISTRY = {
    "vjepa2_l": (VJEPA2, dict()),
    "vjepa2_l_random": (VJEPA2, dict(random_init=True, modes=("static", "clip_mean", "clip_seq"))),
    "dinov2_b": (DINOv2, dict()),
    "dinov2_b_random": (DINOv2, dict(random_init=True)),
    "cosmos_cv8": (CosmosCV8, dict()),
}


def get(name, **kw):
    """Build a registered encoder: get('vjepa2_l', device='cuda', dtype=torch.bfloat16)."""
    cls, defaults = REGISTRY[name]
    return cls(name, **{**defaults, **kw})
