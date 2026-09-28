import gc

import torch

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


# Klein conditioning structure: 3 stacked Qwen3 hidden-layer slices.
# 12288 = 3 * 4096 (layers 9, 18, 27 per Klein TE setup in ComfyUI flux.py).
# 7680 = 3 * 2560 for the smaller Qwen3-4B Klein variant.
def _layer_slice_size(embed_dim: int) -> int:
    """Return the per-layer slice width given the total embed dim."""
    if embed_dim % 3 == 0:
        return embed_dim // 3
    # Unknown architecture — disable layer ops gracefully.
    return embed_dim


def _detect_active_end(meta: dict, seq_len: int, override: int) -> int:
    """Honest active-region detection. Falls back to seq_len, NOT 77."""
    if override > 0:
        return min(override, seq_len)
    attn_mask = meta.get("attention_mask", None)
    if attn_mask is not None and attn_mask.dim() >= 2:
        nonzero = attn_mask[0].nonzero()
        if len(nonzero) > 0:
            return int(nonzero[-1].item()) + 1
    return seq_len


def _resolve_device(name: str):
    if name == "auto":
        if HAS_COMFY:
            return mm.get_torch_device()
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Flux2KleinEnhancer:
    """Honest scalar/whitening operations on the active region of Klein
    conditioning, plus Klein-specific per-Qwen3-layer scaling."""

    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "active_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier on every active-token embedding. 1.0 = unchanged. The model was trained on Qwen3's natural distribution; values far from 1.0 push it off-distribution.",
                }),
                "per_token_whiten": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Amplifies per-token deviation from the sequence mean: (x - mean)*(1+w) + mean. >0 widens spread, <0 compresses. Was called 'contrast' in v1.",
                }),
                "norm_equalize": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend each token toward the per-sequence mean L2 norm. Flattens magnitude variance — fights Qwen3's natural emphasis. 0 = no effect.",
                }),
            },
            "optional": {
                "early_layer_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Klein-specific. Scale the first Qwen3 layer slice (low-level / structural features). Klein conditioning stacks 3 layers along the embed dim; this targets the first.",
                }),
                "mid_layer_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Klein-specific. Scale the middle Qwen3 layer slice (intermediate semantic features).",
                }),
                "late_layer_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Klein-specific. Scale the last Qwen3 layer slice (high-level / abstract semantic features).",
                }),
                "preserve_original": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Linear blend back the unmodified active region. 0.0 = full enhancement, 1.0 = no change.",
                }),
                "active_end_override": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Override the active-region end. 0 = auto-detect from attention_mask, falls back to full sequence length if mask missing.",
                }),
                "device": (devices, {"default": "auto"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = "conditioning/flux2klein"

    def enhance(self, conditioning, active_scale=1.0, per_token_whiten=0.0,
                norm_equalize=0.0, early_layer_scale=1.0, mid_layer_scale=1.0,
                late_layer_scale=1.0, preserve_original=0.0,
                active_end_override=0, device="auto", debug=False):

        if not conditioning:
            return (conditioning,)

        no_op = (
            active_scale == 1.0
            and per_token_whiten == 0.0
            and norm_equalize == 0.0
            and early_layer_scale == 1.0
            and mid_layer_scale == 1.0
            and late_layer_scale == 1.0
            and preserve_original == 0.0
        )
        if no_op:
            if debug:
                print("[Flux2KleinEnhancer] all params neutral, passing through")
            return (conditioning,)

        dev = _resolve_device(device)
        output = []

        for idx, (cond_tensor, meta) in enumerate(conditioning):
            original_dtype = cond_tensor.dtype
            cond = cond_tensor.to(dev, dtype=torch.float32)

            if cond.dim() != 3:
                output.append((cond_tensor, meta))
                continue

            seq_len, embed_dim = cond.shape[1], cond.shape[2]
            active_end = _detect_active_end(meta, seq_len, active_end_override)
            slice_w = _layer_slice_size(embed_dim)

            active = cond[:, :active_end, :].clone()
            original_active = active.clone()

            if debug:
                print(f"[Flux2KleinEnhancer] item {idx} | shape={tuple(cond.shape)} | "
                      f"active=[0:{active_end}] | layer_slice_width={slice_w}")

            # 1) per-token whitening (deviation amplification around seq mean).
            if per_token_whiten != 0.0:
                seq_mean = active.mean(dim=1, keepdim=True)
                active = seq_mean + (active - seq_mean) * (1.0 + per_token_whiten)

            # 2) per-token L2 norm equalization.
            if norm_equalize > 0.0:
                token_norms = active.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                target_norm = token_norms.mean()
                normalized = active / token_norms * target_norm
                active = active * (1.0 - norm_equalize) + normalized * norm_equalize

            # 3) Active-region scalar.
            if active_scale != 1.0:
                active = active * active_scale

            # 4) Klein-specific per-layer scale (only if embed dim is 3*N).
            if slice_w * 3 == embed_dim and (
                early_layer_scale != 1.0 or mid_layer_scale != 1.0 or late_layer_scale != 1.0
            ):
                if early_layer_scale != 1.0:
                    active[:, :, :slice_w] = active[:, :, :slice_w] * early_layer_scale
                if mid_layer_scale != 1.0:
                    active[:, :, slice_w:2 * slice_w] = active[:, :, slice_w:2 * slice_w] * mid_layer_scale
                if late_layer_scale != 1.0:
                    active[:, :, 2 * slice_w:] = active[:, :, 2 * slice_w:] * late_layer_scale

            # 5) Preserve original (linear blend back).
            if preserve_original > 0.0:
                active = active * (1.0 - preserve_original) + original_active * preserve_original

            result = cond.clone()
            result[:, :active_end, :] = active

            if debug:
                diff = (result - cond).abs()
                print(f"  output diff: mean={diff.mean().item():.6f} max={diff.max().item():.6f}")

            output.append((result.to("cpu", dtype=original_dtype), meta))
            del cond, active, original_active, result

        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return (output,)


class Flux2KleinDetailController:
    """Per-section embedding multiplier.

    HONEST MODE (recommended): pair with the Sectioned Encoder. The encoder
    emits per-token section ranges as conditioning metadata
    (`meta["klein_sections"]`); this node multiplies the active embeddings
    inside those exact ranges by user-supplied factors.

    FALLBACK MODE (legacy / not recommended): if no section metadata is
    present, the node falls back to fixed 25%/50%/25% slicing of the active
    region. Those boundaries are arbitrary — Qwen3 has no positional
    semantic role for tokens — and v1 of this node was effectively a placebo
    in this mode. Kept for back-compat with old workflows; switch to the
    Sectioned Encoder for meaningful section-aware scaling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "front_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for the FRONT section. With Sectioned Encoder upstream this is the actual front token range; otherwise it's the first 25% of active tokens (arbitrary).",
                }),
                "mid_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for the MID section.",
                }),
                "end_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for the END section.",
                }),
                "emphasis_start": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Custom emphasis range start (token index).",
                }),
                "emphasis_end": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Custom emphasis range end (0 = disabled).",
                }),
                "emphasis_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Multiplier applied inside [emphasis_start, emphasis_end).",
                }),
                "preserve_original": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Linear blend back the unmodified active region. 0 = full effect.",
                }),
                "device": (devices, {"default": "auto"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2klein"

    def control(self, conditioning, front_mult=1.0, mid_mult=1.0, end_mult=1.0,
                emphasis_start=0, emphasis_end=0, emphasis_mult=1.0,
                preserve_original=0.0, device="auto", debug=False):

        if not conditioning:
            return (conditioning,)

        no_op = (
            front_mult == 1.0 and mid_mult == 1.0 and end_mult == 1.0
            and (emphasis_end == 0 or emphasis_mult == 1.0)
            and preserve_original == 0.0
        )
        if no_op:
            return (conditioning,)

        dev = _resolve_device(device)
        output = []

        for idx, (cond_tensor, meta) in enumerate(conditioning):
            original_dtype = cond_tensor.dtype
            cond = cond_tensor.to(dev, dtype=torch.float32)
            if cond.dim() != 3:
                output.append((cond_tensor, meta))
                continue

            seq_len = cond.shape[1]
            active_end = _detect_active_end(meta, seq_len, 0)

            # Determine section ranges. PREFERRED: from sectioned encoder.
            sections = meta.get("klein_sections")
            if sections and all(k in sections for k in ("front", "mid", "end")):
                front_range = sections["front"]
                mid_range = sections["mid"]
                end_range = sections["end"]
                source = "klein_sections (encoder-emitted, real boundaries)"
            else:
                num = active_end
                f_end = int(num * 0.25)
                m_end = int(num * 0.75)
                front_range = (0, f_end)
                mid_range = (f_end, m_end)
                end_range = (m_end, num)
                source = "fixed 25/50/25 fallback (arbitrary — pair with Sectioned Encoder for honest section ranges)"

            if debug:
                print(f"[Flux2KleinDetailController] item {idx} | active=[0:{active_end}] | source={source}")
                print(f"  front={front_range} mid={mid_range} end={end_range}")

            active = cond[:, :active_end, :].clone()
            original_active = active.clone()

            def _scale(rng, mult):
                s, e = rng
                s = max(0, min(s, active_end))
                e = max(s, min(e, active_end))
                if mult == 1.0 or e <= s:
                    return
                active[:, s:e, :] = active[:, s:e, :] * mult

            _scale(front_range, front_mult)
            _scale(mid_range, mid_mult)
            _scale(end_range, end_mult)

            if emphasis_end > 0 and emphasis_mult != 1.0:
                _scale((emphasis_start, emphasis_end), emphasis_mult)

            if preserve_original > 0.0:
                active = active * (1.0 - preserve_original) + original_active * preserve_original

            result = cond.clone()
            result[:, :active_end, :] = active

            output.append((result.to("cpu", dtype=original_dtype), meta))

        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinDetailController": "FLUX.2 Klein Detail Controller",
}
