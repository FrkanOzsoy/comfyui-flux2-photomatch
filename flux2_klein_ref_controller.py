"""
FLUX.2 Klein Reference Latent Controller v2.0
"""

import torch
import gc

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


def _spatial_token_weights(num_tokens, ref_latent, mode, fade_strength, device):
    if mode == "none" or ref_latent is None:
        return None

    _, _, H, W = ref_latent.shape
    patch_size = 2
    h_p = (H + patch_size // 2) // patch_size
    w_p = (W + patch_size // 2) // patch_size

    y = torch.linspace(0.0, 1.0, h_p, device=device)
    x = torch.linspace(0.0, 1.0, w_p, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    if mode == "center_out":
        dist = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        dist = dist / dist.max().clamp(min=1e-8)
        weights = 1.0 - dist * fade_strength
    elif mode == "edges_out":
        dist = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        dist = dist / dist.max().clamp(min=1e-8)
        weights = (1.0 - fade_strength) + dist * fade_strength
    elif mode == "top_down":
        weights = 1.0 - yy * fade_strength
    elif mode == "left_right":
        weights = 1.0 - xx * fade_strength
    else:
        return None

    weights = weights.clamp(0.0, 5.0).flatten()

    n = weights.shape[0]
    if n > num_tokens:
        weights = weights[:num_tokens]
    elif n < num_tokens:
        pad = torch.ones(num_tokens - n, device=device)
        weights = torch.cat([weights, pad])

    return weights


class Flux2KleinRefLatentController:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.05,
                }),
                "reference_index": ("INT", {
                    "default": 0, "min": 0, "max": 7,
                }),
            },
            "optional": {
                "spatial_fade": (
                    ["none", "center_out", "edges_out", "top_down", "left_right"],
                    {"default": "none"},
                ),
                "spatial_fade_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2klein"

    def control(self, model, conditioning, strength=1.0, reference_index=0,
                spatial_fade="none", spatial_fade_strength=0.5, debug=False):

        m = model.clone()

        ref_latent = None
        if conditioning and spatial_fade != "none":
            for _, meta in conditioning:
                rl = meta.get("reference_latents", None)
                if rl and reference_index < len(rl):
                    ref_latent = rl[reference_index]
                    break

        _strength   = strength
        _ref_idx    = reference_index
        _fade       = spatial_fade
        _fade_s     = spatial_fade_strength
        _ref_latent = ref_latent
        _debug      = debug

        def ref_weight_patch(q, k, v, extra_options={}, **kwargs):
            ref_tokens = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens or _ref_idx >= len(ref_tokens):
                return {}

            total_ref   = sum(ref_tokens)
            tok_start   = sum(ref_tokens[:_ref_idx])
            tok_end     = tok_start + ref_tokens[_ref_idx]
            num_ref_tok = ref_tokens[_ref_idx]

            seq_start = -total_ref + tok_start
            seq_end   = -total_ref + tok_end

            if _fade != "none" and _ref_latent is not None:
                token_w = _spatial_token_weights(
                    num_ref_tok, _ref_latent, _fade, _fade_s, k.device
                )
                if token_w is not None:
                    scale = (_strength * token_w).view(1, 1, -1, 1).to(k.dtype)
                else:
                    scale = _strength
            else:
                scale = _strength

            seq_end_idx = None if seq_end == 0 else seq_end

            k = k.clone()
            v = v.clone()
            k[:, :, seq_start:seq_end_idx, :] = k[:, :, seq_start:seq_end_idx, :] * scale
            v[:, :, seq_start:seq_end_idx, :] = v[:, :, seq_start:seq_end_idx, :] * scale

            if _debug:
                block_idx = extra_options.get("block_index", "?")
                print(
                    f"[RefLatentController] block={block_idx}  "
                    f"ref_index={_ref_idx}  "
                    f"tokens=[{seq_start}:{seq_end}]  "
                    f"strength={_strength:.3f}"
                )
            return {"q": q, "k": k, "v": v}

        m.set_model_attn1_patch(ref_weight_patch)
        return (m, conditioning)


class Flux2KleinTextRefBalance:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "balance": ("FLOAT", {
                    "default": 0.500, "min": 0.000, "max": 1.000, "step": 0.001,
                }),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "balance_streams"
    CATEGORY = "conditioning/flux2klein"

    def balance_streams(self, model, conditioning, balance=0.5, debug=False):
        m = model.clone()

        if balance <= 0.5:
            text_scale = balance * 2.0
            ref_scale  = 1.0
        else:
            text_scale = 1.0
            ref_scale  = (1.0 - balance) * 2.0

        if debug:
            print(
                f"[TextRefBalance] balance={balance:.2f}  "
                f"text_scale={text_scale:.3f}  ref_scale={ref_scale:.3f}"
            )

        _text_s = text_scale
        _ref_s  = ref_scale
        _debug  = debug

        def balance_patch(q, k, v, extra_options={}, **kwargs):
            img_slice  = extra_options.get("img_slice", None)
            ref_tokens = extra_options.get("reference_image_num_tokens", [])

            if img_slice is None and not ref_tokens:
                return {}

            k = k.clone()
            v = v.clone()

            if img_slice is not None and _text_s != 1.0:
                txt_end = img_slice[0]
                k[:, :, :txt_end, :] *= _text_s
                v[:, :, :txt_end, :] *= _text_s

            if ref_tokens and _ref_s != 1.0:
                total_ref = sum(ref_tokens)
                k[:, :, -total_ref:, :] *= _ref_s
                v[:, :, -total_ref:, :] *= _ref_s

            if _debug:
                block_idx = extra_options.get("block_index", "?")
                print(
                    f"[TextRefBalance] block={block_idx}  "
                    f"txt_scale={_text_s:.3f}  ref_scale={_ref_s:.3f}"
                )
            return {"q": q, "k": k, "v": v}

        m.set_model_attn1_patch(balance_patch)
        return (m, conditioning)


class Flux2KleinRefLatentWeight:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_index": ("INT", {
                    "default": 0, "min": 0, "max": 7,
                }),
                "weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "conditioning/flux2klein"

    def execute(self, model, reference_index, weight):
        m = model.clone()

        _ref_idx = reference_index
        _weight  = weight

        def ref_weight_patch(q, k, v, extra_options={}, **kwargs):
            ref_tokens = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens or _ref_idx >= len(ref_tokens):
                return {}

            total_ref = sum(ref_tokens)
            tok_start = sum(ref_tokens[:_ref_idx])
            tok_end   = tok_start + ref_tokens[_ref_idx]

            seq_start = -total_ref + tok_start
            seq_end   = -total_ref + tok_end

            seq_end_idx = None if seq_end == 0 else seq_end

            k = k.clone()
            v = v.clone()
            k[:, :, seq_start:seq_end_idx, :] *= _weight
            v[:, :, seq_start:seq_end_idx, :] *= _weight

            return {"q": q, "k": k, "v": v}

        m.set_model_attn1_patch(ref_weight_patch)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinRefLatentController":  Flux2KleinRefLatentController,
    "Flux2KleinTextRefBalance":       Flux2KleinTextRefBalance,
    "Flux2KleinRefLatentWeight":      Flux2KleinRefLatentWeight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinRefLatentController": "FLUX.2 Klein Ref Latent Controller",
    "Flux2KleinTextRefBalance":      "FLUX.2 Klein Text/Ref Balance",
    "Flux2KleinRefLatentWeight":     "FLUX.2 Klein Ref Latent Weight",
}
