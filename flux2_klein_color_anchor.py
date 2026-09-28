import math
import torch

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class Flux2KleinColorAnchor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Maximum correction strength. "
                        "0.3-0.6 is a good starting range. "
                        "Too high and you override the model's color decisions entirely."
                    ),
                }),
            },
            "optional": {
                "ramp_curve": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": (
                        "Controls the shape of the correction ramp. "
                        "Formula: progress^(1/curve). "
                        "1.0 = linear. "
                        ">1 = fast start, tapers off  (e.g. sqrt for curve=2). "
                        "<1 = slow start, aggressive late  (e.g. squared for curve=0.5). "
                        "For few-step schedules (4-8 steps) values of 2-4 work well "
                        "because they reach useful strength quickly."
                    ),
                }),
                "ref_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 63,
                    "tooltip": "Which reference latent to anchor colors from.",
                }),
                "channel_weights": (["uniform", "by_variance"], {
                    "default": "uniform",
                    "tooltip": (
                        "uniform: correct all channels equally. "
                        "by_variance: weight correction by how stable each channel's "
                        "mean is in the reference (low-variance channels trusted more)."
                    ),
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(
        self,
        model,
        conditioning,
        strength=0.5,
        ramp_curve=1.5,
        ref_index=0,
        channel_weights="uniform",
        debug=False,
    ):
        if strength == 0.0:
            return (model,)

        # ── Extract reference channel means ────────────────────────────────────
        ref_means = None
        ch_trust  = None

        for _, meta in conditioning:
            rl = meta.get("reference_latents", None)

            if rl is None or len(rl) == 0:
                mc = meta.get("model_conds", {})
                cl = mc.get("ref_latents", None)
                if cl is not None and hasattr(cl, "cond") and len(cl.cond) > 0:
                    rl = cl.cond

            if rl is not None and ref_index < len(rl):
                ref = rl[ref_index].float()               # [1, C, H, W]
                # Per-channel spatial mean — shape [1, C, 1, 1]
                ref_means = ref.mean(dim=(-2, -1), keepdim=True)

                if channel_weights == "by_variance":
                    # Channels whose spatial mean varies little are stable color carriers
                    # — trust them more. Use 1/(1+spatial_var) as the weight.
                    spatial_var = ref.var(dim=(-2, -1), keepdim=True)   # [1, C, 1, 1]
                    ch_trust = 1.0 / (1.0 + spatial_var)
                    ch_trust = ch_trust / ch_trust.max().clamp(min=1e-8)  # normalize to [0,1]
                break

        if ref_means is None:
            print("[ColorAnchor] No reference latent found in conditioning — node inactive.")
            return (model,)

        # ── Build closure (avoid mutable state leaking across runs) ───────────
        _ref_means   = ref_means
        _ch_trust    = ch_trust
        _strength    = strength
        _curve       = max(ramp_curve, 1e-3)
        _debug       = debug
        _state       = {
            "sigma_max":          None,
            "last_sigma_logged":  None,
            "step":               0,        # counts callback invocations this run
        }

        def _color_anchor_fn(args: dict) -> torch.Tensor:
            denoised = args["denoised"]   # [B, C, H, W] — model's x0 prediction
            sigma    = args["sigma"]

            # Scalar sigma (ComfyUI passes a tensor)
            try:
                s = sigma.max().item()
            except Exception:
                s = float(sigma)

            # ── Progress signal 1: sigma-based ────────────────────────────────
            # Record the first sigma as sigma_max so the range is normalised to
            # whatever the scheduler actually uses (not the theoretical [0, 1]).
            if _state["sigma_max"] is None or s > _state["sigma_max"]:
                # Reset when a new generation starts (sigma jumps back up)
                _state["sigma_max"] = s
                _state["step"]      = 0

            sigma_max      = _state["sigma_max"]
            sigma_progress = max(0.0, min(1.0,
                (sigma_max - s) / sigma_max if sigma_max > 1e-6 else 0.0))

            # ── Progress signal 2: step-count-based ───────────────────────────
            # Grows as  1 - 0.5^n  regardless of sigma range.
            # Step 1 → 0.50 | Step 2 → 0.75 | Step 3 → 0.875 | Step 4 → 0.9375
            # This ensures strong correction at the final step even in 4-step
            # schedules where sigma only spans 0.97→0.77 (sigma_progress ≤ 0.23).
            _state["step"] += 1
            step_progress = 1.0 - 0.5 ** _state["step"]

            # Take whichever signal is further along so neither fires too early
            # but both can push correction to be strong in their respective regime.
            progress  = max(sigma_progress, step_progress)
            curved    = progress ** (1.0 / _curve)
            effective = _strength * curved

            if effective < 1e-5:
                return denoised

            ref  = _ref_means.to(denoised.device, dtype=denoised.dtype)  # [1,C,1,1]
            cur  = denoised.mean(dim=(-2, -1), keepdim=True)              # [B,C,1,1]

            # How far has the model drifted from the reference color?
            correction = ref - cur   # [B,C,1,1]  (broadcasts ref over batch)

            # Optional per-channel trust weighting
            if _ch_trust is not None:
                trust = _ch_trust.to(denoised.device, dtype=denoised.dtype)
                correction = correction * trust

            # Add correction only to the spatial mean — deviation unchanged
            corrected = denoised + correction * effective

            if _debug and s != _state["last_sigma_logged"]:
                _state["last_sigma_logged"] = s
                mean_drift = (ref - cur).abs().mean().item()
                applied    = (correction * effective).abs().mean().item()
                print(
                    f"[ColorAnchor] step={_state['step']}  sigma={s:.4f}  "
                    f"sigma_prog={sigma_progress:.3f}  step_prog={step_progress:.3f}  "
                    f"progress={progress:.3f}  effective={effective:.3f}  "
                    f"mean_drift={mean_drift:.5f}  applied={applied:.5f}"
                )

            return corrected

        # ── Register the hook ─────────────────────────────────────────────────
        m = model.clone()
        m.model_options.setdefault("sampler_post_cfg_function", []).append(_color_anchor_fn)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinColorAnchor": Flux2KleinColorAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinColorAnchor": "FLUX.2 Klein Color Anchor",
}
