"""Identity Guidance — Sampling Loop Correction."""

import torch
import torch.nn.functional as F


class IdentityGuidance:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "identity_latent": ("LATENT", {
                    "tooltip": "VAE-encoded reference image at full resolution.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How hard to pull toward the reference each step. 0.3 = move 30% of the distance.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "When to start correcting. 0.0 = beginning of denoising.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "When to stop correcting. 0.8 = last 20% runs freely for texture refinement.",
                }),
                "mode": (["adaptive", "direct", "channel_match"], {
                    "default": "adaptive",
                    "tooltip": "adaptive: pulls only where prediction resembles reference. direct: pulls everywhere equally. channel_match: matches color/feature statistics without copying spatial content.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, identity_latent, strength=0.3,
              start_percent=0.0, end_percent=0.8, mode="adaptive"):
        m = model.clone()
        ref = identity_latent["samples"]

        _ref = ref
        _strength = strength
        _start = start_percent
        _end = end_percent
        _mode = mode

        def post_cfg_fn(args):
            denoised = args["denoised"]
            sigma = args["sigma"]

            s_now = float(sigma.flatten()[0])
            progress = max(0.0, min(1.0, 1.0 - s_now))

            if progress < _start or progress > _end:
                return denoised

            ref_resized = _ref.to(device=denoised.device, dtype=denoised.dtype)
            if ref_resized.shape[0] != denoised.shape[0]:
                ref_resized = ref_resized[:1].expand(denoised.shape[0], -1, -1, -1)
            if ref_resized.shape[2:] != denoised.shape[2:]:
                ref_resized = F.interpolate(
                    ref_resized, size=denoised.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            if ref_resized.shape[1] != denoised.shape[1]:
                if ref_resized.shape[1] > denoised.shape[1]:
                    ref_resized = ref_resized[:, :denoised.shape[1]]
                else:
                    ref_resized = F.pad(ref_resized, (0, 0, 0, 0,
                        0, denoised.shape[1] - ref_resized.shape[1]))

            if _mode == "direct":
                correction = ref_resized - denoised
                denoised = denoised + correction * _strength

            elif _mode == "adaptive":
                d_flat = denoised.flatten(2)
                r_flat = ref_resized.flatten(2)
                cos_sim = F.cosine_similarity(d_flat, r_flat, dim=1)
                weight = cos_sim.clamp(0.0, 1.0)
                weight = weight.unsqueeze(1)
                weight = weight.view(denoised.shape[0], 1,
                                     denoised.shape[2], denoised.shape[3])

                correction = ref_resized - denoised
                denoised = denoised + correction * weight * _strength

            elif _mode == "channel_match":
                ref_mean = ref_resized.mean(dim=(2, 3), keepdim=True)
                ref_std = ref_resized.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
                den_mean = denoised.mean(dim=(2, 3), keepdim=True)
                den_std = denoised.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)

                normed = (denoised - den_mean) / den_std
                matched = normed * ref_std + ref_mean

                denoised = denoised + (matched - denoised) * _strength

            return denoised

        m.model_options.setdefault("sampler_post_cfg_function", []).append(post_cfg_fn)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "IdentityGuidance": IdentityGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdentityGuidance": "FLUX.2 Klein Identity Guidance",
}
