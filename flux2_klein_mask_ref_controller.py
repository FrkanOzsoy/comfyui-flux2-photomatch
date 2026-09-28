import gc

import torch
import torch.nn.functional as F


class Flux2KleinMaskRefController:
    """Spatially attenuate the reference latent using a painted mask.

    Multiplies ref_latent by a per-pixel scalar derived from the mask.
    Conditioning latent is mutated in-place (replaces meta['reference_latents']).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Attenuation strength in the BLACK regions of the mask. "
                        "1.0 = black regions multiplied by 0 (full attenuation). "
                        "0.5 = black regions kept at 50% of ref strength. "
                        "0.0 = mask ignored, ref passes through everywhere."
                    ),
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip black/white. Enable if you painted the area you want attenuated rather than preserved.",
                }),
                "feather": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Gaussian blur radius (in latent pixels) on mask edges. 0 = hard edges.",
                }),
                "reference_index": ("INT", {
                    "default": 0, "min": 0, "max": 7, "step": 1,
                    "tooltip": "Which reference latent to attenuate when multiple are connected (0 = first).",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_mask"
    CATEGORY = "conditioning/flux2klein"

    def _resize_mask_to_latent(self, mask: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
        """[B, H, W] (or [H, W]) -> [1, 1, lat_h, lat_w] bilinear-resized."""
        if mask.dim() == 2:
            m = mask.unsqueeze(0).unsqueeze(0).float()
        elif mask.dim() == 3:
            m = mask[0:1].unsqueeze(1).float()
        elif mask.dim() == 4:
            m = mask[0:1, 0:1].float()
        else:
            raise ValueError(f"unexpected mask shape {tuple(mask.shape)}")
        return F.interpolate(m, size=(lat_h, lat_w), mode="bilinear", align_corners=False)

    def _feather_mask(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        ks = radius * 2 + 1
        sigma = max(radius / 3.0, 1e-6)
        ax = torch.arange(ks, dtype=torch.float32, device=mask.device) - radius
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = (gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return F.conv2d(mask, kernel, padding=radius).clamp(0.0, 1.0)

    def apply_mask(self, conditioning, mask, strength=1.0, invert_mask=False,
                   feather=0, reference_index=0, debug=False):
        if not conditioning:
            return (conditioning,)

        if strength == 0.0:
            if debug:
                print("[MaskRefController] strength=0, passing through")
            return (conditioning,)

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            new_meta = meta.copy()
            ref_latents = meta.get("reference_latents", None)
            if not ref_latents or reference_index >= len(ref_latents):
                if debug:
                    print(f"[MaskRefController] item {idx}: no ref latent at index {reference_index}, skipping")
                output.append((cond_tensor, new_meta))
                continue

            # Operate on a copy of the chosen ref latent.
            ref = ref_latents[reference_index].float().clone()
            original_dtype = ref_latents[reference_index].dtype
            _, num_ch, lat_h, lat_w = ref.shape

            spatial_mask = self._resize_mask_to_latent(mask, lat_h, lat_w)
            if invert_mask:
                spatial_mask = 1.0 - spatial_mask
            if feather > 0:
                spatial_mask = self._feather_mask(spatial_mask, feather)

            # mask=1 -> mult=1 (preserved); mask=0 -> mult=(1-strength).
            multiplier = (1.0 - strength * (1.0 - spatial_mask)).to(ref.device)
            modified = ref * multiplier  # broadcasts [1,1,H,W] across channels

            if debug:
                attenuated_pct = float((spatial_mask < 0.5).float().mean().item() * 100)
                mean_attn = float((1.0 - multiplier.mean().item()) * 100)
                print(f"[MaskRefController] item {idx} | ref_idx={reference_index} | "
                      f"latent={lat_h}x{lat_w}x{num_ch}ch | "
                      f"strength={strength} feather={feather} invert={invert_mask}")
                print(f"  attenuated area={attenuated_pct:.1f}%  mean attenuation={mean_attn:.1f}%")
                print(f"  ref before: mean={ref.mean().item():.4f} std={ref.std().item():.4f}")
                print(f"  ref after : mean={modified.mean().item():.4f} std={modified.std().item():.4f}")

            # Replace the chosen ref latent in a new list.
            new_refs = list(ref_latents)
            new_refs[reference_index] = modified.to(original_dtype)
            new_meta["reference_latents"] = new_refs
            output.append((cond_tensor, new_meta))

        gc.collect()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinMaskRefController": Flux2KleinMaskRefController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinMaskRefController": "FLUX.2 Klein Mask Ref Controller",
}
