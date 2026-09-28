from .flux2_klein_ref_controller import NODE_CLASS_MAPPINGS as REF_NODES, NODE_DISPLAY_NAME_MAPPINGS as REF_NAMES
from .flux2_klein_text_enhancer import NODE_CLASS_MAPPINGS as TEXT_NODES, NODE_DISPLAY_NAME_MAPPINGS as TEXT_NAMES
from .flux2_klein_enhancer import Flux2KleinEnhancer, Flux2KleinDetailController
from .flux2_sectioned_encoder import Flux2KleinSectionedEncoder
from .flux2_klein_mask_ref_controller import Flux2KleinMaskRefController
from .flux2_klein_color_anchor import Flux2KleinColorAnchor
from .identity_guidance import IdentityGuidance
from .identity_feature_transfer import (
    IdentityFeatureTransfer,
    IdentityFeatureTransferAdvanced,
    IdentityFeatureTransferV3,
    IdentityFeatureTransferFinal,
)
from .multi_reference_latent import NODE_CLASS_MAPPINGS as MULTI_REF_NODES, NODE_DISPLAY_NAME_MAPPINGS as MULTI_REF_NAMES
from .Flux2klein_Ksampler_exp import NODE_CLASS_MAPPINGS as EXP_NODES, NODE_DISPLAY_NAME_MAPPINGS as EXP_NAMES

# ----------------------------------------------------------------------
# PhotoColorGrainMatch - Lab median/MAD skin-tone + grain match.
# CPU only (OpenCV/NumPy): no model, no VRAM, deterministic.
# ----------------------------------------------------------------------

class PhotoColorGrainMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_photo": ("IMAGE",),
                "skin_mask": ("MASK",),
                "luminance_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_added_grain_8bit": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 16.0, "step": 0.5}),
                "feather_px": ("INT", {"default": 8, "min": 0, "max": 64}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "target_skin_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("matched_image", "added_grain_8bit")
    FUNCTION = "match"
    CATEGORY = "image/postprocessing"

    def match(self, image, target_photo, skin_mask, luminance_strength, color_strength,
              grain_strength, max_added_grain_8bit, feather_px, seed, target_skin_mask=None):
        outputs = []
        added_levels = []
        for index in range(image.shape[0]):
            height, width = image.shape[1:3]
            source = _frame(image, index, height, width)
            target = _frame(target_photo, index, height, width)
            source_mask = _mask(skin_mask, index, height, width)
            reference_mask = _mask(target_skin_mask, index, height, width) if target_skin_mask is not None else source_mask
            if np.count_nonzero(source_mask > 0.5) < 64 or np.count_nonzero(reference_mask > 0.5) < 64:
                outputs.append(source)
                added_levels.append(0.0)
                continue

            source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
            target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
            source_median, source_mad = _median_and_mad(source_lab[source_mask > 0.5])
            target_median, target_mad = _median_and_mad(target_lab[reference_mask > 0.5])
            scale = np.clip(target_mad / source_mad, 0.8, 1.2)
            adjusted_lab = (source_lab - source_median) * scale + target_median
            strength = np.array([luminance_strength, color_strength, color_strength], dtype=np.float32)
            matched_lab = source_lab + (adjusted_lab - source_lab) * strength
            matched_lab[..., 0] = np.clip(matched_lab[..., 0], 0, 100)
            matched_lab[..., 1:] = np.clip(matched_lab[..., 1:], -127, 127)
            matched = np.clip(cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB), 0, 1)

            reference_sigma = _grain_sigma(target, reference_mask)
            source_sigma = _grain_sigma(matched, source_mask)
            add_sigma = np.sqrt(max(0.0, reference_sigma ** 2 - source_sigma ** 2))
            add_sigma = min(add_sigma * grain_strength, max_added_grain_8bit / 255.0)
            rng = np.random.default_rng(seed + index)
            noise = rng.normal(0, add_sigma, (height, width, 1)).astype(np.float32)
            matched = np.clip(matched + noise, 0, 1)

            blend_mask = source_mask
            if feather_px:
                blend_mask = cv2.GaussianBlur(source_mask, (0, 0), feather_px / 3.0)
            result = source * (1 - blend_mask[..., None]) + matched * blend_mask[..., None]
            outputs.append(np.clip(result, 0, 1))
            added_levels.append(float(add_sigma * 255.0))

        result = torch.from_numpy(np.stack(outputs)).to(device=image.device, dtype=image.dtype)
        return result, float(np.mean(added_levels))


NODE_CLASS_MAPPINGS = {
    **REF_NODES,
    **TEXT_NODES,
    **MULTI_REF_NODES,
    **EXP_NODES,
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
    "Flux2KleinMaskRefController": Flux2KleinMaskRefController,
    "Flux2KleinColorAnchor": Flux2KleinColorAnchor,
    "IdentityGuidance": IdentityGuidance,
    "IdentityFeatureTransfer": IdentityFeatureTransfer,
    "IdentityFeatureTransferAdvanced": IdentityFeatureTransferAdvanced,
    "IdentityFeatureTransferV3": IdentityFeatureTransferV3,
    "IdentityFeatureTransferFinal": IdentityFeatureTransferFinal,
    "PhotoColorGrainMatch": PhotoColorGrainMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **REF_NAMES,
    **TEXT_NAMES,
    **MULTI_REF_NAMES,
    **EXP_NAMES,
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinDetailController": "FLUX.2 Klein Detail Controller",
    "Flux2KleinSectionedEncoder": "FLUX.2 Klein Sectioned Encoder",
    "Flux2KleinMaskRefController": "FLUX.2 Klein Mask Ref Controller",
    "Flux2KleinColorAnchor": "FLUX.2 Klein Color Anchor",
    "IdentityGuidance": "FLUX.2 Klein Identity Guidance",
    "IdentityFeatureTransfer": "FLUX.2 Klein Identity Feature Transfer",
    "IdentityFeatureTransferAdvanced": "FLUX.2 Klein Identity Feature Transfer Advanced",
    "IdentityFeatureTransferV3": "FLUX.2 Klein Identity Feature Transfer V3",
    "IdentityFeatureTransferFinal": "Identity Feature Transfer Final",
    "PhotoColorGrainMatch": "Photo Color + Grain Match",
}

__version__ = "3.4.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "1.0.0"
