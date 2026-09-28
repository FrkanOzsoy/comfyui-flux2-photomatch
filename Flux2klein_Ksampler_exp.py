import math

import torch
from tqdm.auto import trange

import comfy.model_management
import comfy.utils
import latent_preview


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15):
    timesteps = torch.linspace(1, 0, num_steps + 1)
    mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    for i, t in enumerate(timesteps):
        tv = t.item()
        if 0 < tv < 1:
            timesteps[i] = _time_shift(mu, 1.0, tv)
    return timesteps.tolist()


class Flux2KleinKSamplerExperimental:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": 200,
                    "tooltip": "Denoising steps. Distilled: 4. Base: 25-50.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "1.0 = full denoise from pure noise. Lower values blend noise into the input latent for img2img.",
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Schedule shift at minimum resolution.",
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Schedule shift at maximum resolution.",
                }),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "cfg_scale": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. 1.0 = disabled (distilled default). Base model uses ~4.0. Requires negative conditioning.",
                }),
                "guidance_embed": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Embedded guidance value. Only active if the loaded model has a guidance embedding layer. Klein 9B models do not — this will be ignored for them.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "Flux2Klein KSampler Experimental"

    def sample(self, model, positive, latent_image, steps, seed,
               denoise=1.0, base_shift=0.5, max_shift=1.15,
               negative=None, cfg_scale=1.0, guidance_embed=1.0):

        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"]
        B, C, H, W = latent.shape

        comfy.model_management.load_models_gpu([model])
        diffusion_model = model.model.diffusion_model
        model_dtype = diffusion_model.dtype

        patch_size = diffusion_model.patch_size
        h_tokens = (H + patch_size // 2) // patch_size
        w_tokens = (W + patch_size // 2) // patch_size
        image_seq_len = h_tokens * w_tokens

        schedule = _get_schedule(steps, image_seq_len, base_shift=base_shift, max_shift=max_shift)

        if denoise < 1.0:
            start_idx = 0
            for i, t in enumerate(schedule):
                if t <= denoise:
                    start_idx = max(0, i)
                    break
            schedule = schedule[start_idx:]

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")

        if denoise < 1.0:
            t_start = schedule[0]
            x = ((1.0 - t_start) * latent.float() + t_start * noise)
        else:
            x = noise

        x = x.to(device=device, dtype=model_dtype)

        cond = positive[0][0].to(device=device, dtype=model_dtype)
        if cond.shape[0] != B:
            cond = cond[:1].expand(B, -1, -1)

        neg_cond = None
        use_cfg = negative is not None and cfg_scale > 1.0
        if use_cfg:
            neg_cond = negative[0][0].to(device=device, dtype=model_dtype)
            if neg_cond.shape[0] != B:
                neg_cond = neg_cond[:1].expand(B, -1, -1)

        cond_meta = positive[0][1] if len(positive[0]) > 1 else {}
        ref_latents = None
        for key in ("ref_latents", "reference_latents", "concat_latent_image"):
            if key in cond_meta:
                ref_val = cond_meta[key]
                if isinstance(ref_val, torch.Tensor):
                    ref_latents = [ref_val.to(device=device, dtype=model_dtype)]
                elif isinstance(ref_val, list):
                    ref_latents = [r.to(device=device, dtype=model_dtype) for r in ref_val]
                break

        has_guidance_embed = getattr(diffusion_model.params, "guidance_embed", False)
        guidance_vec = None
        if has_guidance_embed:
            guidance_vec = torch.full((B,), guidance_embed, device=device, dtype=model_dtype)

        transformer_options = model.model_options.get("transformer_options", {}).copy()

        total_steps = len(schedule) - 1
        pbar = comfy.utils.ProgressBar(total_steps)
        preview_callback = latent_preview.prepare_callback(model, total_steps)

        mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        patch_count = len(model.patches) if hasattr(model, "patches") else 0
        print(f"[Flux2Klein Exp] {W * (16 // patch_size)}x{H * (16 // patch_size)} | "
              f"patches: {patch_count} | mu: {mu:.4f} | shift: {math.exp(mu):.2f} | "
              f"seq_len: {image_seq_len} | steps: {total_steps} | "
              f"cfg: {cfg_scale} | guidance_embed: {has_guidance_embed} | denoise: {denoise}")

        with torch.no_grad():
            for i in trange(total_steps, desc="Flux2Klein Exp"):
                comfy.model_management.throw_exception_if_processing_interrupted()

                t_curr = schedule[i]
                t_prev = schedule[i + 1]
                t_vec = torch.full((B,), t_curr, device=device, dtype=model_dtype)
                transformer_options["sigmas"] = t_vec

                pred = diffusion_model.forward(
                    x, t_vec, cond,
                    y=None,
                    guidance=guidance_vec,
                    ref_latents=ref_latents,
                    control=None,
                    transformer_options=transformer_options,
                )

                if use_cfg:
                    pred_uncond = diffusion_model.forward(
                        x, t_vec, neg_cond,
                        y=None,
                        guidance=guidance_vec,
                        ref_latents=ref_latents,
                        control=None,
                        transformer_options=transformer_options,
                    )
                    pred = pred_uncond + cfg_scale * (pred - pred_uncond)

                if preview_callback is not None:
                    x0_est = (x - t_curr * pred) if t_curr > 1e-6 else x
                    preview_callback(i, x0_est.cpu().float(), x.cpu().float(), total_steps)

                x = x + (t_prev - t_curr) * pred
                pbar.update(1)

        return ({"samples": x.cpu().float()},)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinKSamplerExperimental": Flux2KleinKSamplerExperimental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinKSamplerExperimental": "Flux2Klein KSampler Experimental",
}
