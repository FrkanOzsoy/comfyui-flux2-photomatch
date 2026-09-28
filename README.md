# ComfyUI FLUX.2 Klein Enhancer + Photo Color/Grain Match

A single ComfyUI custom-node pack combining two node sets into one installable
repository:

- **FLUX.2 Klein Enhancer** — conditioning and sampling helpers built for the
  Klein family (colour anchoring, detail control, sectioned encoding, identity
  guidance / feature transfer, multi-reference latents, an experimental KSampler).
- **Photo Color + Grain Match** (`PhotoColorGrainMatch`) — deterministic
  OpenCV/NumPy skin-tone and film-grain matching. Runs entirely on CPU: no model,
  no VRAM, no network access, byte-for-byte reproducible for a given seed.

Neither node set was published to the ComfyUI Registry, so they had to be copied
between machines by hand. This repository makes them a normal `git clone` away.

## Nodes provided

| Node type | Display name |
|---|---|
| `PhotoColorGrainMatch` | Photo Color + Grain Match |
| `Flux2KleinColorAnchor` | FLUX.2 Klein Color Anchor |
| `Flux2KleinEnhancer` | FLUX.2 Klein Enhancer |
| `Flux2KleinDetailController` | FLUX.2 Klein Detail Controller |
| `Flux2KleinSectionedEncoder` | FLUX.2 Klein Sectioned Encoder |
| `Flux2KleinMaskRefController` | FLUX.2 Klein Mask Ref Controller |
| `Flux2KleinTextEnhancer` | FLUX.2 Klein Text Enhancer |
| `IdentityGuidance` | FLUX.2 Klein Identity Guidance |
| `IdentityFeatureTransfer` | FLUX.2 Klein Identity Feature Transfer |
| `IdentityFeatureTransferAdvanced` | FLUX.2 Klein Identity Feature Transfer Advanced |
| `IdentityFeatureTransferV3` | FLUX.2 Klein Identity Feature Transfer V3 |
| `IdentityFeatureTransferFinal` | Identity Feature Transfer Final |
| `Flux2KleinKSamplerExperimental` | Flux2Klein KSampler Experimental |

## Requirements

Only what ComfyUI already ships: `torch`, `numpy`, `opencv-python`.
No `requirements.txt` is needed and nothing is downloaded at import time.

## Install

**ComfyUI Manager** — search for the repository name, or use Manager →
Install custom node → from URL.

**Git**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/comfyui-flux2-photomatch.git
```

Restart ComfyUI. The repository root *is* the node folder, so no extra move
step is required.

## Mask convention (important for API callers)

`PhotoColorGrainMatch` expects two masks, and both are ordinary ComfyUI `MASK`
tensors. When the mask comes from a `LoadImage` node it is derived from the
image's **alpha channel** (transparent = masked), which is what ComfyUI's
clipspace mask painter produces.

If you are driving this from an API rather than the UI, do not rely on the
clipspace painter — it is a browser-side action. Supply a PNG whose alpha
channel already carries the mask. `Lanczos`/`nearest-exact` resizing of a mask
keeps it aligned with the image as long as both are scaled with the same
transform and method.

## Notes on `PhotoColorGrainMatch`

It applies, over the blended region:

```
adjusted = (source - median(source)) * scale + median(reference)
scale    = clip(median_abs_dev(reference) / median_abs_dev(source), 0.8, 1.2)
```

so it performs a Lab **median shift** plus a bounded **contrast/saturation**
rescale, optionally attenuated per channel via `luminance_strength` and
`color_strength`, and adds luminance grain matched to the reference.

Two practical consequences:

- Pass an **all-ones mask** and a whole-frame reference to grade the entire
  image uniformly. A tight mask with a small `feather_px` produces a visibly
  hard-edged patch, because the feather is capped at 64 px.
- `scale` is clamped, so on strongly mismatched input it pins at a bound and
  over-expands chroma. Lower `color_strength` if skin turns waxy or highlights
  go cream.

## Provenance

This pack was assembled from two locally authored node folders; no upstream
source was available for either. The FLUX.2 Klein modules and
`PhotoColorGrainMatch` are preserved as-is — merging added a registration entry
and nothing else.

## License

MIT — see `LICENSE`.
