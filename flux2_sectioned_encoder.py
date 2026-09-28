import re
from typing import Dict, List, Optional, Tuple

import torch


# Klein's chat template (matches ComfyUI's KleinTokenizer.llama_template).
_KLEIN_CHAT_TEMPLATE = (
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _get_hf_tokenizer(clip):
    """Return the underlying HF tokenizer (Qwen3-8B or Qwen3-4B variant)."""
    tok = getattr(clip, "tokenizer", None)
    if tok is None:
        return None
    for attr in ("qwen3_8b", "qwen3_4b"):
        sub = getattr(tok, attr, None)
        if sub is not None and hasattr(sub, "tokenizer"):
            return sub.tokenizer
    return None


def _count_tokens(hf_tok, text: str) -> int:
    if not text:
        return 0
    out = hf_tok(text, add_special_tokens=False, return_tensors=None)
    return len(out["input_ids"])


def _compute_wrapper_lengths(hf_tok) -> Tuple[int, int]:
    """Return (prefix_len, suffix_len) of Klein's chat-template wrapper."""
    prefix, suffix = _KLEIN_CHAT_TEMPLATE.split("{}")
    return _count_tokens(hf_tok, prefix), _count_tokens(hf_tok, suffix)


def _compute_section_ranges(
    hf_tok,
    sections: Dict[str, str],
    separator: str,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """Return {'front': (s, e), 'mid': (s, e), 'end': (s, e)} in encoded-token positions.

    None if the tokenizer is missing — caller falls back to encoding without metadata.
    """
    if hf_tok is None:
        return None

    prefix_len, _ = _compute_wrapper_lengths(hf_tok)
    sep_len = _count_tokens(hf_tok, separator) if separator else 0

    # Tokenize each section's bare text.
    front_n = _count_tokens(hf_tok, sections.get("front", ""))
    mid_n = _count_tokens(hf_tok, sections.get("mid", ""))
    end_n = _count_tokens(hf_tok, sections.get("end", ""))

    # Layout: prefix | front | (sep if both front+mid present) | mid | (sep if mid+end or front+end with no mid) | end
    pos = prefix_len
    ranges: Dict[str, Tuple[int, int]] = {}

    ranges["front"] = (pos, pos + front_n)
    pos += front_n
    if front_n > 0 and mid_n > 0:
        pos += sep_len

    ranges["mid"] = (pos, pos + mid_n)
    pos += mid_n
    if (mid_n > 0 and end_n > 0) or (front_n > 0 and mid_n == 0 and end_n > 0):
        pos += sep_len

    ranges["end"] = (pos, pos + end_n)
    return ranges


def _parse_marker_sections(text: str) -> Optional[Dict[str, str]]:
    """Parse [FRONT]/[MID]/[END] markers from a single combined prompt."""
    if not text:
        return None
    pattern = r"\[(FRONT|MID|END)\](.*?)(?=\[(?:FRONT|MID|END)\]|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    sections = {"front": "", "mid": "", "end": ""}
    for name, content in matches:
        sections[name.lower()] = content.strip()
    return sections


_SEPARATORS = {"comma": ", ", "period": ". ", "space": " ", "newline": "\n"}


class Flux2KleinSectionedEncoder:
    """Tokenize a 3-section prompt and emit per-section token ranges as
    conditioning metadata that the Detail Controller can read."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "front_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "FRONT section text.",
                }),
                "mid_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "MID section text.",
                }),
                "end_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "END section text.",
                }),
                "combined_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional. Single prompt with [FRONT]/[MID]/[END] markers — overrides the three text boxes when non-empty and contains markers.",
                }),
                "separator": (list(_SEPARATORS.keys()), {
                    "default": "comma",
                    "tooltip": "How to join sections in the final prompt sent to Klein.",
                }),
                "show_preview": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "front_section", "mid_section", "end_section", "full_prompt")
    FUNCTION = "encode_sectioned"
    CATEGORY = "conditioning/flux2klein"
    OUTPUT_NODE = True

    def encode_sectioned(self, clip, front_text="", mid_text="", end_text="",
                         combined_prompt="", separator="comma",
                         show_preview=True, debug=False):

        # Resolve sections: marker mode if combined_prompt has [FRONT] etc., else 3-box.
        sections: Dict[str, str]
        marker_sections = _parse_marker_sections(combined_prompt)
        if marker_sections:
            sections = marker_sections
        else:
            sections = {
                "front": front_text or "",
                "mid": mid_text or "",
                "end": end_text or "",
            }

        sep_str = _SEPARATORS.get(separator, ", ")

        # Build the final prompt — drop empty sections from the join so we
        # don't end up with trailing/leading separators.
        parts = [sections[k] for k in ("front", "mid", "end") if sections[k]]
        full_prompt = sep_str.join(parts)

        # Standard CLIP tokenize + encode path.
        tokens = clip.tokenize(full_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        # Compute per-section token boundaries via the underlying HF tokenizer.
        hf_tok = _get_hf_tokenizer(clip)
        ranges = _compute_section_ranges(hf_tok, sections, sep_str)

        meta: Dict = {"pooled_output": pooled}
        if ranges is not None:
            meta["klein_sections"] = ranges
            if debug:
                print(f"[SectionedEncoder] computed klein_sections: {ranges}")
        else:
            print("[SectionedEncoder] WARNING: HF tokenizer not accessible on CLIP — "
                  "no klein_sections metadata emitted. Detail Controller will fall back "
                  "to fixed 25/50/25 slicing.")

        conditioning = [[cond, meta]]

        if show_preview or debug:
            self._print_preview(sections, full_prompt, ranges, separator, hf_tok)

        return (
            conditioning,
            sections["front"],
            sections["mid"],
            sections["end"],
            full_prompt,
        )

    def _print_preview(self, sections, full_prompt, ranges, separator, hf_tok):
        lines = ["", "=" * 70, "FLUX.2 Klein Sectioned Encoding (v2)", "=" * 70]
        lines.append(f"Separator: {separator!r}")

        if hf_tok is not None:
            front_n = _count_tokens(hf_tok, sections["front"])
            mid_n = _count_tokens(hf_tok, sections["mid"])
            end_n = _count_tokens(hf_tok, sections["end"])
            prefix_n, suffix_n = _compute_wrapper_lengths(hf_tok)
            lines.append(f"Section token counts (HF-tokenizer-exact, no padding):")
            lines.append(f"  FRONT: {front_n} tokens   '{sections['front']}'")
            lines.append(f"  MID:   {mid_n} tokens   '{sections['mid']}'")
            lines.append(f"  END:   {end_n} tokens   '{sections['end']}'")
            lines.append(f"Klein wrapper overhead: prefix={prefix_n} suffix={suffix_n} tokens")
        else:
            lines.append("(HF tokenizer not accessible — token counts unavailable)")

        if ranges:
            lines.append("Encoded-sequence section ranges (used by Detail Controller):")
            for name in ("front", "mid", "end"):
                s, e = ranges[name]
                lines.append(f"  {name.upper():5s}  tokens [{s}:{e})  span={e - s}")
            lines.append("Wire this conditioning into Detail Controller — it will scale these exact ranges.")
        else:
            lines.append("WARNING: no section ranges computed (HF tokenizer unavailable).")

        lines.append("-" * 70)
        lines.append(f"Final prompt: {full_prompt!r}")
        lines.append("=" * 70)
        print("\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinSectionedEncoder": "FLUX.2 Klein Sectioned Encoder",
}
