#!/usr/bin/env python3
"""
excat - Generate a quantization signature cat image from an ExLlama quant config.

Takes a quantization_config.json and a base cat image, slices the cat into
horizontal bands (one per model layer), and tints each band based on the
average bits-per-weight of that layer.

Color scheme (asymmetric gradient):
    2 bpw  ->  Red    (255, 0, 0)       heavily quantized
    4 bpw  ->  Orange (255, 165, 0)     neutral setpoint
    8 bpw  ->  Yellow (255, 255, 0)     high fidelity
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


# --- Color constants ---
COLOR_RED = (255, 0, 0)       # 2 bpw
COLOR_ORANGE = (255, 165, 0)  # 4 bpw
COLOR_YELLOW = (255, 255, 0)  # 8 bpw

BPW_LOW = 2.0
BPW_MID = 4.0
BPW_HIGH = 8.0


def bpw_to_color(bpw: float) -> tuple[int, int, int]:
    """Convert a bits-per-weight value to an RGB color.

    Uses an asymmetric piecewise linear gradient:
        [2, 4] -> Red to Orange   (steep: 2 bpw range)
        [4, 8] -> Orange to Yellow (gentle: 4 bpw range)
    """
    bpw = max(BPW_LOW, min(BPW_HIGH, bpw))

    if bpw <= BPW_MID:
        t = (bpw - BPW_LOW) / (BPW_MID - BPW_LOW)
        return (
            int(COLOR_RED[0] + t * (COLOR_ORANGE[0] - COLOR_RED[0])),
            int(COLOR_RED[1] + t * (COLOR_ORANGE[1] - COLOR_RED[1])),
            int(COLOR_RED[2] + t * (COLOR_ORANGE[2] - COLOR_RED[2])),
        )
    else:
        t = (bpw - BPW_MID) / (BPW_HIGH - BPW_MID)
        return (
            int(COLOR_ORANGE[0] + t * (COLOR_YELLOW[0] - COLOR_ORANGE[0])),
            int(COLOR_ORANGE[1] + t * (COLOR_YELLOW[1] - COLOR_ORANGE[1])),
            int(COLOR_ORANGE[2] + t * (COLOR_YELLOW[2] - COLOR_ORANGE[2])),
        )


def parse_quant_config(config_path: str) -> list[float]:
    """Parse a quantization_config.json and return average bpw per layer."""
    with open(config_path) as f:
        data = json.load(f)

    tensor_storage = data.get("tensor_storage", {})

    layer_bpws: dict[int, list[float]] = {}
    for key, entry in tensor_storage.items():
        if "layers." not in key:
            continue
        bpw = entry.get("bits_per_weight")
        if bpw is None:
            continue
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                layer_bpws.setdefault(layer_idx, []).append(bpw)
                break

    if not layer_bpws:
        print("Error: No quantized layers found in config.", file=sys.stderr)
        sys.exit(1)

    return [
        sum(bpws) / len(bpws)
        for _, bpws in sorted(layer_bpws.items())
    ]


def find_content_bbox(img: Image.Image, threshold: int = 240) -> tuple[int, int, int, int]:
    """Find the bounding box of non-white content in the image."""
    gray = img.convert("L")
    pixels = gray.load()
    w, h = gray.size

    top = 0
    for y in range(h):
        for x in range(w):
            if pixels[x, y] < threshold:
                top = y
                break
        else:
            continue
        break

    bottom = h - 1
    for y in range(h - 1, -1, -1):
        for x in range(w):
            if pixels[x, y] < threshold:
                bottom = y
                break
        else:
            continue
        break

    left = 0
    for x in range(w):
        for y in range(h):
            if pixels[x, y] < threshold:
                left = x
                break
        else:
            continue
        break

    right = w - 1
    for x in range(w - 1, -1, -1):
        for y in range(h):
            if pixels[x, y] < threshold:
                right = x
                break
        else:
            continue
        break

    return (left, top, right + 1, bottom + 1)


def generate_excat(
    cat_path: str,
    config_path: str,
    output_path: str,
    border: int = 20,
) -> None:
    """Generate the quantization signature cat image."""
    layer_bpws = parse_quant_config(config_path)
    num_layers = len(layer_bpws)
    print(f"Parsed {num_layers} layers from {config_path}")
    for i, bpw in enumerate(layer_bpws):
        color = bpw_to_color(bpw)
        print(f"  Layer {i:3d}: {bpw:.3f} bpw -> RGB{color}")

    # Load and crop the cat image
    cat = Image.open(cat_path).convert("RGBA")
    bbox = find_content_bbox(cat)
    cat_cropped = cat.crop(bbox)
    cw, ch = cat_cropped.size

    # Make it square-ish: use the larger dimension
    side = max(cw, ch) + 2 * border
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))

    # Center the cat on the canvas
    offset_x = (side - cw) // 2
    offset_y = (side - ch) // 2
    canvas.paste(cat_cropped, (offset_x, offset_y), cat_cropped)

    # Determine the cat content region on the canvas for slicing
    cat_top = offset_y
    cat_bottom = offset_y + ch

    # Create the tinted output
    result = canvas.copy()
    pixels = result.load()

    # For each layer, compute the horizontal band and tint white/light pixels
    for layer_idx, bpw in enumerate(layer_bpws):
        color = bpw_to_color(bpw)

        # Map layer index to pixel rows within the cat content area
        band_top = cat_top + int(layer_idx * ch / num_layers)
        band_bottom = cat_top + int((layer_idx + 1) * ch / num_layers)

        for y in range(band_top, band_bottom):
            for x in range(side):
                r, g, b, a = pixels[x, y]
                # Tint pixels that are light (white or near-white)
                # Preserve dark pixels (outlines) as-is
                brightness = (r + g + b) / 3
                if brightness > 200:
                    # Blend: the lighter the pixel, the more tint we apply
                    blend = (brightness - 200) / 55.0  # 0 at 200, 1 at 255
                    blend = min(1.0, blend)
                    nr = int(r + blend * (color[0] - r))
                    ng = int(g + blend * (color[1] - g))
                    nb = int(b + blend * (color[2] - b))
                    pixels[x, y] = (nr, ng, nb, a)

    result.save(output_path)
    print(f"\nSaved to {output_path} ({side}x{side}px)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a quantization signature cat image from an ExLlama quant config."
    )
    parser.add_argument(
        "config",
        help="Path to quantization_config.json",
    )
    parser.add_argument(
        "-c", "--cat",
        default="cat.png",
        help="Path to base cat image (default: cat.png)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path (default: excat_<config_name>.png)",
    )
    parser.add_argument(
        "-b", "--border",
        type=int,
        default=20,
        help="Border padding in pixels (default: 20)",
    )
    args = parser.parse_args()

    if args.output is None:
        config_stem = Path(args.config).stem
        args.output = f"excat_{config_stem}.png"

    generate_excat(args.cat, args.config, args.output, args.border)


if __name__ == "__main__":
    main()
