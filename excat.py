#!/usr/bin/env python3
"""
excat - Generate a quantization signature cat image from an ExLlama quant config.

Takes a quantization_config.json and a base cat image, slices the cat into
horizontal bands (one per model layer), and tints each band based on the
average bits-per-weight of that layer. Overlays a deterministic fur pattern
derived from the model name hash.

Color scheme (asymmetric gradient):
    2 bpw  ->  Red    (255, 0, 0)       heavily quantized
    4 bpw  ->  Orange (255, 165, 0)     neutral setpoint
    8 bpw  ->  Yellow (255, 255, 0)     high fidelity
"""

import argparse
import hashlib
import json
import math
import sys
from collections import deque
from pathlib import Path

from PIL import Image


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


# ---------------------------------------------------------------------------
# Perlin noise (pure Python, deterministic from seed)
# ---------------------------------------------------------------------------

class PerlinNoise:
    """2D Perlin noise generator with deterministic seeding."""

    def __init__(self, seed: int):
        rng = self._make_rng(seed)
        self.perm = list(range(256))
        for i in range(255, 0, -1):
            j = rng() % (i + 1)
            self.perm[i], self.perm[j] = self.perm[j], self.perm[i]
        self.perm *= 2  # double for wrapping

        # 12 gradient directions
        self.grads = [
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
        ]

    @staticmethod
    def _make_rng(seed: int):
        """Simple LCG random number generator."""
        state = [seed & 0xFFFFFFFF]
        def rng():
            state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
            return state[0]
        return rng

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float, y: float) -> float:
        g = self.grads[hash_val % 12]
        return g[0] * x + g[1] * y

    def noise(self, x: float, y: float) -> float:
        """Return noise value in [-1, 1] for coordinates (x, y)."""
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)

        u = self._fade(xf)
        v = self._fade(yf)

        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]

        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1), self._grad(bb, xf - 1, yf - 1), u)

        return self._lerp(x1, x2, v)

    def octave_noise(self, x: float, y: float, octaves: int = 4, persistence: float = 0.5) -> float:
        """Fractal noise with multiple octaves. Returns value in ~[-1, 1]."""
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_val = 0.0
        for _ in range(octaves):
            total += self.noise(x * frequency, y * frequency) * amplitude
            max_val += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / max_val


# ---------------------------------------------------------------------------
# Pattern generation (seeded from model name)
# ---------------------------------------------------------------------------

PATTERN_TYPES = ["tabby_mackerel", "tabby_classic", "splotches", "spotted"]


def hash_model_name(name: str) -> dict:
    """Hash a model name and derive all pattern parameters deterministically."""
    h = hashlib.sha256(name.encode()).hexdigest()

    # Use different slices of the hash for different parameters
    seed = int(h[:8], 16)
    pattern_idx = int(h[8:10], 16) % len(PATTERN_TYPES)
    scale = 3.0 + (int(h[10:12], 16) / 255.0) * 4.0       # 3.0 - 7.0
    threshold = 0.15 + (int(h[12:14], 16) / 255.0) * 0.20  # 0.15 - 0.35
    angle = (int(h[14:16], 16) / 255.0) * math.pi           # 0 - pi
    stripe_freq = 6.0 + (int(h[16:18], 16) / 255.0) * 6.0  # 6.0 - 12.0
    warp = 0.3 + (int(h[18:20], 16) / 255.0) * 0.7         # 0.3 - 1.0
    density = 0.20 + (int(h[20:22], 16) / 255.0) * 0.15    # 0.20 - 0.35
    octaves = 3 + int(h[22:24], 16) % 3                     # 3 - 5
    spot_count = 15 + int(h[24:26], 16) % 25                # 15 - 39

    params = {
        "seed": seed,
        "pattern_type": PATTERN_TYPES[pattern_idx],
        "scale": scale,
        "threshold": threshold,
        "angle": angle,
        "stripe_freq": stripe_freq,
        "warp": warp,
        "density": density,
        "octaves": octaves,
        "spot_count": spot_count,
    }
    return params


def generate_pattern(w: int, h: int, params: dict) -> list[list[float]]:
    """Generate a fur pattern as a 2D grid of darkness values in [0, 1].

    0 = no marking, 1 = full black marking.
    """
    perlin = PerlinNoise(params["seed"])
    pattern = [[0.0] * w for _ in range(h)]
    ptype = params["pattern_type"]

    cos_a = math.cos(params["angle"])
    sin_a = math.sin(params["angle"])

    # Marking darkness: how dark the black markings are (0.6 = 60% darkened)
    marking = 0.6

    if ptype == "tabby_mackerel":
        # Wavy stripes: sine wave with gentle noise displacement
        freq = params["stripe_freq"]
        warp_strength = params["warp"] * 0.4
        scale = params["scale"] * 0.7
        density = params["density"]
        stripe_width = 0.15 + density * 0.12

        for y in range(h):
            for x in range(w):
                nx = x / w
                ny = y / h
                rx = nx * cos_a + ny * sin_a
                warp_val = perlin.octave_noise(nx * scale, ny * scale, 2)
                stripe = math.sin((rx + warp_val * warp_strength) * freq * math.pi * 2)
                cutoff = 1.0 - stripe_width
                if stripe > cutoff:
                    pattern[y][x] = marking

    elif ptype == "tabby_classic":
        # Swirly/bulls-eye pattern: noise thresholded into organic blobs
        scale = params["scale"]
        threshold = params["threshold"]

        for y in range(h):
            for x in range(w):
                nx = x / w
                ny = y / h
                n1 = perlin.octave_noise(nx * scale, ny * scale, params["octaves"])
                n2 = perlin.octave_noise(
                    nx * scale * 0.5 + 100, ny * scale * 0.5 + 100, 2
                )
                combined = n1 + n2 * 0.3
                if combined > threshold:
                    pattern[y][x] = marking

    elif ptype == "splotches":
        # Large irregular patches using low-frequency noise
        scale = params["scale"] * 0.6
        threshold = params["threshold"] + 0.05

        for y in range(h):
            for x in range(w):
                nx = x / w
                ny = y / h
                n = perlin.octave_noise(nx * scale, ny * scale, 3, persistence=0.6)
                detail = perlin.octave_noise(nx * scale * 3, ny * scale * 3, 2) * 0.15
                combined = n + detail
                if combined > threshold:
                    pattern[y][x] = marking

    elif ptype == "spotted":
        # Spots using Voronoi-like approach seeded from hash
        rng_state = [params["seed"]]
        def rng_float():
            rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7FFFFFFF
            return rng_state[0] / 0x7FFFFFFF

        spot_count = params["spot_count"]
        spots = []
        for _ in range(spot_count):
            sx = rng_float()
            sy = rng_float()
            radius = 0.02 + rng_float() * 0.04
            spots.append((sx, sy, radius))

        scale = params["scale"]
        for y in range(h):
            for x in range(w):
                nx = x / w
                ny = y / h
                warp_x = perlin.noise(nx * scale, ny * scale) * 0.03
                warp_y = perlin.noise(nx * scale + 50, ny * scale + 50) * 0.03
                wnx = nx + warp_x
                wny = ny + warp_y

                for sx, sy, sr in spots:
                    dist = math.sqrt((wnx - sx) ** 2 + (wny - sy) ** 2)
                    if dist < sr * 0.85:  # hard edge slightly inside radius
                        pattern[y][x] = marking
                        break

    return pattern


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

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


def build_background_mask(
    img: Image.Image,
    threshold: int = 200,
    eye_max_fraction: float = 0.0015,
) -> list[list[bool]]:
    """Flood-fill from image edges to identify background pixels, plus eye whites.

    Returns a 2D boolean grid where True = masked (don't tint).
    Also detects small enclosed white regions (eye whites) by finding interior
    white blobs smaller than eye_max_fraction of total image area.
    """
    gray = img.convert("L")
    pixels = gray.load()
    w, h = gray.size

    mask = [[False] * w for _ in range(h)]
    queue = deque()

    # Seed from all edge pixels that are light enough
    for x in range(w):
        for y in (0, h - 1):
            if pixels[x, y] > threshold and not mask[y][x]:
                mask[y][x] = True
                queue.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if pixels[x, y] > threshold and not mask[y][x]:
                mask[y][x] = True
                queue.append((x, y))

    # BFS flood fill through light pixels (background)
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not mask[ny][nx]:
                if pixels[nx, ny] > threshold:
                    mask[ny][nx] = True
                    queue.append((nx, ny))

    # Now find small enclosed white regions (eyes, highlights, etc.)
    # These are light pixels that weren't reached by the background flood-fill
    visited = [[False] * w for _ in range(h)]
    max_eye_pixels = int(w * h * eye_max_fraction)

    for sy in range(h):
        for sx in range(w):
            if visited[sy][sx] or mask[sy][sx]:
                continue
            if pixels[sx, sy] <= threshold:
                visited[sy][sx] = True
                continue
            # Found an unvisited light interior pixel - flood fill to measure the blob
            blob = []
            queue.append((sx, sy))
            visited[sy][sx] = True
            while queue:
                cx, cy = queue.popleft()
                blob.append((cx, cy))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                        visited[ny][nx] = True
                        if pixels[nx, ny] > threshold:
                            queue.append((nx, ny))

            # If the blob is small enough, it's likely an eye/highlight - mask it
            if len(blob) <= max_eye_pixels:
                for bx, by in blob:
                    mask[by][bx] = True

    return mask


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_excat(
    cat_path: str,
    config_path: str,
    output_path: str,
    model_name: str,
    border: int = 20,
) -> None:
    """Generate the quantization signature cat image."""
    layer_bpws = parse_quant_config(config_path)
    num_layers = len(layer_bpws)
    print(f"Parsed {num_layers} layers from {config_path}")
    for i, bpw in enumerate(layer_bpws):
        color = bpw_to_color(bpw)
        print(f"  Layer {i:3d}: {bpw:.3f} bpw -> RGB{color}")

    # Derive pattern from model name
    params = hash_model_name(model_name)
    print(f"\nModel: {model_name}")
    print(f"  Pattern: {params['pattern_type']}")
    print(f"  Scale: {params['scale']:.2f}, Angle: {math.degrees(params['angle']):.1f}deg")
    print(f"  Density: {params['density']:.2f}, Threshold: {params['threshold']:.2f}")

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

    # Build background mask via flood-fill from edges
    bg_mask = build_background_mask(canvas)

    # Generate fur pattern
    print("\nGenerating fur pattern...")
    fur_pattern = generate_pattern(side, side, params)

    # Determine the cat content region on the canvas for slicing
    cat_top = offset_y
    cat_bottom = offset_y + ch

    # Create the tinted output
    result = canvas.copy()
    pixels = result.load()

    # For each layer, compute the horizontal band and tint interior pixels only
    for layer_idx, bpw in enumerate(layer_bpws):
        color = bpw_to_color(bpw)

        # Map layer index to pixel rows within the cat content area
        band_top = cat_top + int(layer_idx * ch / num_layers)
        band_bottom = cat_top + int((layer_idx + 1) * ch / num_layers)

        for y in range(band_top, band_bottom):
            for x in range(side):
                if bg_mask[y][x]:
                    continue
                r, g, b, a = pixels[x, y]
                brightness = (r + g + b) / 3
                if brightness > 200:
                    blend = (brightness - 200) / 55.0
                    blend = min(1.0, blend)
                    nr = int(r + blend * (color[0] - r))
                    ng = int(g + blend * (color[1] - g))
                    nb = int(b + blend * (color[2] - b))

                    # Apply fur pattern as black markings on top
                    fur = fur_pattern[y][x]
                    if fur > 0:
                        nr = int(nr * (1.0 - fur))
                        ng = int(ng * (1.0 - fur))
                        nb = int(nb * (1.0 - fur))

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
        "-n", "--name",
        default=None,
        help="Model name (used to generate fur pattern). If not provided, will prompt.",
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

    if args.name is None:
        args.name = input("Enter model name: ")

    if args.output is None:
        config_stem = Path(args.config).stem
        args.output = f"excat_{config_stem}.png"

    generate_excat(args.cat, args.config, args.output, args.name, args.border)


if __name__ == "__main__":
    main()
