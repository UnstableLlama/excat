# excat

Generate a unique signature cat image for any ExLlama quantized model. Each cat is a visual fingerprint of the quantization profile -- sliced into horizontal bands (one per model layer) and tinted based on the average bits-per-weight of that layer. A deterministic fur pattern is generated from the model name, giving each model its own unique look.

| Qwen3.5-0.8B 2.1bpw | Ministral-3-8B 4bpw, pixelized | Qwen3.5-0.8B 2.1bpw, pixcat | Ministral-3-8B 4bpw, pixcat + pixelized |
|:---:|:---:|:---:|:---:|
| ![Qwen cat](example_qwen_cat.png) | ![Ministral cat pixelized](example_ministral_cat_px.png) | ![Qwen pixcat](example_qwen_pixcat.png) | ![Ministral pixcat pixelized](example_ministral_pixcat_px.png) |

## Color Scheme

The color gradient is asymmetric, reflecting the fact that quality loss is more dramatic at lower bit depths:

| bpw | Color | Meaning |
|-----|-------|---------|
| 2 | Red | Heavily quantized |
| 4 | Orange | Neutral setpoint |
| 8 | Yellow | High fidelity |

The gradient from 2-4 bpw (red to orange) is steeper than 4-8 bpw (orange to yellow), making aggressive low-bit quantization visually louder.

## Fur Patterns

The model name is hashed to deterministically generate a unique fur pattern. The hash controls the pattern type, scale, angle, density, and other parameters -- so the same model name always produces the same cat.

| Pattern | Description |
|---------|-------------|
| Mackerel tabby | Wavy parallel stripes |
| Classic tabby | Swirly, organic splotches |
| Splotches | Large irregular patches |
| Spotted | Scattered round spots |

## Usage

```
python excat.py <quantization_config.json> [-n model_name] [-s style] [-p pixel_size] [-o output.png]
```

**Arguments:**
- `config` -- Path to an ExLlama `quantization_config.json`
- `-n, --name` -- Model name for fur pattern generation (will prompt if not provided)
- `-s, --style` -- Cat style: `cat` (default) or `pixcat`
- `-c, --cat` -- Path to a custom cat image (overrides `--style`)
- `-o, --output` -- Output path (default: `excat_<config_name>.png`)
- `-p, --pixelize` -- Pixelize the fur with given block size (e.g. `-p 10`). Off by default
- `-b, --border` -- Border padding in pixels (default: 20)

**Requirements:** Python 3, Pillow

```
pip install Pillow
```

## How It Works

1. Parses the quantization config and computes the average bpw per layer
2. Hashes the model name to derive fur pattern parameters
3. Crops the base cat image and squares it with a white border
4. Detects background and eye whites via flood-fill so only the cat interior is colored
5. Slices the cat into horizontal bands (one per layer) and tints each based on its bpw
6. Overlays the fur pattern as black markings on the tinted interior
7. Optionally pixelizes the interior for a retro look
