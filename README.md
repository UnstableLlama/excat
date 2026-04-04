# excat

Generate a unique signature cat image for any ExLlama quantized model. Each cat is a visual fingerprint of the quantization profile -- sliced into horizontal bands (one per model layer) and tinted based on the average bits-per-weight of that layer. A deterministic fur pattern is generated from the model name, giving each model its own unique look.

| Qwen3.5-0.8B (2.11bpw, classic tabby) | Ministral-3-8B (4bpw, mackerel tabby) |
|:---:|:---:|
| ![Qwen](example_qwen.png) | ![Ministral](example_ministral.png) |

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
python excat.py <quantization_config.json> [-n model_name] [-c cat.png] [-o output.png] [-b border]
```

**Arguments:**
- `config` -- Path to an ExLlama `quantization_config.json`
- `-n, --name` -- Model name for fur pattern generation (will prompt if not provided)
- `-c, --cat` -- Base cat image (default: `cat.png`)
- `-o, --output` -- Output path (default: `excat_<config_name>.png`)
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
