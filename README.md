# MedGemma CLI Runner

`medgemma.py` is a lightweight command-line helper for running Google's **Med-Gemma** multimodal models locally, either as an image question-answering assistant or as a text-only chat assistant. The script wraps Hugging Face `transformers`, supports optional 4-bit quantization via `bitsandbytes`, and exposes a few flags so you can tweak prompts and hardware usage easily.

## 1. Prerequisites

- Python 3.9 or newer
- Sufficient GPU/CPU memory for the selected Med-Gemma checkpoint (4-bit mode helps on smaller GPUs)
- (Recommended) A CUDA-capable GPU with up-to-date drivers
- Hugging Face account and access token if the chosen model repo is gated

## 2. Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip torch torchvision 
# Core libraries
pip install transformers accelerate huggingface_hub bitsandbytes pillow
```

Notes:

- `bitsandbytes` is optional but strongly recommended if you plan to use 4-bit quantization.

## 3. Authentication

Authenticate to the Hugging Face Hub once, so gated model downloads succeed:

```bash
export HF_TOKEN=hf_your_access_token
```

The script checks `HF_TOKEN` and calls `huggingface_hub.login()` automatically before loading weights.

## 4. Usage

Run the script directly:

```bash
python medgemma.py [options]
```

### 4.1 Image Question Answering

```bash
python medgemma.py \
    --image /path/to/xray.png \
    --prompt "Summarize the key findings." \
    --max_new_tokens 512
```

The script loads the image with Pillow, packages the prompt using the model's chat template, and prints the generated answer.

### 4.2 Text-Only Chat

```bash
python medgemma.py \
    --prompt "Outline a diagnostic approach for chest pain." \
    --system "You are a concise medical assistant."
```

Omit `--image` to enter text-only mode. The system prompt is optional and defaults to a helpful assistant persona.

### 4.3 Thinking Mode (27B Variants)

If you are using a Med-Gemma checkpoint that supports the `<think>` token (e.g., 27B models), enable reflective reasoning:

```bash
python medgemma.py \
    --model google/medgemma-27b \
    --prompt "What differential diagnoses should be considered here?" \
    --think
```

The script prints intermediate thoughts (between `<think>` and `<end_thought>`) before showing the final answer.

## 5. Command-Line Options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model` | `google/medgemma-4b-it` | Hugging Face model repository to load. |
| `--image` | `None` | Path to an input image for VQA mode. |
| `--prompt` | `"What are the key findings in this picture?"` | User prompt/question. |
| `--system` | `"You are a helpful medical assistant."` | System prompt for text-only mode. |
| `--think` | _disabled_ | Adds `<think>` token for models that support multi-step reasoning. |
| `--no-4bit` | _disabled_ | Disable bitsandbytes 4-bit quantization (requires more VRAM). |
| `--dtype` | `bfloat16` | Computation dtype (`bfloat16`, `float16`, `float32`). |
| `--device` | `auto` | Device placement (`auto`, `cuda`, `cpu`). |
| `--max_new_tokens` | `512` | Maximum tokens to generate. |

## 6. Tips & Troubleshooting

- **Missing `bitsandbytes`**: Install the latest prebuilt wheel (`pip install bitsandbytes>=0.43`). If problems persist, rerun with `--no-4bit`.
- **CUDA errors**: Ensure your NVIDIA driver and CUDA runtime match the PyTorch wheel you installed. Use the compatibility matrix from pytorch.org for guidance.
- **Model download failures**: Confirm that `HF_TOKEN` is set and that your account has accepted the model license terms.
- **Logging**: The script prints informational messages for login, quantization, and generation steps. Redirect stdout if you want to capture model outputs: `python medgemma.py ... > output.txt`.
