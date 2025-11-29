# Z-Image Local Demo

Lightweight Gradio UI for running the Tongyi-MAI Z-Image Turbo text-to-image pipeline on your own machine with Hugging Face diffusers.

## What is here
- `app.py`: Gradio Blocks app with prompt controls, safety checks, and a simple stats footer.
- `requirements.txt`: Runtime dependencies pinned to tested versions.

## Prerequisites
- Python 3.10+ recommended.
- CUDA GPU optional; also works on Apple Silicon (MPS) or CPU with slower performance.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app
```bash
python app.py
```
Gradio prints a local URL (default http://127.0.0.1:7860) to open in your browser.

## Configuration
- `ZIMAGE_MODEL_ID`: Override the default `Tongyi-MAI/Z-Image-Turbo` checkpoint.
- `ZIMAGE_CPU_OFFLOAD=1`: Offload layers to CPU when VRAM is limited on CUDA.
- `PYTORCH_ENABLE_MPS_FALLBACK=1`: Helps Apple Silicon fall back to CPU for unsupported ops (set automatically in `app.py`).
- If the model requires authentication, run `huggingface-cli login` before starting the app.

## Notes for git
- `.gitignore` excludes caches, virtual environments, build outputs, and common model weight formats.
- Avoid committing downloaded checkpoints or large artifacts.

