import os
from functools import lru_cache
import time
from typing import Optional, Tuple

import torch
from diffusers import ZImagePipeline
import gradio as gr


MODEL_ID = os.getenv("ZIMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")


def select_device() -> Tuple[str, torch.dtype]:
    """Pick the best device available and a compatible dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        # Use float32 on MPS to avoid NaNs/black images with some models.
        return "mps", torch.float32
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def load_pipeline():
    device, torch_dtype = select_device()
    try:
        pipe = ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Failed to load model '{MODEL_ID}': {exc}") from exc

    # Optional CPU offload to save VRAM on CUDA GPUs.
    if device == "cuda" and os.getenv("ZIMAGE_CPU_OFFLOAD", "0") == "1":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        pipe.enable_vae_slicing()

    pipe.set_progress_bar_config(disable=True)
    return pipe, device, torch_dtype


def generate_image(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int],
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    max_side = 1536
    max_pixels = 1_800_000
    min_steps = 4
    max_steps = 18
    if height > max_side or width > max_side:
        raise gr.Error(f"Image dimensions too large. Max side length is {max_side}.")
    if height * width > max_pixels:
        raise gr.Error(f"Image too large. Max total pixels is {max_pixels:,}.")
    if num_inference_steps < min_steps or num_inference_steps > max_steps:
        raise gr.Error(f"Inference steps must be between {min_steps} and {max_steps}.")

    progress(0, desc="Loading pipeline")
    pipe, device, torch_dtype = load_pipeline()
    progress(0.3, desc="Sampling")
    generator = None
    seed = int(seed) if seed is not None else None
    if seed and seed > 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        seed = None

    start = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    elapsed = time.perf_counter() - start
    footer = (
        f"Device: {device}, dtype: {torch_dtype}, "
        f"{height}x{width}, steps: {num_inference_steps}, guidance: {guidance_scale}, "
        f"seed: {seed or 'random'}, time: {elapsed:.2f}s"
    )
    progress(1.0, desc="Done")
    return result, footer


def build_demo():
    _, device, torch_dtype = load_pipeline()
    description = (
        "Z-Image Turbo text-to-image on your machine. "
        "The app auto-detects the best device (CUDA > MPS > CPU). "
        "Set ZIMAGE_CPU_OFFLOAD=1 to offload layers to CPU on CUDA GPUs "
        "if you run out of VRAM."
    )

    with gr.Blocks(title="Z-Image Gradio Demo") as demo:
        gr.Markdown(f"## Z-Image Turbo\n{description}\n\n**Running on**: `{device}` with dtype `{torch_dtype}`")

        with gr.Row():
            prompt = gr.Textbox(
                lines=4,
                label="Prompt",
                placeholder="Describe the image you want to see...",
            )
        with gr.Row():
            negative_prompt = gr.Textbox(
                lines=2,
                label="Negative prompt",
                placeholder="Things to avoid (optional)",
            )
        with gr.Row():
            height = gr.Slider(
                minimum=512,
                maximum=1152,
                step=64,
                value=1024,
                label="Height",
            )
            width = gr.Slider(
                minimum=512,
                maximum=1152,
                step=64,
                value=1024,
                label="Width",
            )
        with gr.Row():
            steps = gr.Slider(
                minimum=4,
                maximum=18,
                step=1,
                value=9,
                label="Inference steps (Turbo works well around 8-9)",
            )
            guidance = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                step=0.1,
                value=0.0,
                label="Guidance scale (keep 0 for Turbo)",
            )
            seed = gr.Slider(
                minimum=0,
                maximum=2**32 - 1,
                step=1,
                value=0,
                label="Seed (0 = random)",
            )

        generate_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            output_image = gr.Image(label="Generated image (PNG)", type="pil", format="png")
        stats = gr.Markdown()

        generate_btn.click(
            generate_image,
            inputs=[prompt, negative_prompt, height, width, steps, guidance, seed],
            outputs=[output_image, stats],
            show_progress=True,
        )

    return demo


if __name__ == "__main__":
    # Helps MPS gracefully fall back to CPU for unsupported ops.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    demo = build_demo()
    demo.queue(max_size=8).launch(server_name="0.0.0.0")
