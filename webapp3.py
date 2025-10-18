import gradio as gr
import subprocess
from PIL import Image
from pathlib import Path
import time
import sys
import numpy as np
import subprocess, os

# --- Setup directories ---
INPUT_DIR = Path("./input/val")
OUTPUT_DIR = Path("./output/generated")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_inference():
    """Runs your inference.py script"""
    cmd = [
        sys.executable, "inference.py",
        "--dataroot", "./input",
        "--model", "pix2pix"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True


def generate_map(input_image):
    try:
        if input_image is None:
            return "❌ Please upload or draw an image", None

        # 🟢 Case 1: Uploads (PIL Image)
        if isinstance(input_image, Image.Image):
            img = input_image

        # 🟣 Case 2: Drawn canvas (Gradio sends a dict)
        elif isinstance(input_image, dict):
            if "image" in input_image and input_image["image"] is not None:
                arr = np.array(input_image["image"]).astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
            else:
                return "❌ Empty canvas — please draw something", None
        else:
            return "⚠️ Unknown image format", None

        # Save to input/val
        in_dir = Path("input/val")
        in_dir.mkdir(parents=True, exist_ok=True)
        in_path = in_dir / "input.png"
        img.save(in_path)
        print(f"✅ Saved input to {in_path}")

        # Run inference
        cmd = [
            "python", "inference.py",
            "--dataroot", "./input",
            "--model", "pix2pix"
        ]
        subprocess.run(cmd, cwd=".", shell=True)

        # Load latest generated image
        out_dir = Path("output/generated")
        out_images = list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))
        if not out_images:
            return "❌ No output generated", None

        latest = max(out_images, key=os.path.getmtime)
        print(f"✅ Generated output: {latest}")
        return "✅ Generation complete!", Image.open(latest)

    except Exception as e:
        return f"❌ Error: {str(e)}", None


# --- UI ---
with gr.Blocks(title="Sketch-to-Map Generator") as demo:
    gr.Markdown("## 🗺️ Sketch to Map Generator")
    gr.Markdown("Draw or upload a sketch. The model will generate an aerial map.")

    with gr.Row():
        with gr.Column():
            sketch_input = gr.ImageEditor(
                label="Draw or Upload a Sketch",
                height=400,
                width=400,
                brush=gr.Brush(colors=["#cbdfac", "#f3f0e9","#e8e5de", "#aecdfc", "#f99c22"], color_mode="fixed"),
                type="pil"
            )
            generate_btn = gr.Button("🚀 Generate Map")
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            output_img = gr.Image(label="Generated Aerial Map", height=400, width=400)

    generate_btn.click(
        fn=generate_map,
        inputs=sketch_input,
        outputs=[status, output_img]
    )

# --- Run the app ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
