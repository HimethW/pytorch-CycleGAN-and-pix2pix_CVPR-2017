import gradio as gr
import subprocess
from PIL import Image
from pathlib import Path
import os
import time
import sys

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
    """Main pipeline: save input → run inference → return generated output"""

    # Handle ImageEditor output (can be dict or PIL)
    if isinstance(input_image, dict):
        # Try possible keys depending on how the ImageEditor was used
        img = (
            input_image.get("image")
            or input_image.get("background")
            or input_image.get("composite")
        )
        input_image = img

    if not isinstance(input_image, Image.Image):
        return "❌ No image provided or invalid format.", None

    # Ensure directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in INPUT_DIR.glob("*"):
        f.unlink()  # clear old images

    in_path = INPUT_DIR / "input.png"
    input_image.save(in_path)
    print(f"✅ Saved input to {in_path}")

    # Run inference
    success = run_inference()
    if not success:
        return "❌ Inference failed. Check console for details.", None

    # Find latest generated image
    time.sleep(1)
    gen_images = list(OUTPUT_DIR.glob("*.png")) + list(OUTPUT_DIR.glob("*.jpg"))
    if not gen_images:
        return "❌ No output image found in ./output/generated", None

    latest = max(gen_images, key=os.path.getmtime)
    print(f"✅ Found generated image: {latest}")
    output_image = Image.open(latest)

    return "✅ Map generated successfully!", output_image



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
                brush=gr.Brush(colors=["#cae0a9", "#e9e4de", "#aecdfc", "#f99c22"], color_mode="fixed"),
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
