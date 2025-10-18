import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import torch

def run_inference():
    # ---------------------- Configuration ----------------------
    input_dir = "input/val"
    output_dir = "output/generated"
    model_name = "maps_pix2pix_SketchToReal"
    checkpoints_dir = "./checkpoints"
    num_test = 5
    direction = "BtoA"

    # ---------------------- Options Setup ----------------------
    opt = TestOptions().parse()
    opt.dataroot = input_dir
    opt.name = model_name
    opt.model = "pix2pix"
    opt.checkpoints_dir = checkpoints_dir
    opt.phase = ""
    opt.direction = direction
    opt.num_test = num_test
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_threads = 0  # important for Windows

    # ---------------------- Create Dataset and Model ----------------------
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Run Inference ----------------------
    for i, data in enumerate(dataset):
        if i >= num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = data['A_paths'][0]

        # Get generated B
        fake_img = visuals["fake_B"].squeeze().cpu().permute(1, 2, 0).numpy()
        fake_img = ((fake_img + 1) / 2.0 * 255).astype("uint8")
        out_img = Image.fromarray(fake_img)

        # Save
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        out_img.save(out_path)
        print(f"✅ Generated: {out_path}")

if __name__ == "__main__":
    run_inference()
