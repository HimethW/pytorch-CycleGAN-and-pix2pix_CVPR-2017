
import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import torch

def run_inference():
    # Example usage
    input_file = "./input/val/input.png"
    padded_file = "./input/val/input.png"
    pad_single_image(input_file, padded_file)

    # ---------------------- Configuration ----------------------
    input_dir = "./input"
    output_dir = "output/generated"
    model_name = "bags_pix2pix_SketchToReal"
    checkpoints_dir = "./checkpoints"
    num_test = 5
    direction = "AtoB"

    # ---------------------- Options Setup ----------------------
    opt = TestOptions().parse()
    opt.dataroot = input_dir
    opt.name = model_name
    opt.model = "pix2pix"
    opt.checkpoints_dir = checkpoints_dir
    opt.phase = ""
    #opt.dataset_mode = "single"
    opt.direction = direction
    opt.num_test = num_test
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_threads = 0
    opt.model_suffix = ''

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
        print(f"Generated: {out_path}")

def pad_single_image(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    new_w = w * 2  # double width

    # Create new blank image
    new_img = Image.new("RGB", (new_w, h), color=(255, 255, 255))

    # Paste original image on the left
    new_img.paste(img, (0, 0))

    # Optional: fill the right side with same image to preserve context
    # new_img.paste(img, (w, 0))

    # Save back to input folder (overwrite or new path)
    new_img.save(output_path)
    print(f"✅ Padded image saved: {output_path}")

if __name__ == "__main__":
    run_inference()


