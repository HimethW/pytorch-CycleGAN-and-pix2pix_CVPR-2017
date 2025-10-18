import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import sys

sys.path.append('.')

class SimpleMapGenerator:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model exactly like test.py"""
        try:
            from models import create_model
            from options.test_options import TestOptions
            
            # Create command line arguments like test.py
            sys.argv = ['test.py', 
                       '--dataroot', './datasets/maps',
                       '--name', 'maps_pix2pix_SketchToReal', 
                       '--model', 'pix2pix',
                       '--direction', 'BtoA']
            
            opt = TestOptions().parse()
            opt.num_threads = 0
            opt.batch_size = 1
            opt.serial_batches = True
            opt.no_flip = True
            opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Add device
            
            self.model = create_model(opt)
            self.model.setup(opt)
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
    
    def process_image(self, input_image):
        """Process image exactly like test.py"""
        if self.model is None:
            return "Model not loaded"
        
        if input_image is None:
            return "Please upload an image"
        
        try:
            # Convert to RGB and resize
            image = input_image.convert('RGB').resize((256, 256))
            
            # Convert to tensor like the dataset
            image_array = np.array(image).transpose(2, 0, 1)
            image_tensor = torch.from_numpy(image_array).float()
            image_tensor = (image_tensor / 255.0) * 2 - 1  # Normalize to [-1, 1]
            image_tensor = image_tensor.unsqueeze(0)
            
            # Run inference like test.py
            with torch.no_grad():
                self.model.set_input({'B': image_tensor, 'A_paths': ['input']})
                self.model.test()
                visuals = self.model.get_current_visuals()
                result = visuals['fake_A']
            
            # Convert back to image
            result_array = result[0].cpu().float().numpy()
            result_array = (np.transpose(result_array, (1, 2, 0)) + 1) / 2.0 * 255.0
            result_array = np.clip(result_array, 0, 255).astype(np.uint8)
            result_image = Image.fromarray(result_array)
            
            return result_image
            
        except Exception as e:
            return f"Error: {str(e)}"

# Create the interface
generator = SimpleMapGenerator()

def run_inference(image):
    return generator.process_image(image)

# Simple interface with upload only
iface = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(label="Upload Map Sketch", type="pil"),
    outputs=gr.Image(label="Generated Aerial View", type="pil"),
    title="Sketch to Aerial Map Generator",
    description="Upload a map sketch to generate a realistic aerial view"
)

if __name__ == "__main__":
    print("🚀 Starting server...")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)