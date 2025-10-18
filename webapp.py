import gradio as gr
import torch
from PIL import Image, ImageDraw
import numpy as np
import os
import sys
import base64
from io import BytesIO

# Add the current directory to import project modules
sys.path.append('.')

class SketchToMapGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.opt = None
        self.load_model()
        
    def load_model(self):
        """Load the trained pix2pix model with all required attributes"""
        try:
            from models import create_model
            
            print("🔄 Loading trained model...")
            
            # Create a complete MockArgs with ALL required attributes
            class MockArgs:
                def __init__(self):
                    # Required attributes from BaseOptions and TestOptions
                    self.dataroot = './datasets/maps'
                    self.name = 'maps_pix2pix_SketchToReal'
                    self.checkpoints_dir = './checkpoints'
                    self.model = 'pix2pix'
                    self.isTrain = False
                    self.gpu_ids = [0] if torch.cuda.is_available() else []
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.preprocess = 'none'
                    self.load_size = 256
                    self.crop_size = 256
                    self.batch_size = 1
                    self.serial_batches = True
                    self.no_flip = True
                    self.dataset_mode = 'single'
                    self.direction = 'BtoA'
                    self.phase = 'test'
                    self.num_test = 50
                    self.eval = True
                    self.use_wandb = False
                    self.results_dir = './results'
                    self.epoch = 'latest'
                    self.load_iter = 0
                    self.aspect_ratio = 1.0
                    self.display_winsize = 256
                    self.input_nc = 3
                    self.output_nc = 3
                    self.ngf = 64
                    self.ndf = 64
                    self.netG = 'unet_256'
                    self.netD = 'basic'
                    self.n_layers_D = 3
                    self.norm = 'batch'
                    self.init_type = 'normal'
                    self.init_gain = 0.02
                    self.no_dropout = True
                    self.verbose = False
                    self.suffix = ''
                    self.max_dataset_size = float('inf')
                    self.num_threads = 0
            
            # Create model with the complete options
            self.opt = MockArgs()
            self.model = create_model(self.opt)
            self.model.setup(self.opt)
            
            print("✅ Model loaded successfully!")
            print(f"📁 Model: {self.opt.name}")
            print(f"🎯 Direction: {self.opt.direction} (Sketch -> Aerial)")
            print(f"📱 Device: {self.opt.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def extract_image_from_editor_data(self, editor_data):
        """Extract PIL Image from Gradio ImageEditor output"""
        try:
            if editor_data is None:
                return Image.new('RGB', (256, 256), 'white')
                
            if isinstance(editor_data, dict):
                if 'image' in editor_data and editor_data['image'] is not None:
                    image_data = editor_data['image']
                    
                    if isinstance(image_data, str):
                        if image_data.startswith('data:'):
                            base64_data = image_data.split(',')[1]
                        else:
                            base64_data = image_data
                            
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_bytes))
                        return image
                    else:
                        return image_data
                else:
                    return Image.new('RGB', (256, 256), 'white')
            else:
                return editor_data
                
        except Exception as e:
            print(f"Error extracting image: {e}")
            return Image.new('RGB', (256, 256), 'white')
    
    def preprocess_sketch_for_model(self, sketch_image):
        """Convert sketch to the exact format expected by the model"""
        # Resize to model input size
        sketch = sketch_image.resize((256, 256))
        
        # Convert to RGB if needed
        if sketch.mode != 'RGB':
            sketch = sketch.convert('RGB')
        
        # Convert to tensor and normalize to [-1, 1] like the dataset
        sketch_array = np.array(sketch).transpose(2, 0, 1)  # HWC to CHW
        sketch_tensor = torch.from_numpy(sketch_array).float()
        sketch_tensor = (sketch_tensor / 255.0) * 2 - 1  # Normalize to [-1, 1]
        
        return sketch_tensor.unsqueeze(0).to(self.device)  # Add batch dimension and send to device
    
    def tensor_to_image(self, tensor):
        """Convert model output tensor back to PIL Image"""
        # Denormalize from [-1, 1] to [0, 255]
        image = tensor.squeeze(0).cpu().float().numpy()
        image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)
    
    def generate_aerial(self, sketch_data):
        """Generate aerial view from sketch using the trained model"""
        try:
            if self.model is None:
                return self.create_error_image("Model not loaded. Please check console for errors.")
            
            # Extract the actual image from the editor data
            sketch_image = self.extract_image_from_editor_data(sketch_data)
            
            # Preprocess sketch for model input
            sketch_tensor = self.preprocess_sketch_for_model(sketch_image)
            
            print("🎨 Running model inference...")
            
            # Run inference - same as test.py
            with torch.no_grad():
                # For BtoA direction, we need to provide both A and B
                # B is our input sketch, A is a dummy tensor (since we're in BtoA mode)
                dummy_A = torch.zeros_like(sketch_tensor)  # Create dummy A tensor
                
                # Set model input with both A and B
                self.model.set_input({
                    'A': dummy_A,  # Dummy input for A
                    'B': sketch_tensor,  # Our actual sketch input
                    'A_paths': ['dummy'],
                    'B_paths': ['sketch_input']
                })
                
                # Run forward pass
                self.model.test()
                
                # Get generated images
                visuals = self.model.get_current_visuals()
                
                print(f"📊 Available visuals: {list(visuals.keys())}")  # Debug: see what outputs are available
                
                # Try to get the generated image - check different possible keys
                if 'fake_B' in visuals:
                    result = visuals['fake_B']
                    print("✅ Using fake_B as output")
                elif 'fake' in visuals:
                    result = visuals['fake']
                    print("✅ Using fake as output")
                elif 'fake_A' in visuals:
                    result = visuals['fake_A']
                    print("✅ Using fake_A as output")
                else:
                    # If no fake output found, use the first available tensor
                    available_keys = list(visuals.keys())
                    if available_keys:
                        result = visuals[available_keys[0]]
                        print(f"⚠️ Using {available_keys[0]} as output (fallback)")
                    else:
                        raise KeyError("No output images found in visuals")
                
                # Convert to PIL Image
                aerial_image = self.tensor_to_image(result)
            
            print("✅ Inference complete!")
            return aerial_image
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return self.create_error_image(f"Model error: {str(e)}")
    
    def create_error_image(self, message):
        """Create an error image with message"""
        img = Image.new('RGB', (256, 256), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Split long messages
        if len(message) > 30:
            parts = [message[i:i+30] for i in range(0, len(message), 30)]
            for i, part in enumerate(parts):
                draw.text((10, 110 + i*20), part, fill='red')
        else:
            draw.text((10, 120), message, fill='red')
        return img

# Create the interface
def create_drawing_interface():
    generator = SketchToMapGenerator()
    
    # Color guide
    color_guide = """
    **🎨 Color Guide:**
    - ⚪ **WHITE**: Roads & Paths
    - 🟢 **GREEN**: Vegetation & Parks  
    - 🔵 **BLUE**: Water & Rivers
    - ⚫ **BLACK**: Buildings & Structures
    - 🟡 **YELLOW**: Special Areas
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Sketch to Aerial Map") as interface:
        gr.Markdown("# 🗺️ Sketch to Realistic Aerial Generator")
        gr.Markdown("Draw your map sketch and watch the AI transform it into a realistic aerial view!")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎨 Draw Your Map Sketch")
                gr.Markdown(color_guide)
                
                sketch_input = gr.ImageEditor(
                    type="numpy",
                    height=400,
                    width=400,
                    brush=gr.Brush(
                        colors=["#FFFFFF", "#00FF00", "#0000FF", "#000000", "#FFFF00"],
                        color_mode="fixed",
                    ),
                    label="Draw your map sketch",
                    show_download_button=False
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate Aerial View", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear Canvas", variant="secondary")
            
            with gr.Column():
                gr.Markdown("### 🛰️ AI-Generated Aerial View")
                output_image = gr.Image(
                    type="pil",
                    height=400,
                    width=400,
                    label="Realistic Aerial Output",
                    show_download_button=True
                )
        
        # Status indicator
        status_text = gr.Markdown("")
        
        # Update status based on model loading
        if generator.model is not None:
            status_text.value = "### ✅ Status: Model loaded and ready!"
        else:
            status_text.value = "### ❌ Status: Model failed to load. Check console for errors."
        
        # Button actions
        generate_btn.click(
            fn=generator.generate_aerial,
            inputs=sketch_input,
            outputs=output_image
        )
        
        def clear_canvas():
            return None
            
        clear_btn.click(
            fn=clear_canvas,
            inputs=[],
            outputs=sketch_input
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Sketch to Aerial Map Generator")
    print("📍 Server: http://127.0.0.1:7860")
    print("🤖 Using trained pix2pix model: maps_pix2pix_SketchToReal")
    print("🎯 Direction: Sketch (B) → Aerial (A)")
    
    interface = create_drawing_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )