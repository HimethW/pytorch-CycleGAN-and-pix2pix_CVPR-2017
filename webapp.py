import gradio as gr
import torch
from PIL import Image
import numpy as np
import io

class SketchToMapGenerator:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        # You'll need to adapt this based on your model loading
        from models import create_model
        from options.test_options import TestOptions
        
        opt = TestOptions().parse()
        opt.model = 'pix2pix'
        opt.name = 'maps_pix2pix_SketchToReal'  # Your new model name
        opt.isTrain = False
        opt.no_dropout = True
        opt.preprocess = 'none'
        
        model = create_model(opt)
        model.setup(opt)
        return model
    
    def sketch_to_map(self, sketch_image):
        # Preprocess sketch
        sketch = sketch_image.resize((256, 256))
        sketch_array = np.array(sketch)
        
        # Convert to tensor (you may need to adjust this based on your model)
        sketch_tensor = torch.from_numpy(sketch_array).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Generate map
        with torch.no_grad():
            self.model.set_input({'B': sketch_tensor, 'B_paths': ['sketch']})
            self.model.test()
            visuals = self.model.get_current_visuals()
            fake_A = visuals['fake_A']  # This should be the realistic output
        
        # Convert back to image
        result = (fake_A[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(result)

# Create interface
def create_interface():
    generator = SketchToMapGenerator('checkpoints/maps_pix2pix_SketchToReal')
    
    def process_sketch(sketch):
        if sketch is None:
            return None
        return generator.sketch_to_map(sketch)
    
    iface = gr.Interface(
        fn=process_sketch,
        inputs=gr.Sketchpad(
            shape=(256, 256),
            brush_radius=8,
            brush_color="#000000",
            background_color="#FFFFFF",
            label="Draw your map sketch (rivers, roads, buildings)"
        ),
        outputs=gr.Image(
            type="pil",
            label="Generated Realistic Map"
        ),
        title="🎨 Sketch to Realistic Map Generator",
        description="""**Draw a simple map sketch and watch it become a realistic aerial view!**
        
        Drawing tips:
        - Use black lines for roads/rivers
        - Draw rectangles for buildings  
        - Create blobs for forests/parks
        - The AI will fill in realistic details!""",
        examples=[
            ["examples/simple_river_sketch.png"],
            ["examples/city_grid_sketch.png"],
            ["examples/park_area_sketch.png"]
        ]
    )
    return iface

# Launch app
if __name__ == "__main__":
    create_interface().launch(share=True)  # share=True gives you a public link