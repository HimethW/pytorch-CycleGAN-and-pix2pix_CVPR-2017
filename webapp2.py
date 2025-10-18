import gradio as gr
import torch
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import base64
from io import BytesIO

class SketchToMapGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_image_from_editor_data(self, editor_data):
        """Extract PIL Image from Gradio ImageEditor output"""
        try:
            if editor_data is None:
                # Return a blank white image
                return Image.new('RGB', (400, 400), 'white')
                
            if isinstance(editor_data, dict):
                # Gradio ImageEditor returns a dict with 'image' and 'mask'
                if 'image' in editor_data and editor_data['image'] is not None:
                    image_data = editor_data['image']
                    
                    if isinstance(image_data, str):
                        # Base64 string (data:image/png;base64,...)
                        if image_data.startswith('data:'):
                            base64_data = image_data.split(',')[1]
                        else:
                            base64_data = image_data
                            
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_bytes))
                        return image
                    else:
                        # Already a PIL Image or array
                        return image_data
                else:
                    # No image data, return blank
                    return Image.new('RGB', (400, 400), 'white')
            else:
                # Direct image input
                return editor_data
                
        except Exception as e:
            print(f"Error extracting image: {e}")
            return Image.new('RGB', (400, 400), 'white')
    
    def generate_aerial(self, sketch_data):
        """Generate aerial view from sketch data"""
        try:
            # Extract the actual image from the editor data
            sketch_image = self.extract_image_from_editor_data(sketch_data)
            
            # Resize to model input size
            sketch_image = sketch_image.resize((256, 256))
            
            # Generate aerial view based on colors
            aerial_image = self.create_aerial_from_sketch(sketch_image)
            
            return aerial_image
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self.create_error_image(f"Error: {str(e)}")
    
    def create_aerial_from_sketch(self, sketch):
        """Create aerial view based on sketch colors"""
        # Create base aerial image (sky blue background)
        aerial = Image.new('RGB', (256, 256), color=(135, 206, 235))
        draw = ImageDraw.Draw(aerial)
        
        # Convert sketch to numpy array for processing
        sketch_array = np.array(sketch)
        
        # Process different colors
        self.add_terrain_from_color(draw, sketch_array, [255, 255, 255], (100, 100, 100), 'road')      # White -> Gray roads
        self.add_terrain_from_color(draw, sketch_array, [0, 255, 0], (34, 139, 34), 'vegetation')      # Green -> Dark green vegetation
        self.add_terrain_from_color(draw, sketch_array, [0, 0, 255], (65, 105, 225), 'water')          # Blue -> Royal blue water
        self.add_terrain_from_color(draw, sketch_array, [0, 0, 0], (169, 169, 169), 'building')        # Black -> Gray buildings
        self.add_terrain_from_color(draw, sketch_array, [255, 255, 0], (255, 215, 0), 'special')       # Yellow -> Gold special areas
        
        return aerial
    
    def add_terrain_from_color(self, draw, sketch_array, target_color, output_color, terrain_type):
        """Add terrain features based on color detection"""
        try:
            # Create mask for target color (with some tolerance)
            tolerance = 10
            mask = np.all(np.abs(sketch_array - target_color) <= tolerance, axis=-1)
            
            if not np.any(mask):
                return
            
            # Get coordinates of the target color
            coords = np.argwhere(mask)
            
            # Different rendering based on terrain type
            if terrain_type == 'road':
                # Draw roads as lines
                for y, x in coords[::10]:
                    draw.rectangle([x-2, y-1, x+2, y+1], fill=output_color)
            
            elif terrain_type == 'vegetation':
                # Draw vegetation as scattered green areas
                for y, x in coords[::6]:
                    size = np.random.randint(2, 5)
                    draw.ellipse([x-size, y-size, x+size, y+size], fill=output_color)
            
            elif terrain_type == 'water':
                # Draw water as connected blue areas
                for y, x in coords[::4]:
                    size = np.random.randint(3, 7)
                    draw.ellipse([x-size, y-size, x+size, y+size], fill=output_color)
            
            elif terrain_type == 'building':
                # Draw buildings as rectangles
                for y, x in coords[::8]:
                    height = np.random.randint(4, 8)
                    draw.rectangle([x-2, y-height, x+2, y], fill=output_color)
            
            elif terrain_type == 'special':
                # Draw special areas as gold circles
                for y, x in coords[::6]:
                    draw.ellipse([x-3, y-3, x+3, y+3], fill=output_color)
                    
        except Exception as e:
            print(f"Error adding {terrain_type}: {e}")
    
    def create_error_image(self, message):
        """Create an error image with message"""
        img = Image.new('RGB', (256, 256), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        # Simple text display
        draw.text((50, 120), message, fill='red')
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
    
    **✏️ How to use:**
    1. Select a color from the palette
    2. Draw your map features
    3. Click 'Generate Aerial View'
    4. See your sketch transformed!
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Sketch to Aerial Map") as interface:
        gr.Markdown("# 🗺️ Sketch to Aerial Map Generator")
        gr.Markdown("Draw your map using colors and generate a realistic aerial view!")
        
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
                gr.Markdown("### 🛰️ Generated Aerial View")
                output_image = gr.Image(
                    type="pil",
                    height=400,
                    width=400,
                    label="Aerial Map Output",
                    show_download_button=True
                )
        
        # Instructions
        gr.Markdown("""
        ### 📝 Instructions:
        1. **Choose a color** from the palette above the canvas
        2. **Draw** your map features (roads, rivers, buildings, etc.)
        3. **Click Generate** to see the aerial view
        4. **Use different colors** for different terrain types
        
        ### 🎯 Tips:
        - Draw thicker lines for better detection
        - Use white for roads and paths
        - Use blue for rivers and lakes
        - Use green for parks and forests
        - Use black for buildings
        """)
        
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
    print("📍 Server will start at: http://127.0.0.1:7860")
    print("🎨 Draw with colors and see instant aerial views!")
    
    interface = create_drawing_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True  # Enable debug mode for more info
    )