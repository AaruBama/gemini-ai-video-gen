from google import genai
from google.genai import types
import PIL.Image
import io
import os
import time
import gradio as gr
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ImageService:
    """Handles image generation"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    def generate_image(self, source_img, scenic_img, prompt, model, aspect_ratio, progress=gr.Progress()):
        """Generate fused image"""
        logger.info("=" * 60)
        logger.info("STEP 2: IMAGE FUSION STARTED")
        logger.info("=" * 60)
        
        if source_img is None or scenic_img is None:
            return None, None, "❌ Error: Both images required"
        
        if not prompt or "Error" in prompt or "❌" in prompt:
            return None, None, "❌ Error: Valid prompt required"
        
        try:
            progress(0.2, desc="🎨 Preparing fusion...")
            logger.info(f"📤 Using model: {model}")
            logger.info(f"📐 Aspect ratio: {aspect_ratio}")
            
            response = self.client.models.generate_content(
                model=model,
                contents=[prompt, source_img, scenic_img],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                )
            )
            
            progress(0.7, desc="🖼️ Processing image...")
            
            for part in response.parts:
                if part.inline_data is not None:
                    pil_image = PIL.Image.open(io.BytesIO(part.inline_data.data))
                    
                    # Save to images directory
                    timestamp = int(time.time())
                    filename = f"fusion_{timestamp}.png"
                    save_path = os.path.join(Config.IMAGES_DIR, filename)
                    pil_image.save(save_path)
                    
                    logger.info(f"💾 Saved: {save_path}")
                    logger.info("✅ STEP 2 COMPLETE")
                    progress(1.0, desc="✅ Image ready!")
                    
                    return pil_image, save_path, f"✅ Saved to {save_path}"
            
            return None, None, "❌ Error: No image data in response"
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg