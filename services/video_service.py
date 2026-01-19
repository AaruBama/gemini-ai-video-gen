from google import genai
from google.genai import types
import time
import os
import gradio as gr
from utils.logger import setup_logger
from config import Config

logger = setup_logger(__name__)

class VideoService:
    """Handles video generation"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    def generate_video(self, mode, image_path, ref_images, first_frame, last_frame, 
                      prompt, model, aspect_ratio, progress=gr.Progress()):
        """Generate video based on selected mode"""
        logger.info("=" * 60)
        logger.info(f"STEP 3: VIDEO GENERATION - {mode.upper()} MODE")
        logger.info("=" * 60)
        
        if not prompt or "Error" in prompt or "❌" in prompt:
            return None, "❌ Error: Valid prompt required"
        
        try:
            progress(0.1, desc="📤 Preparing...")
            
            # Build video generation config
            if mode == "default":
                result = self._generate_default(image_path, prompt, model, aspect_ratio, progress)
            elif mode == "reference":
                result = self._generate_with_references(ref_images, prompt, model, aspect_ratio, progress)
            elif mode == "interpolation":
                result = self._generate_interpolation(first_frame, last_frame, prompt, model, aspect_ratio, progress)
            else:
                return None, "❌ Error: Invalid mode"
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    def _generate_default(self, image_input, prompt, model, aspect_ratio, progress):
        """Default mode: single image to video"""
        if image_input is None:
            return None, "❌ Error: Image not found"
        
        import PIL.Image
        if isinstance(image_input, PIL.Image.Image):
            # Convert PIL Image to bytes
            logger.info("✓ Converting uploaded PIL Image to bytes")
            import io
            img_byte_arr = io.BytesIO()
            image_input.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
        elif isinstance(image_input, str) and os.path.exists(image_input):
            # Read from file path
            logger.info(f"✓ Reading image from path: {image_input}")
            with open(image_input, 'rb') as f:
                image_bytes = f.read()
        else:
            return None, "❌ Error: Invalid image input"
        
        progress(0.2, desc="🎬 Starting video generation...")
        
        operation = self.client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=types.Image(image_bytes=image_bytes, mime_type="image/png"),
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution="720p",
                duration_seconds=4
            )
        )
        
        return self._poll_and_save(operation, progress)
    
    def _generate_with_references(self, ref_images, prompt, model, aspect_ratio, progress):
        """Reference images mode"""
        if not ref_images or len(ref_images) == 0:
            return None, "❌ Error: Upload 1-3 reference images"
        
        progress(0.2, desc="🎬 Preparing reference images...")
        
        reference_image_objects = []
        import PIL.Image
        import io
        
        for idx, img in enumerate(ref_images[:3]):  # Max 3 images
            if img is not None:
                try:
                    # Handle both PIL Images and file paths
                    if isinstance(img, PIL.Image.Image):
                        # Already a PIL Image
                        logger.info(f"✓ Reference {idx+1}: PIL Image")
                        pil_img = img
                    elif isinstance(img, str):
                        # File path - load as PIL Image
                        logger.info(f"✓ Reference {idx+1}: Loading from {img}")
                        pil_img = PIL.Image.open(img)
                    else:
                        logger.warning(f"⚠️ Reference {idx+1}: Unknown type {type(img)}, skipping")
                        continue
                    
                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    logger.info(f"  Converted to {len(img_bytes)} bytes")
                    
                    ref_img = types.VideoGenerationReferenceImage(
                        image=types.Image(image_bytes=img_bytes, mime_type="image/png"),
                        reference_type="asset"
                    )
                    reference_image_objects.append(ref_img)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing reference image {idx+1}: {e}")
                    continue
        
        if len(reference_image_objects) == 0:
            return None, "❌ Error: No valid reference images could be processed"
        
        logger.info(f"✓ Using {len(reference_image_objects)} reference images")
        
        operation = self.client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                reference_images=reference_image_objects,
                aspect_ratio=aspect_ratio,
                resolution="720p",
                duration_seconds=5
            )
        )
        
        return self._poll_and_save(operation, progress)
    
    def _generate_interpolation(self, first_frame, last_frame, prompt, model, aspect_ratio, progress):
        """First and last frame mode"""
        if first_frame is None or last_frame is None:
            return None, "❌ Error: Upload both first and last frames"
        
        progress(0.2, desc="🎬 Preparing interpolation...")
        
        # Convert first frame
        import io
        first_byte_arr = io.BytesIO()
        first_frame.save(first_byte_arr, format='PNG')
        first_bytes = first_byte_arr.getvalue()
        
        # Convert last frame
        last_byte_arr = io.BytesIO()
        last_frame.save(last_byte_arr, format='PNG')
        last_bytes = last_byte_arr.getvalue()
        
        operation = self.client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=types.Image(image_bytes=first_bytes, mime_type="image/png"),
            config=types.GenerateVideosConfig(
                last_frame=types.Image(image_bytes=last_bytes, mime_type="image/png"),
                aspect_ratio=aspect_ratio,
                resolution="720p",
                duration_seconds=5
            )
        )
        
        return self._poll_and_save(operation, progress)
    
    def _poll_and_save(self, operation, progress):
        """Poll operation and save video"""
        max_wait_time = 300
        start_time = time.time()
        poll_count = 0
        
        logger.info("⏳ Waiting for video generation...")
        
        while not operation.done:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                return None, f"❌ Timeout (exceeded {max_wait_time//60} min)"
            
            progress_pct = min(0.2 + (elapsed / max_wait_time) * 0.7, 0.9)
            progress(progress_pct, desc=f"🎬 Rendering... ({int(elapsed)}s / ~180s)")
            
            poll_count += 1
            logger.info(f"Poll #{poll_count}: {elapsed:.0f}s elapsed...")
            time.sleep(10)
            
            operation = self.client.operations.get(operation)
        
        logger.info(f"✓ Complete! Time: {time.time() - start_time:.0f}s")
        progress(0.95, desc="💾 Downloading...")
        
        if operation.response and hasattr(operation.response, 'generated_videos'):
            generated_video = operation.response.generated_videos[0]
            
            self.client.files.download(file=generated_video.video)
            video_bytes = generated_video.video.video_bytes
            
            timestamp = int(time.time())
            filename = f"video_{timestamp}.mp4"
            save_path = os.path.join(Config.VIDEOS_DIR, filename)
            
            with open(save_path, 'wb') as f:
                f.write(video_bytes)
            
            file_size_mb = len(video_bytes) / (1024 * 1024)
            logger.info(f"💾 Saved: {save_path} ({file_size_mb:.2f} MB)")
            logger.info("✅ STEP 3 COMPLETE")
            
            progress(1.0, desc="✅ Video complete!")
            return save_path, f"✅ Saved ({file_size_mb:.2f} MB)"
        else:
            return None, "❌ Error: No video in response"
