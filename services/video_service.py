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
    
    def generate_video(self, mode, image_path, ref_images, first_frame, last_frame, extension_video,
                      prompt, model, aspect_ratio, duration, progress=gr.Progress()):
        """Generate video based on selected mode"""
        logger.info("=" * 60)
        logger.info(f"STEP 3: VIDEO GENERATION - {mode.upper()} MODE")
        logger.info("=" * 60)
        
        if not prompt or "Error" in prompt or "❌" in prompt:
            return None, "❌ Error: Valid prompt required"
        
        # Validate and sanitize duration for Veo 3.1 (only accepts 4, 6, or 8 seconds)
        valid_durations = [4, 6, 8]
        try:
            duration = int(duration)
            if duration not in valid_durations:
                # Clamp to nearest valid duration
                if duration < 4:
                    duration = 4
                elif duration == 5:
                    duration = 6
                elif duration == 7:
                    duration = 8
                elif duration > 8:
                    duration = 8
                logger.warning(f"Duration adjusted to nearest valid value: {duration} seconds (Veo 3.1 only accepts 4, 6, or 8)")
        except (TypeError, ValueError):
            logger.warning(f"Invalid duration type: {type(duration)}, defaulting to 8")
            duration = 8
        
        logger.info(f"Using duration: {duration} seconds")
        
        try:
            progress(0.1, desc="📤 Preparing...")
            
            # Build video generation config
            if mode == "default":
                result = self._generate_default(image_path, prompt, model, aspect_ratio, duration, progress)
            elif mode == "reference":
                result = self._generate_with_references(ref_images, prompt, model, aspect_ratio, duration, progress)
            elif mode == "interpolation":
                result = self._generate_interpolation(first_frame, last_frame, prompt, model, aspect_ratio, duration, progress)
            elif mode == "text_to_video":
                result = self._generate_text_to_video(prompt, model, aspect_ratio, duration, progress)
            elif mode == "video_extension":
                result = self._generate_video_extension(extension_video, prompt, model, aspect_ratio, duration, progress)
            else:
                return None, "❌ Error: Invalid mode"
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    def _generate_default(self, image_input, prompt, model, aspect_ratio, duration, progress):
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
                duration_seconds=duration
            )
        )
        
        return self._poll_and_save(operation, progress)
    
    def _generate_with_references(self, ref_images, prompt, model, aspect_ratio, duration, progress):
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
                    # Gradio Gallery returns tuples of (image, caption)
                    if isinstance(img, tuple):
                        logger.info(f"✓ Reference {idx+1}: Extracting from Gradio Gallery tuple")
                        img = img[0]  # Extract the image from the tuple
                    
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
        
        # Debug logging for duration
        logger.info(f"DEBUG: duration value = {duration}, type = {type(duration)}")
        logger.info(f"DEBUG: Creating GenerateVideosConfig with duration_seconds={duration}")
        
        operation = self.client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                reference_images=reference_image_objects,
                aspect_ratio=aspect_ratio,
                resolution="720p",
                duration_seconds=duration
            )
        )
        
        return self._poll_and_save(operation, progress)
    
    def _generate_interpolation(self, first_frame, last_frame, prompt, model, aspect_ratio, duration, progress):
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
                duration_seconds=duration
            )
        )
        
        return self._poll_and_save(operation, progress)
    
    def _generate_text_to_video(self, prompt, model, aspect_ratio, duration, progress):
        """Text-to-video mode: generate video from prompt only"""
        if not prompt or "Error" in prompt or "❌" in prompt:
            return None, "❌ Error: Valid prompt required for text-to-video"
        
        logger.info("✓ Generating video from text prompt only")
        progress(0.2, desc="🎬 Starting text-to-video generation...")
        
        operation = self.client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution="720p",
                duration_seconds=duration
            )
        )
        
        return self._poll_and_save(operation, progress)
    
    def _generate_video_extension(self, video_input, prompt, model, aspect_ratio, duration, progress):
        """Video extension mode: extend an existing Veo-generated video
        
        IMPORTANT: Video extension only works with videos that were previously generated by Veo.
        You cannot extend arbitrary uploaded videos - they must be from a prior Veo generation.
        """
        if video_input is None:
            return None, "❌ Error: Upload a Veo-generated video to extend"
        
        if not prompt or "Error" in prompt or "❌" in prompt:
            return None, "❌ Error: Valid prompt required for video extension"
        
        logger.info("✓ Video extension mode")
        progress(0.1, desc="📤 Uploading video...")
        
        try:
            # Handle video input - can be file path string or file object
            if isinstance(video_input, str):
                # File path
                logger.info(f"✓ Reading video from path: {video_input}")
                video_path = video_input
            else:
                # Assume it's a file-like object from Gradio
                logger.info("✓ Processing uploaded video file")
                video_path = video_input
            
            progress(0.2, desc="🎬 Checking video metadata...")
            
            # Check if this is a Veo-generated video with saved URI metadata
            video_uri = None
            if isinstance(video_path, str) and video_path.endswith('.mp4'):
                # Check for .uri metadata file
                uri_file = video_path.replace('.mp4', '.uri')
                if os.path.exists(uri_file):
                    with open(uri_file, 'r') as f:
                        video_uri = f.read().strip()
                    logger.info(f"✓ Found saved video URI: {video_uri}")
            
            if video_uri:
                # Use the saved URI directly (no upload needed)
                video_obj = types.Video(uri=video_uri)
                logger.info("✓ Using saved Veo video URI for extension")
            else:
                # Fallback: try uploading (will likely fail with API error)
                logger.warning("⚠️  No video URI metadata found - attempting upload (may fail)")
                logger.info(f"📤 Uploading video file to Gemini API...")
                uploaded_file = self.client.files.upload(file=video_path)
                logger.info(f"✓ Video uploaded: {uploaded_file.name}")
                video_obj = types.Video(uri=uploaded_file.uri)
            
            progress(0.3, desc="🎬 Starting video extension...")
            
            # Generate extended video
            # Based on official API example
            operation = self.client.models.generate_videos(
                model=model,
                prompt=prompt,
                video=video_obj,
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    resolution="720p"
                )
            )
            
            return self._poll_and_save(operation, progress)
            
        except Exception as e:
            error_msg = f"❌ Error uploading/processing video: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
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
        
        # Enhanced error checking
        if not hasattr(operation, 'response') or operation.response is None:
            logger.error("❌ Operation has no response")
            return None, "❌ Error: Operation completed but no response received"
        
        if not hasattr(operation.response, 'generated_videos'):
            logger.error(f"❌ Response structure unexpected: {dir(operation.response)}")
            return None, "❌ Error: Unexpected response structure"
        
        if operation.response.generated_videos is None or len(operation.response.generated_videos) == 0:
            logger.error("❌ No generated videos in response")
            
            # Check for error information
            if hasattr(operation, 'error') and operation.error:
                error_msg = f"❌ Generation failed: {operation.error}"
                logger.error(error_msg)
                return None, error_msg
            
            return None, "❌ Error: No video generated (possible content filter)"
        
        try:
            generated_video = operation.response.generated_videos[0]
            logger.info(f"✓ Video object received: {type(generated_video)}")
            
            # Download video
            logger.info("📥 Downloading video bytes...")
            self.client.files.download(file=generated_video.video)
            
            if not hasattr(generated_video.video, 'video_bytes'):
                logger.error("❌ Video object has no video_bytes attribute")
                return None, "❌ Error: Could not access video data"
            
            video_bytes = generated_video.video.video_bytes
            
            if not video_bytes or len(video_bytes) == 0:
                logger.error("❌ Video bytes are empty")
                return None, "❌ Error: Downloaded video is empty"
            
            logger.info(f"✓ Downloaded {len(video_bytes)} bytes")
            
            # Save to file
            timestamp = int(time.time())
            filename = f"video_{timestamp}.mp4"
            save_path = os.path.join(Config.VIDEOS_DIR, filename)
            
            with open(save_path, 'wb') as f:
                f.write(video_bytes)
            
            # Save video URI as metadata for future extension
            if hasattr(generated_video.video, 'uri') and generated_video.video.uri:
                metadata_path = save_path.replace('.mp4', '.uri')
                with open(metadata_path, 'w') as f:
                    f.write(generated_video.video.uri)
                logger.info(f"💾 Saved video URI metadata: {metadata_path}")
            
            file_size_mb = len(video_bytes) / (1024 * 1024)
            logger.info(f"💾 Saved: {save_path} ({file_size_mb:.2f} MB)")
            logger.info("✅ STEP 3 COMPLETE")
            
            progress(1.0, desc="✅ Video complete!")
            return save_path, f"✅ Saved ({file_size_mb:.2f} MB)"
            
        except AttributeError as e:
            logger.error(f"❌ Attribute error accessing video: {e}")
            logger.error(f"Operation response structure: {dir(operation.response)}")
            return None, f"❌ Error accessing video data: {str(e)}"
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            return None, f"❌ Error: {str(e)}"