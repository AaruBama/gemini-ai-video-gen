import gradio as gr
from google import genai
from google.genai import types
import PIL.Image
import time
import os
import io
import logging

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY environment variable not set. Please set it before running.")

client = genai.Client(api_key=API_KEY)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"✓ Output directory created: {OUTPUT_DIR}")

# --- PHASE 1: PROMPT GENERATION ---
def generate_prompts(source_img, scenic_img, progress=gr.Progress()):
    """Phase 1: AI analyzes images and generates prompts."""
    logger.info("=" * 60)
    logger.info("STEP 1: PROMPT GENERATION STARTED")
    logger.info("=" * 60)
    
    if source_img is None or scenic_img is None:
        logger.error("❌ Missing images - both source and scenic required")
        return "❌ Error: Both images required", "❌ Error: Both images required"
    
    logger.info(f"✓ Source image type: {type(source_img)}")
    logger.info(f"✓ Scenic image type: {type(scenic_img)}")
    progress(0.1, desc="🔍 Analyzing images...")
    
    system_instruction = """
You are an AI Video Director. Create two prompts for image fusion and video animation.

TASK:
1. Analyze the 'Source Character' and 'Scenic Background' images.
2. Write 'Fusion Prompt': Describe how to merge the person from the source into the scenery.
   - Be specific about character details and background elements.
   - Choose an appropriate outfit for the character based on scene and background
   - Mention lighting and atmosphere matching.
   - Make skin hyper-realistic and not ultra-smooth.
   - Add negative prompts
3. Write 'Motion Prompt': Describe how the character should move in the video.
   - Include camera movement, character actions, and background activity
   - Add negative prompts

OUTPUT FORMAT:
Return ONLY two prompts separated by a pipe (|).
Format: <Fusion Prompt>|<Motion Prompt>
No labels, markdown, or extra text.
"""
    
    try:
        logger.info("📤 Sending request to Gemini 2.0 Flash...")
        progress(0.3, desc="🤖 Gemini analyzing images...")
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[system_instruction, source_img, scenic_img]
        )
        
        logger.info(f"✓ Response received - length: {len(response.text)} chars")
        progress(0.8, desc="📝 Processing AI response...")
        
        raw_text = response.text.strip()
        logger.info(f"Raw response preview: {raw_text[:200]}...")
        
        # Remove markdown if present
        if raw_text.startswith('```'):
            logger.info("⚠️  Removing markdown formatting...")
            lines = raw_text.split('\n')
            raw_text = '\n'.join(line for line in lines if not line.startswith('```'))
            raw_text = raw_text.strip()
        
        if "|" in raw_text:
            parts = raw_text.split('|', 1)
            nano_prompt = parts[0].strip()
            veo_prompt = parts[1].strip()
            
            logger.info("✅ STEP 1 COMPLETE - Prompts generated successfully")
            logger.info(f"   Fusion prompt: {nano_prompt[:100]}...")
            logger.info(f"   Motion prompt: {veo_prompt[:100]}...")
            progress(1.0, desc="✅ Prompts generated!")
            
            return nano_prompt, veo_prompt
        else:
            logger.warning("⚠️  Pipe separator not found - using fallback")
            progress(1.0, desc="⚠️  Using fallback prompts")
            return raw_text, "The character moves naturally with subtle, realistic motions."
            
    except Exception as e:
        error_msg = f"❌ Error generating prompts: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, error_msg

# --- PHASE 2: IMAGE FUSION ---
def generate_anchor(source_img, scenic_img, nano_prompt, progress=gr.Progress()):
    """Phase 2: Fuses character into background using Gemini 2.5 Flash Image."""
    logger.info("=" * 60)
    logger.info("STEP 2: IMAGE FUSION STARTED")
    logger.info("=" * 60)
    
    if source_img is None or scenic_img is None:
        logger.error("❌ Missing images")
        return None, None, "❌ Error: Both images required"
    
    if not nano_prompt or "Error" in nano_prompt or "❌" in nano_prompt:
        logger.error("❌ Invalid fusion prompt")
        return None, None, "❌ Error: Valid fusion prompt required. Please complete Step 1 first."
    
    try:
        logger.info(f"✓ Using prompt: {nano_prompt[:150]}...")
        progress(0.2, desc="🎨 Preparing image fusion...")
        
        logger.info("📤 Calling Gemini 2.5 Flash Image (Nano Banana)...")
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[nano_prompt, source_img, scenic_img],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )
        )
        
        logger.info(f"✓ Response received - parts count: {len(response.parts)}")
        progress(0.7, desc="🖼️  Processing generated image...")
        
        for i, part in enumerate(response.parts):
            logger.info(f"  Part {i}: type={type(part)}, has inline_data={part.inline_data is not None}")
            
            if part.inline_data is not None:
                logger.info(f"  ✓ Found image data - size: {len(part.inline_data.data)} bytes")
                
                # Convert to PIL Image
                pil_image = PIL.Image.open(io.BytesIO(part.inline_data.data))
                logger.info(f"  ✓ PIL Image created - size: {pil_image.size}, mode: {pil_image.mode}")
                
                # Save locally
                timestamp = int(time.time())
                anchor_path = os.path.join(OUTPUT_DIR, f"anchor_frame_{timestamp}.png")
                pil_image.save(anchor_path)
                logger.info(f"  💾 Saved to: {anchor_path}")
                
                logger.info("✅ STEP 2 COMPLETE - Anchor frame generated")
                progress(1.0, desc="✅ Anchor frame ready!")
                
                return pil_image, anchor_path, f"✅ Saved to {anchor_path}"
        
        logger.error("❌ No image data found in response parts")
        return None, None, "❌ Error: No image data in response"
        
    except Exception as e:
        error_msg = f"❌ Error in image fusion: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, None, error_msg

# --- PHASE 3: VIDEO GENERATION ---
def generate_video(anchor_path, veo_prompt, progress=gr.Progress()):
    """Phase 3: Animates anchor image using Veo 3.1."""
    logger.info("=" * 60)
    logger.info("STEP 3: VIDEO GENERATION STARTED")
    logger.info("=" * 60)
    
    if not anchor_path or not os.path.exists(anchor_path):
        logger.error("❌ Anchor frame not found")
        return None, "❌ Error: Anchor frame not found. Complete Step 2 first."
    
    if not veo_prompt or "Error" in veo_prompt or "❌" in veo_prompt:
        logger.error("❌ Invalid motion prompt")
        return None, "❌ Error: Valid motion prompt required"
    
    try:
        progress(0.1, desc="📤 Preparing anchor image...")
        
        with open(anchor_path, 'rb') as f:
            image_bytes = f.read()

        
        progress(0.2, desc="🎬 Starting Veo video generation...")
        logger.info("📤 Calling Veo 3.1 Generate Preview...")
        
        # Pass the Image object directly
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=veo_prompt,
            image=types.Image(
                image_bytes=image_bytes, # API specific field
                mime_type="image/png"
            ),
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                resolution="720p",
                duration_seconds=8
            )
        )
        
        
        logger.info(f"✓ Operation started - name: {operation.name}")
        logger.info(f"  Done: {operation.done}")
        
        # Poll for completion with timeout
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        poll_count = 0
        
        logger.info("⏳ Waiting for video generation (this takes 1-3 minutes)...")
        
        while not operation.done:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"❌ Timeout after {elapsed:.0f} seconds")
                return None, f"❌ Error: Video generation timeout (exceeded {max_wait_time//60} minutes)"
            
            progress_pct = min(0.2 + (elapsed / max_wait_time) * 0.7, 0.9)
            progress(progress_pct, desc=f"🎬 Rendering video... ({int(elapsed)}s / ~180s)")
            
            poll_count += 1
            logger.info(f"  Poll #{poll_count}: {elapsed:.0f}s elapsed, operation still running...")
            time.sleep(10)
            
            # Refresh operation status
            operation = client.operations.get(operation)
        
        logger.info(f"✓ Video generation complete! Total time: {time.time() - start_time:.0f}s")
        progress(0.95, desc="💾 Downloading video...")
        
        # Extract video
        if operation.response and hasattr(operation.response, 'generated_videos'):
            logger.info(f"operations is {operation.response}")
            generated_video = operation.response.generated_videos[0]
            
            # Download video
            logger.info("📥 Downloading video bytes...")
            client.files.download(file=generated_video.video)
            video_bytes = generated_video.video.video_bytes
            logger.info(f"✓ Downloaded {len(video_bytes)} bytes")
            
            # Save to file
            timestamp = int(time.time())
            video_filename = os.path.join(OUTPUT_DIR, f"insta_reel_{timestamp}.mp4")
            
            with open(video_filename, 'wb') as f:
                f.write(video_bytes)
            
            file_size_mb = len(video_bytes) / (1024 * 1024)
            logger.info(f"💾 Saved video: {video_filename} ({file_size_mb:.2f} MB)")
            logger.info("✅ STEP 3 COMPLETE - Video generated successfully")
            
            progress(1.0, desc="✅ Video complete!")
            return video_filename, f"✅ Video saved to {video_filename} ({file_size_mb:.2f} MB)"
        else:
            logger.error("❌ No video in operation response")
            return None, "❌ Error: No video in operation response"
        
    except Exception as e:
        error_msg = f"❌ Error generating video: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

# --- CUSTOM CSS ---
custom_css = """
.container {
    max-width: 1400px;
    margin: auto;
}

.header-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.step-container {
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    background: #f9fafb;
}

.step-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.success-box {
    background: #d1fae5;
    border: 1px solid #6ee7b7;
    padding: 1rem;
    border-radius: 8px;
    color: #065f46;
}

.error-box {
    background: #fee2e2;
    border: 1px solid #fca5a5;
    padding: 1rem;
    border-radius: 8px;
    color: #991b1b;
}

.info-box {
    background: #dbeafe;
    border: 1px solid #93c5fd;
    padding: 1rem;
    border-radius: 8px;
    color: #1e40af;
    margin: 1rem 0;
}
"""

# --- UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Instagram AI Video Studio") as demo:
    
    # Step 1: Prompt Generation
    with gr.Column(elem_classes="step-container"):
        gr.Markdown('<div class="step-header">📸 Step 1: Upload Images & Generate Prompts</div>')
        
        with gr.Row():
            source_input = gr.Image(
                label="Source Character",
                type="pil",
                height=300,
                sources=["upload", "clipboard"]
            )
            scenic_input = gr.Image(
                label="Scenic Background",
                type="pil",
                height=300,
                sources=["upload", "clipboard"]
            )
        
        btn_analyze = gr.Button(
            "🪄 Generate AI Prompts",
            variant="primary",
            size="lg",
            scale=1
        )
        
        with gr.Row():
            nano_prompt = gr.Textbox(
                label="🎨 Image Fusion Prompt (Editable)",
                placeholder="AI will generate this based on your images...",
                lines=4,
                interactive=True,
                max_lines=6
            )
            veo_prompt = gr.Textbox(
                label="🎬 Video Motion Prompt (Editable)",
                placeholder="AI will generate this based on your images...",
                lines=4,
                interactive=True,
                max_lines=6
            )
    
    # Step 2: Anchor Frame Generation
    with gr.Column(elem_classes="step-container"):
        gr.Markdown('<div class="step-header">🖼️ Step 2: Generate Fusion Image</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                anchor_output = gr.Image(
                    label="🎨 Fused Anchor Frame",
                    height=400
                )
            with gr.Column(scale=1):
                anchor_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
                btn_image = gr.Button(
                    "🖌️ Create Fusion Image",
                    size="lg",
                    variant="secondary"
                )
                gr.Markdown("*Uses Gemini 2.5 Flash Image*")
        
        anchor_cache = gr.State()
    
    # Step 3: Video Generation
    with gr.Column(elem_classes="step-container"):
        gr.Markdown('<div class="step-header">🎥 Step 3: Generate Animated Video</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                video_output = gr.Video(
                    label="🎬 Final Instagram Reel (9:16)",
                    height=500
                )
            with gr.Column(scale=1):
                video_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
                btn_video = gr.Button(
                    "🚀 Render Video",
                    size="lg",
                    variant="stop"
                )
                gr.Markdown("""
                *Uses Veo 3.1 - Takes 1-3 minutes*
                
                ⏱️ **Please be patient!**
                Video generation is a complex process.
                """)
    
    # Tips Section
    gr.Markdown("""
    ---
    ### 💡 Pro Tips
    - **Image Quality:** Use high-resolution, well-lit images for best results
    - **Clear Subject:** Ensure the character is clearly visible without obstructions
    - **Edit Prompts:** Fine-tune AI-generated prompts for better control
    - **Aspect Ratio:** Videos are optimized for Instagram Reels (9:16 vertical)
    - **File Storage:** All outputs saved to `outputs/` directory (auto-cleanup after 48h)
    
    ### 🔧 Technical Details
    - **Step 1:** Gemini 2.0 Flash (prompt engineering)
    - **Step 2:** Gemini 2.5 Flash Image / Nano Banana (image fusion)
    - **Step 3:** Veo 3.1 (video generation - 720p, 9:16)
    
    ### 📝 Logs
    Check your terminal/console for detailed logging of each step.
    """)

    # Wire up the workflow
    btn_analyze.click(
        fn=generate_prompts,
        inputs=[source_input, scenic_input],
        outputs=[nano_prompt, veo_prompt]
    )
    
    btn_image.click(
        fn=generate_anchor,
        inputs=[source_input, scenic_input, nano_prompt],
        outputs=[anchor_output, anchor_cache, anchor_status]
    )
    
    btn_video.click(
        fn=generate_video,
        inputs=[anchor_cache, veo_prompt],
        outputs=[video_output, video_status]
    )

if __name__ == "__main__":
    logger.info("🚀 Starting Instagram AI Video Studio...")
    logger.info(f"✓ API Key configured: {API_KEY[:20]}...")
    logger.info(f"✓ Output directory: {OUTPUT_DIR}")
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )