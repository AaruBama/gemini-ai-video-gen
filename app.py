import gradio as gr
from google import genai
from config import Config
from utils.logger import setup_logger
from utils.file_manager import FileManager
from services.prompt_service import PromptService
from services.image_service import ImageService
from services.video_service import VideoService
from ui.components import UIComponents

logger = setup_logger(__name__)

# Validate configuration
Config.validate()
client = genai.Client(api_key=Config.API_KEY)

# Initialize services
prompt_service = PromptService(client)
image_service = ImageService(client)
video_service = VideoService(client)

# Custom CSS
custom_css = """
.step-container {
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    background: #f9fafb;
}

.info-box {
    background: #dbeafe;
    border: 1px solid #93c5fd;
    padding: 1rem;
    border-radius: 8px;
    color: #1e40af;
    margin: 1rem 0;
}

.tab-nav button {
    font-size: 1.1rem !important;
    padding: 0.75rem 1.5rem !important;
}
"""

def update_video_mode_ui(mode):
    """Update UI based on selected video generation mode"""
    if mode == "Default (Single Image)":
        return (
            gr.update(visible=True),   # upload_image
            gr.update(visible=False),  # ref_images
            gr.update(visible=False),  # first_frame
            gr.update(visible=False)   # last_frame
        )
    elif mode == "Reference Images (Up to 3)":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    elif mode == "First & Last Frame":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True)
        )

def refresh_library(filter_type):
    """Refresh library view"""
    files = FileManager.get_library_files()
    
    if filter_type == "All":
        display_data = files
    elif filter_type == "Images":
        display_data = {'images': files['images'], 'videos': []}
    else:  # Videos
        display_data = {'images': [], 'videos': files['videos']}
    
    return UIComponents.create_library_view(display_data)

# Build UI
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Instagram AI Video Studio") as demo:
    
    UIComponents.create_header()
    
    with gr.Tabs() as tabs:
        # ===== STUDIO TAB =====
        with gr.Tab("🎬 Studio"):
            
            gr.Markdown("""
            <div class="info-box">
            <strong>📖 Quick Guide:</strong><br>
            <b>Step 1:</b> Upload images → Generate prompts (or write your own)<br>
            <b>Step 2:</b> Create fusion image combining character and background<br>
            <b>Step 3:</b> Animate into video with multiple mode options
            </div>
            """)
            
            # STEP 1: Prompt Generation
            with gr.Group(elem_classes="step-container"):
                UIComponents.create_step_container("📸 Step 1: Upload Images & Generate Prompts")
                
                with gr.Row():
                    source_input = gr.Image(label="Source Character", type="pil", height=300)
                    scenic_input = gr.Image(label="Scenic Background", type="pil", height=300)
                
                btn_analyze = gr.Button("🪄 Generate AI Prompts", variant="primary", size="lg")
                
                with gr.Row():
                    nano_prompt = gr.Textbox(
                        label="🎨 Image Fusion Prompt (Editable)",
                        placeholder="AI will generate this...",
                        lines=4,
                        max_lines=8
                    )
                    veo_prompt = gr.Textbox(
                        label="🎬 Video Motion Prompt (Editable)",
                        placeholder="AI will generate this...",
                        lines=4,
                        max_lines=8
                    )
            
            # STEP 2: Image Generation
            with gr.Group(elem_classes="step-container"):
                UIComponents.create_step_container("🖼️ Step 2: Generate Fusion Image")
                
                with gr.Row():
                    image_model = gr.Dropdown(
                        choices=list(Config.IMAGE_MODELS.keys()),
                        value=list(Config.IMAGE_MODELS.keys())[0],
                        label="Image Model"
                    )
                    image_aspect = gr.Dropdown(
                        choices=list(Config.ASPECT_RATIOS.keys()),
                        value="Portrait (9:16)",
                        label="Aspect Ratio"
                    )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        anchor_output = gr.Image(label="🎨 Fused Image", height=400)
                    with gr.Column(scale=1):
                        anchor_status = gr.Textbox(label="Status", interactive=False, lines=3)
                        btn_image = gr.Button("🖌️ Create Fusion Image", size="lg", variant="secondary")
                
                anchor_cache = gr.State()
            
            # STEP 3: Video Generation
            with gr.Group(elem_classes="step-container"):
                UIComponents.create_step_container("🎥 Step 3: Generate Animated Video")
                
                with gr.Row():
                    video_model = gr.Dropdown(
                        choices=list(Config.VIDEO_MODELS.keys()),
                        value=list(Config.VIDEO_MODELS.keys())[0],
                        label="Video Model"
                    )
                    video_aspect = gr.Dropdown(
                        choices=list(Config.ASPECT_RATIOS.keys()),
                        value="Portrait (9:16)",
                        label="Aspect Ratio"
                    )
                    video_mode = gr.Dropdown(
                        choices=list(Config.VIDEO_MODES.keys()),
                        value="Default (Single Image)",
                        label="Generation Mode"
                    )
                
                # Mode-specific inputs
                with gr.Row():
                    upload_image = gr.Image(
                        label="📤 Upload Custom Image (Optional - Uses Step 2 image if empty)",
                        type="pil",
                        visible=True
                    )
                    ref_images = gr.Gallery(
                        label="📤 Upload Reference Images (1-3)",
                        visible=False,
                        columns=3,
                        height=200
                    )
                
                with gr.Row():
                    first_frame = gr.Image(
                        label="📤 First Frame",
                        type="pil",
                        visible=False
                    )
                    last_frame = gr.Image(
                        label="📤 Last Frame",
                        type="pil",
                        visible=False
                    )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        video_output = gr.Video(label="🎬 Final Video", height=500)
                    with gr.Column(scale=1):
                        video_status = gr.Textbox(label="Status", interactive=False, lines=3)
                        btn_video = gr.Button("🚀 Render Video", size="lg", variant="stop")
                        gr.Markdown("*⏱️ Takes 1-3 minutes*")
            
            # Tips
            with gr.Accordion("💡 Pro Tips & Settings", open=False):
                gr.Markdown("""
                ### Video Generation Modes
                - **Default:** Uses image from Step 2 (or uploaded image)
                - **Reference Images:** Upload 1-3 images to guide content/style
                - **First & Last Frame:** Define start and end frames for interpolation
                
                ### Best Practices
                - Use high-resolution, well-lit images
                - Be specific in prompts for better results
                - Edit AI prompts to fine-tune output
                - Check library for all generated content
                """)
        
        # ===== LIBRARY TAB =====
        with gr.Tab("📚 My Library"):
            gr.Markdown("### Your Generated Content")
            
            with gr.Row():
                library_filter = gr.Radio(
                    choices=["All", "Images", "Videos"],
                    value="All",
                    label="Filter"
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            library_gallery = gr.HTML()
            
            # Load library on tab open
            refresh_btn.click(
                fn=refresh_library,
                inputs=[library_filter],
                outputs=[library_gallery]
            )
            
            library_filter.change(
                fn=refresh_library,
                inputs=[library_filter],
                outputs=[library_gallery]
            )
    
    # ===== EVENT HANDLERS =====
    
    # Step 1: Generate prompts
    btn_analyze.click(
        fn=prompt_service.generate_prompts,
        inputs=[source_input, scenic_input],
        outputs=[nano_prompt, veo_prompt]
    )
    
    # Step 2: Generate image
    def generate_image_wrapper(source, scenic, prompt, model_name, aspect_name, progress=gr.Progress()):
        model = Config.IMAGE_MODELS[model_name]
        aspect = Config.ASPECT_RATIOS[aspect_name]
        return image_service.generate_image(source, scenic, prompt, model, aspect, progress)
    
    btn_image.click(
        fn=generate_image_wrapper,
        inputs=[source_input, scenic_input, nano_prompt, image_model, image_aspect],
        outputs=[anchor_output, anchor_cache, anchor_status]
    )
    
    # Step 3: Generate video
    def generate_video_wrapper(mode_name, upload_img, ref_imgs, first, last, 
                               anchor_path, prompt, model_name, aspect_name, progress=gr.Progress()):
        mode = Config.VIDEO_MODES[mode_name]
        model = Config.VIDEO_MODELS[model_name]
        aspect = Config.ASPECT_RATIOS[aspect_name]
        
        # Use uploaded image if provided, otherwise use anchor from step 2
        image_to_use = upload_img if upload_img is not None else anchor_path
        
        return video_service.generate_video(
            mode, image_to_use, ref_imgs, first, last, prompt, model, aspect, progress
        )
    
    btn_video.click(
        fn=generate_video_wrapper,
        inputs=[video_mode, upload_image, ref_images, first_frame, last_frame,
                anchor_cache, veo_prompt, video_model, video_aspect],
        outputs=[video_output, video_status]
    )
    
    # Update UI when video mode changes
    video_mode.change(
        fn=update_video_mode_ui,
        inputs=[video_mode],
        outputs=[upload_image, ref_images, first_frame, last_frame]
    )

if __name__ == "__main__":
    logger.info("🚀 Starting Instagram AI Video Studio...")
    logger.info(f"✓ API Key configured")
    logger.info(f"✓ Output directory: {Config.OUTPUT_DIR}")
    
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True
    )