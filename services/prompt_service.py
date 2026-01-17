from google import genai
from google.genai import types
import gradio as gr
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PromptService:
    """Handles prompt generation"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    def generate_prompts(self, source_img, scenic_img, progress=gr.Progress()):
        """Generate fusion and motion prompts"""
        logger.info("=" * 60)
        logger.info("STEP 1: PROMPT GENERATION STARTED")
        logger.info("=" * 60)
        
        if source_img is None or scenic_img is None:
            logger.error("❌ Missing images")
            return "❌ Error: Both images required", "❌ Error: Both images required"
        
        progress(0.1, desc="🔍 Analyzing images...")
        
        system_instruction = """
You are an AI Video Director. Create two prompts for image fusion and video animation.

TASK:
1. Analyze the 'Source Character' and 'Scenic Background' images.
2. Write 'Fusion Prompt': Describe how to merge the person from the source into the scenery.
   - Be specific about character details and background elements
   - Choose appropriate outfit based on scene
   - Mention lighting and atmosphere matching
   - Make skin hyper-realistic, not ultra-smooth
3. Write 'Motion Prompt': Describe how the character should move in the video.
   - Include camera movement, character actions, background activity

OUTPUT FORMAT:
Return ONLY two prompts separated by a pipe (|).
Format: <Fusion Prompt>|<Motion Prompt>
No labels, markdown, or extra text.
"""
        
        try:
            progress(0.3, desc="🤖 Gemini analyzing...")
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[system_instruction, source_img, scenic_img]
            )
            
            logger.info(f"✓ Response received - {len(response.text)} chars")
            progress(0.8, desc="📝 Processing...")
            
            raw_text = response.text.strip()
            if raw_text.startswith('```'):
                lines = raw_text.split('\n')
                raw_text = '\n'.join(line for line in lines if not line.startswith('```')).strip()
            
            if "|" in raw_text:
                parts = raw_text.split('|', 1)
                nano_prompt = parts[0].strip()
                veo_prompt = parts[1].strip()
                logger.info("✅ STEP 1 COMPLETE")
                progress(1.0, desc="✅ Prompts generated!")
                return nano_prompt, veo_prompt
            else:
                logger.warning("⚠️ Pipe separator not found")
                return raw_text, "The character moves naturally with subtle, realistic motions."
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, error_msg