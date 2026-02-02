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
A hyper-realistic fusion of the source character seamlessly wearing the provided dress image, preserving exact facial identity, body proportions, and natural skin texture with visible pores and subtle imperfections. The dress is accurately fitted to the character’s body, respecting fabric structure, seams, folds, and material behavior exactly as shown in the reference image. Ensure correct draping, stretch, and weight of the garment with realistic fabric tension around shoulders, waist, and hips. Lighting on the dress matches the character’s lighting environment with proper highlights, shadows, and color fidelity. Skin remains realistic and detailed, not overly smooth or plastic. The outfit blends naturally with the character through precise alignment, scale, and contact shadows, creating a convincing virtual try-on result.|A clean, premium medium shot with a gentle camera pan and slight push-in to showcase the outfit. The character performs natural try-on motions such as a subtle turn of the torso, relaxed arm movements, slight posture adjustments, and soft breathing to demonstrate garment fit and flow. Fabric responds realistically to motion with natural folds and micro-movements. The camera remains smooth and stabilized, focusing attention on how the dress moves and fits on the character for a high-end virtual try-on presentation. Write 'Motion Prompt': Describe how the character should move in the video.
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