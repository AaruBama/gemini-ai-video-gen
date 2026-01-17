import os
import logging

class Config:
    """Application configuration"""
    API_KEY = os.getenv("GOOGLE_API_KEY")
    OUTPUT_DIR = "outputs"
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
    
    # Model options
    IMAGE_MODELS = {
        "Gemini 2.5 Flash Image (Fast)": "gemini-2.5-flash-image",
        "Gemini 3 Pro Image (High Quality)": "gemini-3-pro-image-preview"
    }
    
    VIDEO_MODELS = {
        "Veo 3.1 (High Quality)": "veo-3.1-generate-preview",
        "Veo 3.1 Fast (Faster)": "veo-3.1-fast-generate-preview"
    }
    
    ASPECT_RATIOS = {
        "Portrait (9:16)": "9:16",
        "Landscape (16:9)": "16:9"
    }
    
    VIDEO_MODES = {
        "Default (Single Image)": "default",
        "Reference Images (Up to 3)": "reference",
        "First & Last Frame": "interpolation"
    }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for directory in [cls.OUTPUT_DIR, cls.IMAGES_DIR, cls.VIDEOS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.API_KEY:
            raise ValueError("❌ GOOGLE_API_KEY environment variable not set")
        cls.setup_directories()