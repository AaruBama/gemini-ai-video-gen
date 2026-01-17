import gradio as gr
from config import Config

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def create_header():
        """Create application header"""
        return gr.Markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1>🎬 Instagram AI Video Studio</h1>
            <p style="font-size: 1.2rem;">Create Professional AI-Powered Videos in 3 Simple Steps</p>
            <p style="opacity: 0.9;">Powered by Google Gemini & Veo 3.1</p>
        </div>
        """)
    
    @staticmethod
    def create_step_container(title: str):
        """Create step container with header"""
        return gr.Markdown(f'<div style="font-size: 1.3rem; font-weight: 600; color: #1f2937; margin-bottom: 1rem;">{title}</div>')
    
    @staticmethod
    def create_library_view(files_data):
        """Create library file grid"""
        images = files_data['images']
        videos = files_data['videos']
        
        html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem;">'
        
        for img in images:
            html += f"""
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; background: white;">
                <img src="file/{img['path']}" style="width: 100%; border-radius: 4px; margin-bottom: 0.5rem;">
                <div style="font-size: 0.9rem; color: #6b7280;">
                    <div style="font-weight: 600; margin-bottom: 0.25rem;">{img['name'][:20]}...</div>
                    <div>{img['size']} • {img['created']}</div>
                </div>
            </div>
            """
        
        for vid in videos:
            html += f"""
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; background: white;">
                <video controls style="width: 100%; border-radius: 4px; margin-bottom: 0.5rem;">
                    <source src="file/{vid['path']}" type="video/mp4">
                </video>
                <div style="font-size: 0.9rem; color: #6b7280;">
                    <div style="font-weight: 600; margin-bottom: 0.25rem;">{vid['name'][:20]}...</div>
                    <div>{vid['size']} • {vid['created']}</div>
                </div>
            </div>
            """
        
        html += '</div>'
        return html