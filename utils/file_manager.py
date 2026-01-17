import os
import glob
from typing import List, Dict
from pathlib import Path
from datetime import datetime

class FileManager:
    """Manages file operations and library"""
    
    @staticmethod
    def get_library_files() -> Dict[str, List[Dict]]:
        """Get all generated files categorized"""
        from config import Config
        
        images = []
        videos = []
        
        # Get images
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for filepath in glob.glob(os.path.join(Config.IMAGES_DIR, ext)):
                stat = os.stat(filepath)
                images.append({
                    'path': filepath,
                    'name': os.path.basename(filepath),
                    'size': FileManager._format_size(stat.st_size),
                    'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M')
                })
        
        # Get videos
        for filepath in glob.glob(os.path.join(Config.VIDEOS_DIR, '*.mp4')):
            stat = os.stat(filepath)
            videos.append({
                'path': filepath,
                'name': os.path.basename(filepath),
                'size': FileManager._format_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M')
            })
        
        return {
            'images': sorted(images, key=lambda x: x['created'], reverse=True),
            'videos': sorted(videos, key=lambda x: x['created'], reverse=True)
        }
    
    @staticmethod
    def _format_size(bytes: int) -> str:
        """Format bytes to human readable size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} TB"
