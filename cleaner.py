#!/usr/bin/env python3

import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger('MediPredict_Cleanup')

class ProjectCleaner:
    def __init__(self, base_dir='.'):
        """
        Initialize ProjectCleaner
        
        Args:
            base_dir (str): Base directory of the project
        """
        self.base_dir = os.path.abspath(base_dir)
        
        # Files and directories to keep
        self.keep_files = [
            'setup.py', 
            'train_models.py', 
            'requirements.txt', 
            'README.md',
            '.gitignore',
            'app.py'
        ]
        
        self.keep_dirs = [
            'medipredict', 
            'data', 
            'models', 
            'notebooks', 
            'tests'
        ]
        
        # Files and directories to remove
        self.remove_patterns = [
            '__pycache__', 
            '.DS_Store', 
            '*.pyc', 
            '*.log',
            '.idea',
            '.vscode',
            '*.egg-info'
        ]

    def _safe_remove(self, path):
        """
        Safely remove a file or directory
        
        Args:
            path (str): Path to file or directory
        """
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                logger.info(f"Removed directory: {path}")
            else:
                os.remove(path)
                logger.info(f"Removed file: {path}")
        except Exception as e:
            logger.warning(f"Could not remove {path}: {e}")

    def clean_project(self):
        """
        Clean up the project directory
        """
        logger.info("Starting MediPredict project cleanup...")
        
        # Walk through the project directory
        for root, dirs, files in os.walk(self.base_dir, topdown=False):
            # Remove unwanted directories
            for dir_name in dirs[:]:
                full_path = os.path.join(root, dir_name)
                
                # Remove __pycache__ and other unwanted directories
                if any(pattern in dir_name for pattern in self.remove_patterns):
                    self._safe_remove(full_path)
                    dirs.remove(dir_name)
            
            # Remove unwanted files
            for file_name in files:
                full_path = os.path.join(root, file_name)
                
                # Check if file should be removed
                if (
                    any(pattern in file_name for pattern in self.remove_patterns) and
                    file_name not in self.keep_files
                ):
                    self._safe_remove(full_path)

        # Create necessary directories
        necessary_dirs = [
            'data/raw', 
            'data/processed', 
            'models', 
            'models/preprocessor', 
            'visualizations'
        ]
        
        for dir_path in necessary_dirs:
            full_path = os.path.join(self.base_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {full_path}")

        # Remove redundant setup/run scripts
        redundant_scripts = [
            'setup.sh', 
            'run_medipredict.sh', 
            'run_simple.sh', 
            'run.py',
            'fix_xgboost_mac.sh',
            'complete_fix.sh',
            'immediate_fix.sh'
        ]
        
        for script in redundant_scripts:
            script_path = os.path.join(self.base_dir, script)
            if os.path.exists(script_path):
                self._safe_remove(script_path)

        logger.info("MediPredict project cleanup complete!")

def main():
    cleaner = ProjectCleaner()
    cleaner.clean_project()

if __name__ == '__main__':
    main()