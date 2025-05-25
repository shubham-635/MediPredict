#!/usr/bin/env python3

import os
import sys
import subprocess
import platform
import venv
from pathlib import Path

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_colored(message, color=Colors.GREEN):
    """Print colored messages"""
    print(f"{color}{message}{Colors.RESET}")

def check_python_version():
    """Verify Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored("Error: Python 3.8+ is required.", Colors.RED)
        sys.exit(1)

def create_virtual_environment(env_path):
    """Create a virtual environment"""
    try:
        print_colored("\n🔧 Creating virtual environment...", Colors.BLUE)
        venv.create(env_path, with_pip=True)
        print_colored("✅ Virtual environment created successfully!", Colors.GREEN)
    except Exception as e:
        print_colored(f"❌ Error creating virtual environment: {e}", Colors.RED)
        sys.exit(1)

def install_dependencies(env_path):
    """Install project dependencies"""
    pip_path = os.path.join(env_path, 'bin', 'pip') if platform.system() != 'Windows' else os.path.join(env_path, 'Scripts', 'pip')
    
    requirements = [
        'numpy>=1.20.0,<1.24.0',
        'pandas>=1.3.0,<2.0.0',
        'scikit-learn>=1.0.0,<1.3.0',
        'matplotlib>=3.4.0,<3.8.0',
        'seaborn>=0.11.0,<0.13.0',
        'flask>=2.0.0,<3.0.0',
        'joblib>=1.1.0,<1.3.0'
    ]

    # Special handling for M1/M2 Macs
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        requirements.append('tensorflow-macos>=2.9.0')
    else:
        requirements.append('tensorflow>=2.9.0,<2.13.0')

    try:
        print_colored("\n📦 Installing dependencies...", Colors.BLUE)
        subprocess.check_call([pip_path, 'install', '--upgrade', 'pip'])
        subprocess.check_call([pip_path, 'install'] + requirements)
        print_colored("✅ Dependencies installed successfully!", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Error installing dependencies: {e}", Colors.RED)
        sys.exit(1)

def create_directories():
    """Create necessary project directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'models/diabetes',
        'models/heart_disease',
        'models/stroke',
        'models/preprocessor',
        'visualizations'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print_colored("\n📂 Created project directories", Colors.BLUE)

def check_datasets():
    """Check if datasets are present"""
    datasets = [
        'data/raw/heart_disease.csv',
        'data/raw/diabetes.csv',
        'data/raw/stroke.csv',
        'data/raw/framingham_heart_study.csv'
    ]
    
    missing_datasets = [ds for ds in datasets if not os.path.exists(ds)]
    
    if missing_datasets:
        print_colored("\n⚠️ Missing datasets:", Colors.YELLOW)
        for ds in missing_datasets:
            print_colored(f"   - {ds}", Colors.YELLOW)
        print_colored("\nWARNING: Some datasets are missing. The application will use synthetic data.", Colors.RED)
    else:
        print_colored("✅ All datasets are present", Colors.GREEN)

def prompt_train_models():
    """Prompt user to train models"""
    while True:
        response = input("\n🤔 Would you like to train the models now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            try:
                subprocess.run([sys.executable, 'train_models.py'], check=True)
                break
            except subprocess.CalledProcessError:
                print_colored("❌ Model training failed.", Colors.RED)
                break
        elif response in ['n', 'no']:
            print_colored("⏩ Skipping model training. You can train later using train_models.py", Colors.YELLOW)
            break
        else:
            print_colored("Invalid input. Please enter 'y' or 'n'.", Colors.RED)

def main():
    """Main setup script"""
    print_colored("🩺 MediPredict Setup", Colors.GREEN)
    print_colored("=====================", Colors.GREEN)
    
    check_python_version()
    
    # Determine virtual environment path
    base_dir = Path(__file__).parent
    env_path = base_dir / 'medipredict_env'
    
    create_virtual_environment(env_path)
    
    # Activate virtual environment (Note: This is a subprocess approach)
    activate_this = env_path / ('bin/activate_this.py' if platform.system() != 'Windows' else 'Scripts/activate_this.py')
    
    try:
        exec(open(activate_this).read(), {'__file__': str(activate_this)})
    except Exception as e:
        print_colored(f"❌ Could not activate virtual environment: {e}", Colors.RED)
        sys.exit(1)
    
    install_dependencies(env_path)
    create_directories()
    check_datasets()
    prompt_train_models()
    
    print_colored("\n🎉 Setup Complete!", Colors.GREEN)
    print_colored("To activate the environment: source medipredict_env/bin/activate", Colors.BLUE)

if __name__ == '__main__':
    main()