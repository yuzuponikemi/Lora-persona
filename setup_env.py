#!/usr/bin/env python3
"""
Environment setup and management script for LoRA Persona project
"""

import subprocess
import sys
import os
from pathlib import Path

def check_uv_installed():
    """Check if uv is installed"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ uv is not installed")
            return False
    except FileNotFoundError:
        print("❌ uv is not installed")
        return False

def check_environment():
    """Check if we're in the correct virtual environment"""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment found: .venv")
        
        # Check if we're currently in the venv
        current_python = sys.executable
        expected_python = venv_path / "Scripts" / "python.exe"
        
        if Path(current_python).resolve() == expected_python.resolve():
            print("✅ Currently using the project virtual environment")
            return True
        else:
            print("⚠️  Not using the project virtual environment")
            print(f"Current Python: {current_python}")
            print(f"Expected Python: {expected_python}")
            print("Activate with: .venv\\Scripts\\activate")
            return False
    else:
        print("❌ Virtual environment not found")
        return False

def install_packages():
    """Install packages using uv"""
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        print("📦 Installing packages with uv...")
        result = subprocess.run(['uv', 'pip', 'install', '-r', 'requirements.txt'])
        if result.returncode == 0:
            print("✅ Packages installed successfully")
        else:
            print("❌ Failed to install packages")
    else:
        print("❌ requirements.txt not found")

def check_gpu():
    """Check GPU availability for PyTorch"""
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"🚀 Environment: TRAINING (GPU-enabled)")
            return "training"
        else:
            print("💻 Environment: DEVELOPMENT (CPU-only)")
            print("💡 This is normal for laptop development")
            print("💡 Use workstation for GPU training")
            return "development"
            
    except ImportError:
        print("❌ PyTorch not installed")
        return "unknown"

def setup_environment():
    """Complete environment setup"""
    print("🚀 Setting up LoRA Persona environment...\n")
    
    # Check uv installation
    if not check_uv_installed():
        print("\n💡 Install uv with: winget install --id=astral-sh.uv -e")
        return
    
    # Create virtual environment if it doesn't exist
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("📝 Creating virtual environment...")
        subprocess.run(['uv', 'venv', '--python', '3.11'])
        print("✅ Virtual environment created")
    
    # Check environment
    check_environment()
    
    # Install packages
    install_packages()
    
    # Check GPU
    check_gpu()
    
    print("\n🎉 Environment setup complete!")
    print("\n📝 Next steps:")
    print("1. Activate the environment: .venv\\Scripts\\activate")
    print("2. Start Jupyter: jupyter lab")
    print("3. Select 'LoRA Persona (uv)' kernel in your notebook")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            setup_environment()
        elif sys.argv[1] == "check":
            check_uv_installed()
            check_environment()
            check_gpu()
        elif sys.argv[1] == "install":
            install_packages()
        else:
            print("Usage: python setup_env.py [setup|check|install]")
    else:
        # Default: just check status
        print("🔍 Environment Status Check\n")
        check_uv_installed()
        check_environment()
        check_gpu()
        
        print("\n💡 Commands:")
        print("  python setup_env.py setup   - Complete environment setup")
        print("  python setup_env.py check   - Check environment status")
        print("  python setup_env.py install - Install/update packages")

if __name__ == "__main__":
    main()