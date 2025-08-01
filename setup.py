#!/usr/bin/env python3
"""
Setup Script for Invoice Extraction System

This script sets up the environment and installs all necessary dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    success = run_command("pip install -r requirements.txt", "Installing requirements")
    
    if not success:
        print("\nTrying alternative installation methods...")
        
        # Try installing core dependencies first
        core_deps = [
            "torch",
            "torchvision", 
            "transformers",
            "opencv-python",
            "Pillow",
            "numpy",
            "pandas",
            "scikit-learn"
        ]
        
        for dep in core_deps:
            run_command(f"pip install {dep}", f"Installing {dep}")
    
    return success

def install_system_dependencies():
    """Install system-level dependencies"""
    print("\nInstalling system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Ubuntu/Debian
        if os.path.exists("/etc/debian_version"):
            packages = [
                "tesseract-ocr",
                "tesseract-ocr-eng",
                "poppler-utils",
                "libgl1-mesa-glx",
                "libglib2.0-0"
            ]
            
            for package in packages:
                run_command(f"sudo apt-get install -y {package}", f"Installing {package}")
        
        # CentOS/RHEL
        elif os.path.exists("/etc/redhat-release"):
            packages = [
                "tesseract",
                "tesseract-langpack-eng",
                "poppler-utils"
            ]
            
            for package in packages:
                run_command(f"sudo yum install -y {package}", f"Installing {package}")
    
    elif system == "darwin":  # macOS
        run_command("brew install tesseract poppler", "Installing Tesseract and Poppler")
    
    elif system == "windows":
        print("⚠ Windows users: Please install Tesseract manually from https://github.com/UB-Mannheim/tesseract/wiki")
        print("⚠ Also install Poppler for Windows if needed")

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "models/trained",
        "models/configs",
        "logs",
        "output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def download_sample_data():
    """Download sample datasets"""
    print("\nDownloading sample data...")
    
    try:
        from src.data_processing.download_datasets import DatasetDownloader
        
        downloader = DatasetDownloader()
        downloader.create_sample_invoices()
        print("✓ Sample data created")
        return True
    except Exception as e:
        print(f"⚠ Sample data creation failed: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("\nTesting installation...")
    
    tests = [
        ("import torch", "PyTorch"),
        ("import cv2", "OpenCV"),
        ("import transformers", "Transformers"),
        ("import PIL", "Pillow"),
        ("import numpy", "NumPy"),
        ("import pandas", "Pandas")
    ]
    
    all_passed = True
    for test, name in tests:
        try:
            exec(test)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_passed = False
    
    # Test OCR
    try:
        import pytesseract
        print("✓ Tesseract OCR available")
    except ImportError:
        print("⚠ Tesseract OCR not available")
    
    return all_passed

def main():
    """Main setup function"""
    print("=== Invoice Extraction System Setup ===\n")
    
    # Check Python version
    if not check_python_version():
        print("Setup failed due to incompatible Python version")
        return False
    
    # Create directories
    create_directories()
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install Python dependencies
    if not install_dependencies():
        print("⚠ Some dependencies may not have installed correctly")
    
    # Download sample data
    download_sample_data()
    
    # Test installation
    if test_installation():
        print("\n=== Setup completed successfully! ===")
        print("\nNext steps:")
        print("1. Run: python test_extraction.py")
        print("2. Or run: python src/extraction/extract.py --input path/to/invoice.jpg")
        print("3. Check the notebooks/ directory for exploration")
        return True
    else:
        print("\n⚠ Setup completed with warnings. Some features may not work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 