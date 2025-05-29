#!/usr/bin/env python3
"""
Demo Launcher for PDF Page Ordering System
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'gradio',
        'fastapi',
        'uvicorn',
        'pydantic',
        'pydantic-settings',
        'python-multipart',
        'PyMuPDF',
        'pytesseract', 
        'Pillow',
        'opencv-python',
        'langchain',
        'langchain-google-genai',
        'google-generativeai',
        'langgraph',
        'numpy',
        'python-dotenv',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_environment():
    """Check environment setup"""
    issues = []
    
    # Check Google API key
    if not os.getenv('GOOGLE_API_KEY'):
        issues.append("⚠️ GOOGLE_API_KEY environment variable not set")
        logger.warning("Some features may not work without Google API key")
    
    # Check current directory
    current_dir = Path.cwd()
    if not (current_dir / 'src').exists():
        issues.append("❌ src directory not found - run from project root")
    
    if not (current_dir / 'gradio_demo.py').exists():
        issues.append("❌ gradio_demo.py not found")
    
    if issues:
        for issue in issues:
            logger.error(issue)
        return False
    
    return True

def setup_environment():
    """Setup environment variables if needed"""
    env_file = Path('.env')
    if not env_file.exists():
        logger.info("Creating .env file template...")
        env_content = """# PDF Page Ordering System Configuration

# Google AI API Key (required for advanced analysis)
GOOGLE_API_KEY=your_google_api_key_here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# OCR Configuration
MIN_OCR_CONFIDENCE=0.7
TESSERACT_LANG=vie+eng
OCR_DPI=300

# Processing Configuration
MAX_FILE_SIZE_MB=50
TIMEOUT_SECONDS=300
MAX_WORKERS=4

# Gemini Configuration
GEMINI_MODEL=gemini-2.5-flash-preview-0417
GEMINI_TEMPERATURE=0.1
"""
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logger.info(f"Created {env_file} - please configure your API keys")

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🚀 PDF Page Ordering System - Gradio Demo           ║
║                                                              ║
║        Advanced AI-powered document page ordering            ║
║        Built with LangGraph + Gradio                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

✨ Features:
  📄 Multi-format PDF support (text, scanned, mixed)
  🔢 Advanced page number extraction
  📊 Section and structure analysis  
  🔗 Content continuity analysis
  🎯 Multiple ordering strategies
  📈 Quality metrics and confidence scoring
  🧪 Built-in shuffling for accuracy testing

🌐 Demo will be available at: http://localhost:7860
📚 Upload a PDF and try different ordering strategies!
"""
    print(banner)

def run_demo():
    """Run the Gradio demo"""
    try:
        # Import and run the demo
        from gradio_demo import create_demo_interface
        
        logger.info("Creating Gradio interface...")
        demo = create_demo_interface()
        
        logger.info("Starting Gradio server...")
        logger.info("Access the demo at: http://localhost:7860")
        logger.info("Press Ctrl+C to stop the server")
        
        # Launch with optimized settings
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Set to True for public sharing
            debug=False,   # Set to True for development
            show_error=True,
            quiet=False,
            inbrowser=True,
            show_tips=True,
            height=800,
            max_threads=10
        )
        
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Failed to run demo: {e}")
        logger.info("Check the error messages above and try again")

def main():
    """Main function"""
    print_banner()
    
    logger.info("Checking system requirements...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed - please install missing packages")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed - please fix issues above")
        sys.exit(1)
    
    # Setup environment if needed
    setup_environment()
    
    # Additional setup tips
    if not os.getenv('GOOGLE_API_KEY'):
        logger.warning("🔑 Set GOOGLE_API_KEY environment variable for full functionality:")
        logger.warning("   export GOOGLE_API_KEY='your-api-key-here'")
        logger.warning("   Or add it to the .env file")
        logger.warning("")
        logger.info("⚡ Demo will still work with limited functionality")
        logger.info("")
    
    # Run the demo
    logger.info("🚀 Starting PDF Page Ordering Demo...")
    run_demo()

if __name__ == "__main__":
    main()