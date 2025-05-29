#!/usr/bin/env python3

import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    import uvicorn
    from src.api.main import app
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"""
🚀 PDF Page Ordering Agent API
===============================
Starting server on http://{host}:{port}

API Documentation:
- Swagger UI: http://{host}:{port}/docs
- ReDoc: http://{host}:{port}/redoc

Endpoints:
- POST /process-pdf - Process PDF and return ordered pages
- POST /analyze-pdf - Analyze PDF structure
- GET /health - Health check
    """)
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )