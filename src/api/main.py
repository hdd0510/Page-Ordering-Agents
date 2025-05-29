from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import tempfile
import os
import logging
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import agents and models
try:
    from ..agents.ocr_agent import OCRAgent
    from ..agents.page_ordering_agent import PageOrderingAgent
    from ..models.schemas import OrderingResult, PDFDocument
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    # For standalone testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.agents.ocr_agent import OCRAgent
    from src.agents.page_ordering_agent import PageOrderingAgent
    from src.models.schemas import OrderingResult, PDFDocument

app = FastAPI(
    title="PDF Page Ordering Agent API",
    description="Advanced PDF processing with page ordering using AI agents",
    version="1.0.0"
)

# Configuration
MIN_OCR_CONFIDENCE = float(os.getenv("MIN_OCR_CONFIDENCE", "0.7"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Initialize agents
try:
    ocr_agent = OCRAgent(min_confidence=MIN_OCR_CONFIDENCE)
    ordering_agent = PageOrderingAgent()
    logger.info("Agents initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agents: {e}")
    ocr_agent = None
    ordering_agent = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PDF Page Ordering Agent API",
        "version": "1.0.0",
        "status": "healthy" if (ocr_agent and ordering_agent) else "degraded",
        "endpoints": {
            "/process-pdf": "Process PDF and return ordered pages",
            "/analyze-pdf": "Analyze PDF structure without ordering",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "healthy" if (ocr_agent and ordering_agent) else "unhealthy"
    return {
        "status": status,
        "service": "PDF Page Ordering Agent",
        "agents": {
            "ocr_agent": ocr_agent is not None,
            "ordering_agent": ordering_agent is not None
        }
    }

@app.post("/process-pdf", response_model=OrderingResult)
async def process_pdf(file: UploadFile = File(...)):
    """
    Process uploaded PDF and return ordered page sequence.
    
    Supports both text-based and scanned PDFs.
    """
    if not ocr_agent or not ordering_agent:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        logger.info(f"Processing uploaded PDF: {file.filename}")
        
        # Step 1: Extract text/OCR
        pdf_document = ocr_agent.process_pdf(tmp_file_path)
        
        if not pdf_document.fragments:
            raise HTTPException(
                status_code=422, 
                detail="No readable content found in PDF"
            )
        
        # Step 2: Order pages
        result = ordering_agent.process_document(
            document_id=file.filename,
            fragments=pdf_document.fragments
        )
        
        logger.info(f"Successfully processed {file.filename} using {result.method_used}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.post("/analyze-pdf", response_model=PDFDocument)
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze PDF structure without ordering (for debugging).
    """
    if not ocr_agent:
        raise HTTPException(status_code=503, detail="OCR service temporarily unavailable")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        logger.info(f"Analyzing PDF: {file.filename}")
        pdf_document = ocr_agent.process_pdf(tmp_file_path)
        
        # Add some analysis metadata
        pdf_document.metadata["analyzed_at"] = str(time.time())
        pdf_document.metadata["fragment_count"] = len(pdf_document.fragments)
        
        return pdf_document
        
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.post("/process-shuffled-pdf", response_model=OrderingResult)
async def process_shuffled_pdf(file: UploadFile = File(...)):
    """
    Process uploaded PDF by first shuffling its pages randomly, then reordering them.
    
    This endpoint:
    1. Takes an original PDF
    2. Shuffles the pages randomly
    3. Processes and reorders them back to correct order
    4. Returns the ordering result
    
    Supports both text-based and scanned PDFs.
    """
    if not ocr_agent or not ordering_agent:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(content)
        orig_file_path = tmp_file.name
    
    try:
        logger.info(f"Processing and shuffling PDF: {file.filename}")
        
        # Step 1: Shuffle the PDF
        from ..tools.pdf_tools import shuffle_pdf
        shuffled_path, page_mapping = shuffle_pdf(orig_file_path)
        
        # Step 2: Process the shuffled PDF
        pdf_document = ocr_agent.process_pdf(shuffled_path)
        
        if not pdf_document.fragments:
            raise HTTPException(
                status_code=422, 
                detail="No readable content found in PDF"
            )
        
        # Add original page number mapping to fragments
        for fragment in pdf_document.fragments:
            shuffled_page = fragment.original_page_number - 1  # Convert to 0-indexed
            # Find the original page (before shuffling) for this fragment
            for orig_page, shuf_page in page_mapping.items():
                if shuf_page == shuffled_page:
                    # Store the original page number in metadata
                    fragment.metadata = fragment.metadata or {}
                    fragment.metadata["original_before_shuffle"] = orig_page + 1  # Convert back to 1-indexed
                    break
        
        # Step 3: Order pages
        result = ordering_agent.process_document(
            document_id=file.filename,
            fragments=pdf_document.fragments
        )
        
        # Add shuffle mapping to result
        result.debug_info = result.debug_info or {}
        result.debug_info["shuffle_mapping"] = page_mapping
        
        logger.info(f"Successfully processed shuffled PDF {file.filename} using {result.method_used}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing shuffled PDF failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        for path in [orig_file_path, shuffled_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {path}: {e}")

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to avoid 404 errors in logs."""
    return Response(content=b"", media_type="image/x-icon")

# Exception handlers
@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

if __name__ == "__main__":
    import uvicorn
    import time
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")