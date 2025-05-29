# src/__init__.py
"""
PDF Page Ordering Agent System

A sophisticated AI-powered system for processing and reordering PDF pages 
using LangGraph agents and advanced OCR techniques.
"""

__version__ = "1.0.0"
__author__ = "dinzdzun"

# src/models/__init__.py
from src.models.schemas import (
    PDFType,
    ProcessingStatus,
    Fragment,
    PDFDocument,
    OrderingResult
)

__all__ = [
    "PDFType",
    "ProcessingStatus", 
    "Fragment",
    "PDFDocument",
    "OrderingResult"
]

# src/tools/__init__.py
from src.tools.pdf_tools import extract_text_from_pdf, convert_pdf_to_images
from src.tools.ocr_tools import preprocess_image_for_ocr, ocr_image_with_confidence, batch_ocr_images
from src.tools.page_tools import extract_page_hint_advanced, validate_page_sequence, simple_page_sort
from src.tools.continuity_tools import continuity_score_detailed, find_best_continuation

__all__ = [
    # PDF tools
    "extract_text_from_pdf",
    "convert_pdf_to_images",
    
    # OCR tools
    "preprocess_image_for_ocr",
    "ocr_image_with_confidence", 
    "batch_ocr_images",
    
    # Page tools
    "extract_page_hint_advanced",
    "validate_page_sequence",
    "simple_page_sort",
    
    # Continuity tools
    "continuity_score_detailed",
    "find_best_continuation"
]

# src/agents/__init__.py
from src.agents.ocr_agent import OCRAgent
from src.agents.page_ordering_agent import PageOrderingAgent

__all__ = [
    "OCRAgent",
    "PageOrderingAgent"
]

# src/api/__init__.py
from src.api.main import app

__all__ = ["app"]