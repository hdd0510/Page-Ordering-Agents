import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Tuple
import logging
from ..models.schemas import PDFType
import numpy as np
import tempfile
import os
import random

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> Tuple[List[str], PDFType]:
    """Extract text from PDF and determine if it's text-based or scanned."""
    try:
        doc = fitz.open(file_path)
        pages_text = []
        empty_pages = 0
        
        for page in doc:
            text = page.get_text()
            pages_text.append(text)
            
            if not text.strip():
                empty_pages += 1
        
        doc.close()
        
        # Determine PDF type
        if empty_pages == 0:
            pdf_type = PDFType.TEXT_BASED
        elif empty_pages == len(pages_text):
            pdf_type = PDFType.SCANNED
        else:
            pdf_type = PDFType.MIXED
            
        logger.info(f"Extracted text from {len(pages_text)} pages, type: {pdf_type}")
        return pages_text, pdf_type
        
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        return [], PDFType.SCANNED

def convert_pdf_to_images(file_path: str, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF pages to images for OCR processing."""
    try:
        doc = fitz.open(file_path)
        images = []
        
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        doc.close()
        logger.info(f"Converted {len(images)} pages to images")
        return images
        
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {str(e)}")
        return []

def shuffle_pdf(input_pdf_path: str):
    """
    Shuffle the pages of a PDF and create a new PDF with shuffled pages.
    Returns the path to the shuffled PDF and the shuffling mapping.
    """
    # Open input PDF
    pdf_doc = fitz.open(input_pdf_path)
    page_count = len(pdf_doc)
    
    if page_count <= 1:
        logger.info("PDF has only one page, no shuffling needed")
        return input_pdf_path, {0: 0}
    
    # Create shuffling mapping
    original_order = list(range(page_count))
    shuffled_order = original_order.copy()
    random.shuffle(shuffled_order)
    
    # Create a mapping from original to shuffled position
    page_mapping = {orig: shuffled for orig, shuffled in zip(original_order, shuffled_order)}
    
    # Create a new document with shuffled pages
    shuffled_doc = fitz.open()
    
    # Add pages in shuffled order
    for orig_idx in shuffled_order:
        shuffled_doc.insert_pdf(pdf_doc, from_page=orig_idx, to_page=orig_idx)
    
    # Save shuffled document to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        shuffled_pdf_path = tmp_file.name
        shuffled_doc.save(shuffled_pdf_path)
    
    logger.info(f"Created shuffled PDF with {page_count} pages")
    return shuffled_pdf_path, page_mapping