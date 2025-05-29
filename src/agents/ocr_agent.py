from ..tools.pdf_tools import extract_text_from_pdf, convert_pdf_to_images
from ..tools.ocr_tools import batch_ocr_images
from ..models.schemas import PDFDocument, Fragment, PDFType
from typing import List
import uuid
import logging

logger = logging.getLogger(__name__)

class OCRAgent:
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        logger.info(f"OCR Agent initialized with min_confidence: {min_confidence}")
    
    def process_pdf(self, file_path: str) -> PDFDocument:
        """Process PDF and extract text content."""
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            # First attempt: extract text directly
            pages_text, pdf_type = extract_text_from_pdf(file_path)
            
            doc = PDFDocument(
                file_path=file_path,
                pdf_type=pdf_type,
                total_pages=len(pages_text)
            )
            
            if pdf_type == PDFType.TEXT_BASED:
                # Use extracted text directly
                doc.fragments = self._create_fragments_from_text(pages_text)
                logger.info(f"Extracted text from {len(pages_text)} pages")
                
            elif pdf_type == PDFType.SCANNED:
                # Perform OCR
                doc.fragments = self._process_scanned_pdf(file_path, len(pages_text))
                logger.info(f"OCR processed {len(pages_text)} scanned pages")
                
            else:  # MIXED
                # Combine text extraction and OCR
                doc.fragments = self._process_mixed_pdf(file_path, pages_text)
                logger.info(f"Mixed processing for {len(pages_text)} pages")
            
            doc.processing_status = "completed"
            logger.info(f"Successfully processed PDF with {len(doc.fragments)} fragments")
            return doc
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {str(e)}")
            # Return empty document with error status
            return PDFDocument(
                file_path=file_path,
                pdf_type=PDFType.TEXT_BASED,
                total_pages=0,
                processing_status="failed",
                metadata={"error": str(e)}
            )
    
    def _create_fragments_from_text(self, pages_text: List[str]) -> List[Fragment]:
        """Create fragments from extracted text."""
        fragments = []
        
        for i, text in enumerate(pages_text):
            if text.strip():  # Skip empty pages
                fragment = Fragment(
                    id=str(uuid.uuid4()),
                    content=text.strip(),
                    confidence_score=1.0,  # High confidence for direct text
                    original_page_number=i + 1
                )
                fragments.append(fragment)
        
        logger.info(f"Created {len(fragments)} fragments from text")
        return fragments
    
    def _process_scanned_pdf(self, file_path: str, num_pages: int) -> List[Fragment]:
        """Process scanned PDF with OCR."""
        try:
            # Convert to images
            images = convert_pdf_to_images(file_path)
            
            if not images:
                logger.warning("No images extracted from PDF")
                return []
            
            # Perform batch OCR
            ocr_results = batch_ocr_images(images)
            
            fragments = []
            for i, (text, confidence) in enumerate(ocr_results):
                if text.strip() and confidence >= self.min_confidence:
                    fragment = Fragment(
                        id=str(uuid.uuid4()),
                        content=text.strip(),
                        confidence_score=confidence,
                        original_page_number=i + 1
                    )
                    fragments.append(fragment)
                else:
                    logger.warning(f"Page {i+1} skipped due to low OCR confidence: {confidence:.2f}")
            
            return fragments
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return []
    
    def _process_mixed_pdf(self, file_path: str, pages_text: List[str]) -> List[Fragment]:
        """Process mixed PDF (some text, some scanned)."""
        try:
            fragments = []
            
            # Identify which pages need OCR
            pages_needing_ocr = [i for i, text in enumerate(pages_text) if not text.strip()]
            
            ocr_dict = {}
            if pages_needing_ocr:
                # Convert only needed pages to images
                all_images = convert_pdf_to_images(file_path)
                if all_images:
                    ocr_images = [all_images[i] for i in pages_needing_ocr if i < len(all_images)]
                    ocr_results = batch_ocr_images(ocr_images)
                    ocr_dict = dict(zip(pages_needing_ocr, ocr_results))
            
            # Create fragments
            for i, text in enumerate(pages_text):
                if text.strip():
                    # Use extracted text
                    fragment = Fragment(
                        id=str(uuid.uuid4()),
                        content=text.strip(),
                        confidence_score=1.0,
                        original_page_number=i + 1
                    )
                    fragments.append(fragment)
                elif i in ocr_dict:
                    # Use OCR result
                    ocr_text, confidence = ocr_dict[i]
                    if ocr_text.strip() and confidence >= self.min_confidence:
                        fragment = Fragment(
                            id=str(uuid.uuid4()),
                            content=ocr_text.strip(),
                            confidence_score=confidence,
                            original_page_number=i + 1
                        )
                        fragments.append(fragment)
            
            return fragments
            
        except Exception as e:
            logger.error(f"Mixed PDF processing failed: {str(e)}")
            return []