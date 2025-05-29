"""
PDF text extraction tool using PyMuPDF
"""
import fitz  # PyMuPDF
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

from ..base import ExtractionTool, ToolResult
from ...models.schemas import PDFType

logger = logging.getLogger(__name__)

class PDFExtractor(ExtractionTool):
    """Extract text content from PDF files"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.min_text_length = self.get_config_value("min_text_length", 10)
        self.detect_images = self.get_config_value("detect_images", True)
        
    @property
    def name(self) -> str:
        return "pdf_extractor"
    
    @property
    def description(self) -> str:
        return "Extract text content from PDF files and determine PDF type"
    
    @property
    def capabilities(self) -> List[str]:
        return super().capabilities + ["pdf_text_extraction", "pdf_type_detection", "page_analysis"]
    
    def _execute(self, pdf_path: str) -> ToolResult:
        """
        Extract text from PDF and determine type
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ToolResult with extracted text and PDF type info
        """
        try:
            # Validate input
            if not pdf_path or not Path(pdf_path).exists():
                return ToolResult(
                    success=False,
                    error_message=f"PDF file not found: {pdf_path}"
                )
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            if doc.page_count == 0:
                return ToolResult(
                    success=False,
                    error_message="PDF has no pages"
                )
            
            # Extract text from all pages
            pages_data = []
            text_pages = 0
            image_pages = 0
            empty_pages = 0
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text
                text_content = page.get_text()
                
                # Analyze page content
                has_text = bool(text_content.strip())
                has_images = self._detect_images_in_page(page) if self.detect_images else False
                
                page_info = {
                    "page_number": page_num + 1,
                    "text": text_content,
                    "has_text": has_text,
                    "has_images": has_images,
                    "char_count": len(text_content),
                    "word_count": len(text_content.split()) if text_content else 0,
                    "extraction_method": "direct_text"
                }
                
                pages_data.append(page_info)
                
                # Count page types
                if has_text:
                    text_pages += 1
                elif has_images:
                    image_pages += 1
                else:
                    empty_pages += 1
            
            # Close document
            doc.close()
            
            # Determine PDF type
            pdf_type = self._determine_pdf_type(text_pages, image_pages, empty_pages, doc.page_count)
            
            # Calculate extraction confidence
            confidence = self._calculate_extraction_confidence(pages_data, pdf_type)
            
            # Prepare result data
            result_data = {
                "pdf_type": pdf_type.value,
                "pages": pages_data,
                "total_pages": doc.page_count,
                "text_pages": text_pages,
                "image_pages": image_pages,
                "empty_pages": empty_pages,
                "extraction_stats": {
                    "total_characters": sum(p["char_count"] for p in pages_data),
                    "total_words": sum(p["word_count"] for p in pages_data),
                    "avg_chars_per_page": sum(p["char_count"] for p in pages_data) / len(pages_data),
                    "text_coverage": text_pages / doc.page_count
                },
                "method": "pymupdf_direct"
            }
            
            return ToolResult(
                success=True,
                data=result_data,
                confidence=confidence,
                metadata={
                    "pdf_path": pdf_path,
                    "pdf_type": pdf_type.value,
                    "extraction_method": "direct_text"
                }
            )
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
            return ToolResult(
                success=False,
                error_message=f"PDF extraction failed: {str(e)}",
                metadata={"pdf_path": pdf_path}
            )
    
    def _detect_images_in_page(self, page) -> bool:
        """Detect if page contains images"""
        try:
            # Get image list from page
            image_list = page.get_images()
            return len(image_list) > 0
        except:
            return False
    
    def _determine_pdf_type(self, text_pages: int, image_pages: int, empty_pages: int, total_pages: int) -> PDFType:
        """Determine PDF type based on page analysis"""
        
        if text_pages == 0:
            # No text pages - likely scanned
            return PDFType.SCANNED
        elif text_pages == total_pages:
            # All pages have text - text-based
            return PDFType.TEXT_BASED
        else:
            # Mixed content
            text_ratio = text_pages / total_pages
            if text_ratio >= 0.8:
                # Mostly text with some scanned pages
                return PDFType.TEXT_BASED
            elif text_ratio <= 0.2:
                # Mostly scanned with some text
                return PDFType.SCANNED
            else:
                # Truly mixed
                return PDFType.MIXED
    
    def _calculate_extraction_confidence(self, pages_data: List[Dict], pdf_type: PDFType) -> float:
        """Calculate confidence score for extraction"""
        
        if not pages_data:
            return 0.0
        
        # Base confidence by PDF type
        base_confidence = {
            PDFType.TEXT_BASED: 0.95,
            PDFType.MIXED: 0.75,
            PDFType.SCANNED: 0.60  # Lower because we haven't done OCR yet
        }
        
        confidence = base_confidence.get(pdf_type, 0.5)
        
        # Adjust based on content quality
        total_chars = sum(p["char_count"] for p in pages_data)
        avg_chars_per_page = total_chars / len(pages_data)
        
        # Boost confidence for substantial text content
        if avg_chars_per_page > 1000:  # Good amount of text
            confidence += 0.05
        elif avg_chars_per_page < 100:  # Very little text
            confidence -= 0.1
        
        # Check for consistent content across pages
        word_counts = [p["word_count"] for p in pages_data if p["word_count"] > 0]
        if word_counts:
            # Calculate coefficient of variation
            mean_words = sum(word_counts) / len(word_counts)
            if mean_words > 0:
                variance = sum((x - mean_words) ** 2 for x in word_counts) / len(word_counts)
                std_dev = variance ** 0.5
                cv = std_dev / mean_words
                
                # Lower variation = higher confidence
                if cv < 0.5:  # Consistent page lengths
                    confidence += 0.05
                elif cv > 2.0:  # Very inconsistent
                    confidence -= 0.05
        
        return min(max(confidence, 0.0), 1.0)
    
    def extract_page_range(self, pdf_path: str, start_page: int, end_page: int) -> ToolResult:
        """Extract text from specific page range"""
        try:
            doc = fitz.open(pdf_path)
            
            # Validate page range
            if start_page < 1 or end_page > doc.page_count or start_page > end_page:
                return ToolResult(
                    success=False,
                    error_message=f"Invalid page range: {start_page}-{end_page} for {doc.page_count} pages"
                )
            
            pages_data = []
            for page_num in range(start_page - 1, end_page):
                page = doc[page_num]
                text_content = page.get_text()
                
                page_info = {
                    "page_number": page_num + 1,
                    "text": text_content,
                    "char_count": len(text_content),
                    "word_count": len(text_content.split()) if text_content else 0
                }
                pages_data.append(page_info)
            
            doc.close()
            
            return ToolResult(
                success=True,
                data={"pages": pages_data, "page_range": f"{start_page}-{end_page}"},
                confidence=1.0
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Page range extraction failed: {str(e)}"
            )
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Get PDF metadata information"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            info = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": doc.page_count,
                "is_pdf": doc.is_pdf,
                "needs_password": doc.needs_pass,
                "is_closed": doc.is_closed
            }
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get PDF metadata: {e}")
            return {}
    
    def validate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Validate PDF file integrity"""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        try:
            # Check file exists
            if not Path(pdf_path).exists():
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Try to open PDF
            doc = fitz.open(pdf_path)
            
            # Basic validation
            if doc.page_count == 0:
                validation_result["errors"].append("PDF has no pages")
            
            if doc.needs_pass:
                validation_result["warnings"].append("PDF is password protected")
            
            # Check for corruption by trying to access first and last page
            try:
                first_page = doc[0]
                first_page.get_text()
                
                if doc.page_count > 1:
                    last_page = doc[doc.page_count - 1]
                    last_page.get_text()
                
            except Exception as e:
                validation_result["errors"].append(f"PDF appears corrupted: {str(e)}")
            
            # Gather info
            validation_result["info"] = {
                "page_count": doc.page_count,
                "file_size": Path(pdf_path).stat().st_size,
                "needs_password": doc.needs_pass,
                "pdf_version": getattr(doc, 'pdf_version', 'unknown')
            }
            
            doc.close()
            
            # Set valid flag
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            validation_result["errors"].append(f"Failed to validate PDF: {str(e)}")
        
        return validation_result