"""
Fragment processor for converting raw page data to structured fragments
"""
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base import ExtractionTool, ToolResult
from ...models.schemas import Fragment
from ...utils.helpers import extract_text_sample, clean_text_for_analysis

logger = logging.getLogger(__name__)

class FragmentProcessor(ExtractionTool):
    """Process raw page data into structured document fragments"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.min_content_length = self.get_config_value("min_content_length", 10)
        self.max_content_length = self.get_config_value("max_content_length", 100000)
        self.sample_chars = self.get_config_value("sample_chars", 200)
        self.clean_text = self.get_config_value("clean_text", True)
        self.auto_detect_empty = self.get_config_value("auto_detect_empty", True)
        
    @property
    def name(self) -> str:
        return "fragment_processor"
    
    @property
    def description(self) -> str:
        return "Convert raw page data into structured document fragments"
    
    @property
    def capabilities(self) -> List[str]:
        return super().capabilities + ["fragment_creation", "text_sampling", "content_validation"]
    
    def _execute(self, raw_pages: List[Dict[str, Any]], document_id: Optional[str] = None) -> ToolResult:
        """
        Convert raw pages to fragments
        
        Args:
            raw_pages: List of raw page data dictionaries
            document_id: Optional document identifier
            
        Returns:
            ToolResult with processed fragments
        """
        try:
            if not raw_pages:
                return ToolResult(
                    success=False,
                    error_message="No raw pages provided for processing"
                )
            
            document_id = document_id or str(uuid.uuid4())
            processed_fragments = []
            processing_stats = {
                "total_pages": len(raw_pages),
                "successful_fragments": 0,
                "empty_fragments": 0,
                "low_quality_fragments": 0,
                "high_quality_fragments": 0
            }
            
            for page_data in raw_pages:
                fragment_result = self._process_single_page(page_data, document_id)
                
                if fragment_result["success"]:
                    processed_fragments.append(fragment_result["fragment"])
                    processing_stats["successful_fragments"] += 1
                    
                    # Quality assessment
                    quality = fragment_result["quality_score"]
                    if quality < 0.3:
                        processing_stats["low_quality_fragments"] += 1
                    elif quality > 0.7:
                        processing_stats["high_quality_fragments"] += 1
                    
                    if fragment_result["is_empty"]:
                        processing_stats["empty_fragments"] += 1
                else:
                    logger.warning(f"Failed to process page {page_data.get('page_number', 'unknown')}: {fragment_result.get('error', 'Unknown error')}")
            
            # Calculate overall processing quality
            success_rate = processing_stats["successful_fragments"] / processing_stats["total_pages"]
            quality_score = self._calculate_overall_quality(processing_stats, processed_fragments)
            
            result_data = {
                "fragments": [f.dict() for f in processed_fragments],
                "processing_stats": processing_stats,
                "document_id": document_id,
                "success_rate": success_rate,
                "quality_score": quality_score
            }
            
            return ToolResult(
                success=len(processed_fragments) > 0,
                data=result_data,
                confidence=success_rate,
                metadata={
                    "total_pages": len(raw_pages),
                    "processed_fragments": len(processed_fragments),
                    "processing_method": "structured_conversion"
                }
            )
            
        except Exception as e:
            logger.error(f"Fragment processing failed: {str(e)}")
            return ToolResult(
                success=False,
                error_message=f"Fragment processing failed: {str(e)}"
            )
    
    def _process_single_page(self, page_data: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        """Process a single page into a fragment"""
        try:
            # Extract page information
            page_number = page_data.get("page_number", 1)
            content = page_data.get("text", "")
            extraction_method = page_data.get("extraction_method", "unknown")
            confidence = page_data.get("confidence", 1.0)
            
            # Clean content if enabled
            if self.clean_text and content:
                content = clean_text_for_analysis(content)
            
            # Check if content is empty or too short
            is_empty = not content or len(content.strip()) < self.min_content_length
            
            # Truncate if too long
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
                logger.warning(f"Truncated content for page {page_number} (was {len(content)} chars)")
            
            # Generate fragment ID
            fragment_id = f"{document_id}_page_{page_number:03d}"
            
            # Create text samples
            start_part, end_part = extract_text_sample(content, self.sample_chars, self.sample_chars)
            
            # Calculate quality score
            quality_score = self._calculate_fragment_quality(content, confidence, extraction_method)
            
            # Create fragment
            fragment = Fragment(
                id=fragment_id,
                content=content,
                start_part=start_part,
                end_part=end_part,
                original_page_number=page_number,
                extraction_method=extraction_method,
                confidence_score=confidence,
                processing_timestamp=datetime.now(),
                metadata={
                    "document_id": document_id,
                    "quality_score": quality_score,
                    "is_empty": is_empty,
                    "content_length": len(content),
                    "word_count": len(content.split()) if content else 0,
                    "processing_notes": self._generate_processing_notes(page_data, quality_score)
                }
            )
            
            return {
                "success": True,
                "fragment": fragment,
                "quality_score": quality_score,
                "is_empty": is_empty
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_fragment_quality(self, content: str, confidence: float, extraction_method: str) -> float:
        """Calculate quality score for a fragment"""
        
        if not content or not content.strip():
            return 0.0
        
        quality_factors = []
        
        # 1. Content length factor
        content_length = len(content.strip())
        if content_length > 1000:
            length_factor = 1.0
        elif content_length > 500:
            length_factor = 0.8
        elif content_length > 100:
            length_factor = 0.6
        else:
            length_factor = 0.3
        quality_factors.append(("length", length_factor, 0.3))
        
        # 2. Extraction confidence factor
        confidence_factor = min(confidence, 1.0)
        quality_factors.append(("confidence", confidence_factor, 0.4))
        
        # 3. Extraction method factor
        method_factors = {
            "direct_text": 1.0,
            "ocr": 0.7,
            "unknown": 0.5
        }
        method_factor = method_factors.get(extraction_method, 0.5)
        quality_factors.append(("method", method_factor, 0.2))
        
        # 4. Content structure factor
        structure_factor = self._analyze_content_structure(content)
        quality_factors.append(("structure", structure_factor, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        return min(max(total_score, 0.0), 1.0)
    
    def _analyze_content_structure(self, content: str) -> float:
        """Analyze content structure quality"""
        if not content:
            return 0.0
        
        # Check for basic text structure indicators
        indicators = {
            "has_sentences": '.' in content or '!' in content or '?' in content,
            "has_paragraphs": '\n' in content,
            "has_proper_words": any(word.isalpha() and len(word) > 2 for word in content.split()[:10]),
            "reasonable_length": 50 <= len(content) <= 10000,
            "not_mostly_numbers": sum(c.isdigit() for c in content) < len(content) * 0.8
        }
        
        structure_score = sum(indicators.values()) / len(indicators)
        return structure_score
    
    def _generate_processing_notes(self, page_data: Dict[str, Any], quality_score: float) -> List[str]:
        """Generate processing notes for debugging"""
        notes = []
        
        # Quality-based notes
        if quality_score < 0.3:
            notes.append("Low quality content detected")
        elif quality_score > 0.8:
            notes.append("High quality content")
        
        # Method-based notes
        extraction_method = page_data.get("extraction_method", "unknown")
        if extraction_method == "ocr":
            ocr_confidence = page_data.get("confidence", 0)
            if ocr_confidence < 0.7:
                notes.append(f"Low OCR confidence: {ocr_confidence:.2f}")
        
        # Content-based notes
        content = page_data.get("text", "")
        if not content.strip():
            notes.append("Empty or whitespace-only content")
        elif len(content) < 50:
            notes.append("Very short content")
        elif len(content) > 50000:
            notes.append("Very long content")
        
        # Image detection
        if page_data.get("has_images", False):
            notes.append("Page contains images")
        
        return notes
    
    def _calculate_overall_quality(self, stats: Dict[str, int], fragments: List[Fragment]) -> float:
        """Calculate overall processing quality score"""
        
        if not fragments:
            return 0.0
        
        # Factor 1: Success rate
        success_rate = stats["successful_fragments"] / stats["total_pages"]
        
        # Factor 2: Quality distribution
        if fragments:
            avg_quality = sum(f.metadata.get("quality_score", 0.5) for f in fragments) / len(fragments)
        else:
            avg_quality = 0.0
        
        # Factor 3: Empty content penalty
        empty_rate = stats["empty_fragments"] / max(stats["successful_fragments"], 1)
        empty_penalty = min(empty_rate, 0.5)  # Cap penalty at 0.5
        
        # Combine factors
        overall_quality = (success_rate * 0.4 + avg_quality * 0.5 - empty_penalty * 0.1)
        
        return min(max(overall_quality, 0.0), 1.0)
    
    def process_mixed_content(self, text_pages: List[Dict], ocr_pages: List[Dict], document_id: str) -> ToolResult:
        """Process mixed content from both direct text and OCR"""
        try:
            # Combine and sort pages
            all_pages = []
            
            # Add text pages
            for page in text_pages:
                page["extraction_method"] = "direct_text"
                page["confidence"] = 1.0
                all_pages.append(page)
            
            # Add OCR pages
            for page in ocr_pages:
                page["extraction_method"] = "ocr"
                # OCR confidence should already be set
                all_pages.append(page)
            
            # Sort by page number
            all_pages.sort(key=lambda x: x.get("page_number", 0))
            
            # Process using standard method
            return self._execute(all_pages, document_id)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Mixed content processing failed: {str(e)}"
            )
    
    def validate_fragments(self, fragments: List[Fragment]) -> Dict[str, Any]:
        """Validate processed fragments"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        if not fragments:
            validation_result["errors"].append("No fragments to validate")
            validation_result["valid"] = False
            return validation_result
        
        # Check for duplicate IDs
        fragment_ids = [f.id for f in fragments]
        if len(set(fragment_ids)) != len(fragment_ids):
            validation_result["errors"].append("Duplicate fragment IDs found")
            validation_result["valid"] = False
        
        # Check for empty content
        empty_count = sum(1 for f in fragments if not f.content.strip())
        if empty_count > 0:
            validation_result["warnings"].append(f"{empty_count} fragments have empty content")
        
        # Check page number sequence
        page_numbers = [f.original_page_number for f in fragments if f.original_page_number]
        if page_numbers:
            expected_pages = set(range(1, len(fragments) + 1))
            actual_pages = set(page_numbers)
            missing_pages = expected_pages - actual_pages
            if missing_pages:
                validation_result["warnings"].append(f"Missing page numbers: {sorted(missing_pages)}")
        
        # Quality statistics
        quality_scores = [f.metadata.get("quality_score", 0.5) for f in fragments]
        validation_result["statistics"] = {
            "total_fragments": len(fragments),
            "empty_fragments": empty_count,
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "high_quality_count": sum(1 for q in quality_scores if q > 0.7),
            "low_quality_count": sum(1 for q in quality_scores if q < 0.3)
        }
        
        return validation_result
    
    def regenerate_samples(self, fragments: List[Fragment], sample_length: int = 200) -> List[Fragment]:
        """Regenerate start_part and end_part samples with different length"""
        updated_fragments = []
        
        for fragment in fragments:
            start_part, end_part = extract_text_sample(
                fragment.content, sample_length, sample_length
            )
            
            # Create updated fragment
            updated_fragment = Fragment(
                id=fragment.id,
                content=fragment.content,
                start_part=start_part,
                end_part=end_part,
                original_page_number=fragment.original_page_number,
                extraction_method=fragment.extraction_method,
                confidence_score=fragment.confidence_score,
                processing_timestamp=fragment.processing_timestamp,
                metadata=fragment.metadata
            )
            
            updated_fragments.append(updated_fragment)
        
        return updated_fragments