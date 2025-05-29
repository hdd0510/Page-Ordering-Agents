"""
Extraction nodes for LangGraph workflow
"""
import logging
import uuid
import random
from typing import Dict, Any, List
from .base_node import BaseNode
from ..models.schemas import ProcessingState, PDFType
from ..tools.extraction.pdf_extractor import PDFExtractor
from ..tools.extraction.fragment_processor import FragmentProcessor
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class PDFExtractionNode(BaseNode):
    """Node for extracting text content from PDF files"""
    
    def __init__(self):
        super().__init__()
        self.pdf_extractor = PDFExtractor()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "pdf_extraction"
    
    @property
    def description(self) -> str:
        return "Extract text content from PDF files using direct extraction and OCR"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Extract text content from PDF"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            pdf_path = state.get("pdf_path")
            if not pdf_path:
                self._handle_error(state, ValueError("PDF path not provided"), "Missing PDF path")
                return state
            
            # Extract text using PDF extractor tool
            extraction_result = self.pdf_extractor.execute(pdf_path)
            
            if not extraction_result.success:
                self._handle_error(
                    state, 
                    Exception(extraction_result.error_message), 
                    "PDF extraction failed"
                )
                return state
            
            # Update state with extraction results
            extraction_data = extraction_result.data
            state["pdf_type"] = extraction_data["pdf_type"]
            state["raw_pages"] = extraction_data["pages"]
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["extraction"] = {
                "pdf_type": extraction_data["pdf_type"],
                "total_pages": len(extraction_data["pages"]),
                "extraction_method": extraction_data.get("method", "unknown"),
                "processing_time_ms": extraction_result.processing_time_ms,
                "confidence": extraction_result.confidence
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Extracted {len(extraction_data['pages'])} pages from {pdf_path}, "
                f"type: {extraction_data['pdf_type']}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "PDF extraction failed")
        
        finally:
            self._log_step_end(state)
        
        return state

class FragmentProcessingNode(BaseNode):
    """Node for processing raw pages into structured fragments"""
    
    def __init__(self):
        super().__init__()
        self.fragment_processor = FragmentProcessor()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "fragment_processing"
    
    @property
    def description(self) -> str:
        return "Convert raw page data into structured document fragments"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Process raw pages into fragments"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            raw_pages = state.get("raw_pages", [])
            if not raw_pages:
                self._add_warning(state, "No raw pages available for fragment processing")
                state["fragments"] = []
                return state
            
            # Process pages into fragments
            processing_result = self.fragment_processor.execute(
                raw_pages, 
                document_id=state.get("document_id", "unknown")
            )
            
            if not processing_result.success:
                self._handle_error(
                    state,
                    Exception(processing_result.error_message),
                    "Fragment processing failed"
                )
                return state
            
            # Update state
            fragments_data = processing_result.data["fragments"]
            state["fragments"] = fragments_data
            state["original_order"] = list(range(len(fragments_data)))
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["fragment_processing"] = {
                "raw_pages_count": len(raw_pages),
                "fragments_created": len(fragments_data),
                "processing_success_rate": len(fragments_data) / max(len(raw_pages), 1),
                "average_fragment_length": sum(len(f.get("content", "")) for f in fragments_data) / max(len(fragments_data), 1),
                "processing_time_ms": processing_result.processing_time_ms
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Created {len(fragments_data)} fragments from {len(raw_pages)} raw pages"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Fragment processing failed")
        
        finally:
            self._log_step_end(state)
        
        return state

class ShuffleNode(BaseNode):
    """Node for randomly shuffling fragments (for testing purposes)"""
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "shuffle_fragments"
    
    @property
    def description(self) -> str:
        return "Randomly shuffle document fragments for testing ordering accuracy"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Shuffle fragments if enabled"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            # Check if shuffling is enabled
            shuffle_enabled = state.get("shuffle_enabled", False)
            fragments = state.get("fragments", [])
            
            if not shuffle_enabled:
                # No shuffling - maintain original order
                state["shuffled_order"] = state.get("original_order", list(range(len(fragments))))
                self.logger.info("Shuffling disabled - maintaining original order")
                return state
            
            if len(fragments) < 2:
                self._add_warning(state, "Cannot shuffle - need at least 2 fragments")
                state["shuffled_order"] = state.get("original_order", [])
                return state
            
            # Create shuffled order
            original_order = state.get("original_order", list(range(len(fragments))))
            shuffled_order = original_order.copy()
            random.shuffle(shuffled_order)
            
            # Apply shuffling to fragments
            original_fragments = fragments.copy()
            shuffled_fragments = [original_fragments[i] for i in shuffled_order]
            
            # Update state
            state["fragments"] = shuffled_fragments
            state["shuffled_order"] = shuffled_order
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["shuffling"] = {
                "enabled": True,
                "original_order": original_order,
                "shuffled_order": shuffled_order,
                "shuffle_mapping": {orig: shuf for orig, shuf in zip(original_order, shuffled_order)},
                "fragments_shuffled": len(fragments)
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Shuffled {len(fragments)} fragments: {original_order} -> {shuffled_order}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Fragment shuffling failed")
        
        finally:
            self._log_step_end(state)
        
        return state

class ValidationNode(BaseNode):
    """Node for validating processing state at various stages"""
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "validation"
    
    @property
    def description(self) -> str:
        return "Validate processing state and data integrity"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Validate current processing state"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            validation_results = {
                "state_validation": self._validate_state_structure(state),
                "data_validation": self._validate_data_integrity(state),
                "fragment_validation": self._validate_fragments(state),
                "overall_valid": True
            }
            
            # Check if any validation failed
            for validation_type, result in validation_results.items():
                if isinstance(result, dict) and not result.get("valid", True):
                    validation_results["overall_valid"] = False
                    for error in result.get("errors", []):
                        self._add_warning(state, f"{validation_type}: {error}")
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["validation"] = validation_results
            state["debug_info"] = debug_info
            
            if validation_results["overall_valid"]:
                self.logger.info("State validation passed")
            else:
                self.logger.warning("State validation found issues")
            
        except Exception as e:
            self._handle_error(state, e, "State validation failed")
        
        finally:
            self._log_step_end(state)
        
        return state
    
    def _validate_state_structure(self, state: ProcessingState) -> Dict[str, Any]:
        """Validate the structure of processing state"""
        required_keys = ["document_id", "pdf_path", "current_step", "processing_start_time"]
        errors = []
        
        for key in required_keys:
            if key not in state:
                errors.append(f"Missing required key: {key}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "required_keys_present": len(required_keys) - len(errors)
        }
    
    def _validate_data_integrity(self, state: ProcessingState) -> Dict[str, Any]:
        """Validate data integrity and consistency"""
        errors = []
        warnings = []
        
        # Check fragments consistency
        fragments = state.get("fragments", [])
        original_order = state.get("original_order", [])
        
        if len(fragments) != len(original_order):
            errors.append(f"Fragment count mismatch: {len(fragments)} fragments vs {len(original_order)} in order")
        
        # Check for duplicate fragment IDs
        if fragments:
            fragment_ids = [f.get("id") for f in fragments if f.get("id")]
            if len(set(fragment_ids)) != len(fragment_ids):
                errors.append("Duplicate fragment IDs detected")
        
        # Check processing times
        step_timings = state.get("step_timings", {})
        total_time = sum(step_timings.values())
        if total_time > self.settings.timeout_seconds:
            warnings.append(f"Processing time ({total_time:.2f}s) approaching timeout")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "integrity_score": 1.0 - (len(errors) * 0.2 + len(warnings) * 0.1)
        }
    
    def _validate_fragments(self, state: ProcessingState) -> Dict[str, Any]:
        """Validate individual fragments"""
        fragments = state.get("fragments", [])
        errors = []
        warnings = []
        
        empty_fragments = 0
        short_fragments = 0
        
        for i, fragment in enumerate(fragments):
            if not fragment.get("id"):
                errors.append(f"Fragment {i} missing ID")
            
            content = fragment.get("content", "")
            if not content.strip():
                empty_fragments += 1
            elif len(content.strip()) < 50:  # Very short fragments
                short_fragments += 1
        
        if empty_fragments > 0:
            warnings.append(f"{empty_fragments} fragments are empty")
        
        if short_fragments > len(fragments) * 0.5:  # More than 50% are short
            warnings.append(f"{short_fragments} fragments are very short")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_fragments": len(fragments),
            "empty_fragments": empty_fragments,
            "short_fragments": short_fragments,
            "quality_score": max(0.0, 1.0 - (empty_fragments + short_fragments) / max(len(fragments), 1))
        }
        