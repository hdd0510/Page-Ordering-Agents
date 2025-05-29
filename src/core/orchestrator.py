"""
Main orchestrator for PDF page ordering system
"""
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..models.schemas import (
    ProcessingState, ProcessingStatus, OrderingResult, 
    OrderingMethod, PDFDocument, calculate_ordering_confidence
)
from ..graph.builder import WorkflowFactory, validate_workflow
from ..config.settings import get_settings
from .exceptions import (
    ProcessingError, ValidationError, WorkflowError, 
    ConfigurationError, TimeoutError
)

logger = logging.getLogger(__name__)

class PDFOrderingOrchestrator:
    """Main orchestrator for PDF page ordering workflow"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.settings = get_settings()
        self.config = config or {}
        self.workflow = None
        self.workflow_type = self.config.get("workflow_type", "standard")
        self._initialize_workflow()
        
    def _initialize_workflow(self):
        """Initialize the LangGraph workflow"""
        try:
            if self.workflow_type == "standard":
                self.workflow = WorkflowFactory.create_standard_workflow(self.config)
            elif self.workflow_type == "testing":
                self.workflow = WorkflowFactory.create_testing_workflow(self.config)
            elif self.workflow_type == "production":
                self.workflow = WorkflowFactory.create_production_workflow(self.config)
            else:
                raise ConfigurationError(f"Unknown workflow type: {self.workflow_type}")
            
            # Validate workflow
            validation_result = validate_workflow(self.workflow)
            if not validation_result["valid"]:
                raise WorkflowError(f"Workflow validation failed: {validation_result['errors']}")
            
            logger.info(f"Initialized {self.workflow_type} workflow with {validation_result['node_count']} nodes")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {e}")
            raise ConfigurationError(f"Workflow initialization failed: {str(e)}")
    
    def process_pdf(
        self, 
        pdf_path: str, 
        filename: Optional[str] = None,
        shuffle_for_testing: bool = False,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> OrderingResult:
        """
        Process a PDF file and return ordered page sequence
        
        Args:
            pdf_path: Path to PDF file
            filename: Original filename (optional)
            shuffle_for_testing: Whether to shuffle pages for testing
            custom_config: Custom configuration overrides
            
        Returns:
            OrderingResult with ordered fragment sequence
        """
        # Validate inputs
        if not Path(pdf_path).exists():
            raise ValidationError(f"PDF file not found: {pdf_path}")
        
        if not Path(pdf_path).suffix.lower() == '.pdf':
            raise ValidationError(f"File is not a PDF: {pdf_path}")
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Initialize processing state
        initial_state = self._create_initial_state(
            document_id=document_id,
            pdf_path=pdf_path,
            filename=filename or Path(pdf_path).name,
            shuffle_enabled=shuffle_for_testing,
            custom_config=custom_config
        )
        
        logger.info(f"Starting PDF processing for document {document_id}: {filename}")
        
        try:
            # Execute workflow
            final_state = self._execute_workflow(initial_state)
            
            # Create result
            result = self._create_ordering_result(final_state)
            
            logger.info(
                f"Successfully processed {document_id}: "
                f"method={result.method_used}, confidence={result.confidence_score:.3f}, "
                f"time={result.processing_time_seconds:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed for document {document_id}: {str(e)}")
            
            # Return fallback result
            return self._create_fallback_result(
                document_id=document_id,
                filename=filename,
                error=str(e),
                processing_time=time.time() - initial_state["processing_start_time"]
            )
    
    def analyze_pdf_structure(
        self, 
        pdf_path: str,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze PDF structure without ordering (for debugging/inspection)
        
        Args:
            pdf_path: Path to PDF file
            filename: Original filename (optional)
            
        Returns:
            Detailed analysis results
        """
        # Similar to process_pdf but stops after analysis
        document_id = str(uuid.uuid4())
        
        initial_state = self._create_initial_state(
            document_id=document_id,
            pdf_path=pdf_path,
            filename=filename or Path(pdf_path).name,
            shuffle_enabled=False,
            analysis_only=True
        )
        
        try:
            # Execute partial workflow (analysis only)
            partial_state = self._execute_analysis_workflow(initial_state)
            
            return {
                "document_id": document_id,
                "filename": filename,
                "pdf_type": partial_state.get("pdf_type", "unknown"),
                "total_fragments": len(partial_state.get("fragments", [])),
                "page_hints_found": len(partial_state.get("page_hints", {})),
                "section_analysis": partial_state.get("section_analysis", {}),
                "continuity_analysis": {
                    "total_pairs": len(partial_state.get("continuity_scores", {})),
                    "average_score": self._calculate_average_continuity(partial_state)
                },
                "processing_time": time.time() - initial_state["processing_start_time"],
                "debug_info": partial_state.get("debug_info", {}),
                "errors": partial_state.get("errors", [])
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for document {document_id}: {str(e)}")
            return {
                "document_id": document_id,
                "error": str(e),
                "success": False
            }
    
    def _create_initial_state(
        self,
        document_id: str,
        pdf_path: str,
        filename: str,
        shuffle_enabled: bool = False,
        custom_config: Optional[Dict[str, Any]] = None,
        analysis_only: bool = False
    ) -> ProcessingState:
        """Create initial processing state"""
        
        return {
            # Document info
            "document_id": document_id,
            "pdf_path": pdf_path,
            "pdf_type": "",
            "filename": filename,
            
            # Processing stages
            "raw_pages": [],
            "fragments": [],
            
            # Testing options
            "shuffle_enabled": shuffle_enabled,
            "original_order": [],
            "shuffled_order": [],
            
            # Analysis results
            "page_hints": {},
            "section_analysis": {},
            "continuity_scores": {},
            "structure_analysis": {},
            
            # Results
            "selected_strategy": "",
            "final_order": [],
            "confidence_score": 0.0,
            "quality_metrics": {},
            
            # Metadata
            "processing_start_time": time.time(),
            "step_timings": {},
            "debug_info": {
                "workflow_type": self.workflow_type,
                "custom_config": custom_config or {},
                "analysis_only": analysis_only
            },
            "errors": [],
            "current_step": "initialization"
        }
    
    def _execute_workflow(self, initial_state: ProcessingState) -> ProcessingState:
        """Execute the complete workflow"""
        try:
            # Set timeout
            timeout = self.settings.timeout_seconds
            start_time = time.time()
            
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Check for timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Workflow execution exceeded {timeout}s timeout")
            
            return final_state
            
        except Exception as e:
            if isinstance(e, (ProcessingError, ValidationError, WorkflowError, TimeoutError)):
                raise
            else:
                raise WorkflowError(f"Workflow execution failed: {str(e)}")
    
    def _execute_analysis_workflow(self, initial_state: ProcessingState) -> ProcessingState:
        """Execute only the analysis portion of workflow"""
        # This would require a modified workflow that stops after analysis
        # For now, execute full workflow and ignore ordering results
        return self._execute_workflow(initial_state)
    
    def _create_ordering_result(self, final_state: ProcessingState) -> OrderingResult:
        """Create OrderingResult from final workflow state"""
        
        # Determine method used
        method_mapping = {
            "page_numbers": OrderingMethod.PAGE_NUMBERS,
            "sections": OrderingMethod.SECTIONS,
            "continuity": OrderingMethod.CONTINUITY,
            "hybrid": OrderingMethod.HYBRID,
            "fallback": OrderingMethod.FALLBACK
        }
        
        method_used = method_mapping.get(
            final_state.get("selected_strategy", "fallback"),
            OrderingMethod.FALLBACK
        )
        
        # Calculate processing time
        processing_time = time.time() - final_state["processing_start_time"]
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(final_state)
        
        return OrderingResult(
            document_id=final_state["document_id"],
            ordered_fragment_ids=final_state.get("final_order", []),
            confidence_score=final_state.get("confidence_score", 0.0),
            method_used=method_used,
            processing_time_seconds=processing_time,
            page_hints_found=len(final_state.get("page_hints", {})),
            section_analysis_success=final_state.get("section_analysis", {}).get("hierarchy_detected", False),
            continuity_analysis_success=len(final_state.get("continuity_scores", {})) > 0,
            debug_info=final_state.get("debug_info", {}),
            quality_metrics=quality_metrics
        )
    
    def _create_fallback_result(
        self,
        document_id: str,
        filename: Optional[str],
        error: str,
        processing_time: float
    ) -> OrderingResult:
        """Create fallback result when processing fails"""
        
        return OrderingResult(
            document_id=document_id,
            ordered_fragment_ids=[],
            confidence_score=0.0,
            method_used=OrderingMethod.FALLBACK,
            processing_time_seconds=processing_time,
            debug_info={
                "error": error,
                "filename": filename,
                "processing_failed": True
            }
        )
    
    def _calculate_quality_metrics(self, final_state: ProcessingState) -> Dict[str, float]:
        """Calculate quality metrics for the ordering result"""
        metrics = {}
        
        # Fragment processing success rate
        raw_pages_count = len(final_state.get("raw_pages", []))
        fragments_count = len(final_state.get("fragments", []))
        metrics["fragment_extraction_rate"] = fragments_count / max(raw_pages_count, 1)
        
        # Page hint coverage
        page_hints_count = len(final_state.get("page_hints", {}))
        metrics["page_hint_coverage"] = page_hints_count / max(fragments_count, 1)
        
        # Section analysis success
        section_analysis = final_state.get("section_analysis", {})
        metrics["section_analysis_success"] = 1.0 if section_analysis.get("hierarchy_detected", False) else 0.0
        
        # Continuity analysis quality
        continuity_scores = final_state.get("continuity_scores", {})
        if continuity_scores:
            avg_continuity = sum(continuity_scores.values()) / len(continuity_scores)
            metrics["average_continuity_score"] = avg_continuity
        else:
            metrics["average_continuity_score"] = 0.0
        
        # Overall processing success
        errors_count = len(final_state.get("errors", []))
        metrics["processing_success_rate"] = 1.0 if errors_count == 0 else max(0.0, 1.0 - errors_count * 0.1)
        
        return metrics
    
    def _calculate_average_continuity(self, state: ProcessingState) -> float:
        """Calculate average continuity score"""
        continuity_scores = state.get("continuity_scores", {})
        if not continuity_scores:
            return 0.0
        
        return sum(continuity_scores.values()) / len(continuity_scores)
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the current workflow"""
        return {
            "workflow_type": self.workflow_type,
            "config": self.config,
            "settings": {
                "timeout_seconds": self.settings.timeout_seconds,
                "max_file_size_mb": self.settings.max_file_size_mb,
                "min_ocr_confidence": self.settings.min_ocr_confidence
            },
            "capabilities": [
                "pdf_extraction",
                "page_hint_detection", 
                "section_analysis",
                "continuity_analysis",
                "multiple_ordering_strategies",
                "quality_metrics"
            ]
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate settings
        if not self.settings.google_api_key:
            validation_results["warnings"].append("Google API key not configured")
        
        if self.settings.timeout_seconds < 60:
            validation_results["warnings"].append("Timeout may be too low for large documents")
        
        # Validate workflow
        if not self.workflow:
            validation_results["valid"] = False
            validation_results["errors"].append("Workflow not initialized")
        
        return validation_results

# Factory function for easy initialization
def create_orchestrator(
    workflow_type: str = "standard",
    config: Optional[Dict[str, Any]] = None
) -> PDFOrderingOrchestrator:
    """Create PDF ordering orchestrator with specified configuration"""
    
    full_config = {"workflow_type": workflow_type}
    if config:
        full_config.update(config)
    
    return PDFOrderingOrchestrator(full_config)

# Export main classes
__all__ = [
    "PDFOrderingOrchestrator", "create_orchestrator"
]   