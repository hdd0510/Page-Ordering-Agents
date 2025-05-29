"""
Data models and schemas for PDF Page Ordering System
"""
from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime

# Enums
class PDFType(str, Enum):
    TEXT_BASED = "text_based"
    SCANNED = "scanned" 
    MIXED = "mixed"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class OrderingMethod(str, Enum):
    PAGE_NUMBERS = "page_numbers"
    SECTIONS = "sections"
    CONTINUITY = "continuity"
    HYBRID = "hybrid"
    FALLBACK = "fallback"

class AnalysisType(str, Enum):
    EXTRACTION = "extraction"
    PAGE_HINTS = "page_hints"
    SECTIONS = "sections"
    CONTINUITY = "continuity"
    ORDERING = "ordering"

# Core Data Models
class Fragment(BaseModel):
    """Represents a document fragment/page"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    start_part: Optional[str] = Field(None, description="First 200 chars")
    end_part: Optional[str] = Field(None, description="Last 200 chars")
    
    # Analysis results
    page_hint: Optional[int] = Field(None, description="Extracted page number") 
    section_info: Dict[str, Any] = Field(default_factory=dict)
    continuity_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    original_page_number: Optional[int] = None
    extraction_method: str = "unknown"
    confidence_score: float = 0.0
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def model_post_init(self, _):
        """Auto-generate start_part and end_part if not provided"""
        if self.content and not self.start_part:
            if len(self.content) > 400:
                self.start_part = self.content[:200]
                self.end_part = self.content[-200:]
            else:
                self.start_part = self.content
                self.end_part = self.content

class PDFDocument(BaseModel):
    """Represents a PDF document being processed"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    filename: Optional[str] = None
    pdf_type: PDFType
    total_pages: int = 0
    
    # Processing results
    fragments: List[Fragment] = []
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    
    # Analysis results
    analysis_results: Dict[AnalysisType, Dict[str, Any]] = Field(default_factory=dict)
    
    # Metadata
    file_size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

class OrderingResult(BaseModel):
    """Result of page ordering process"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    ordered_fragment_ids: List[str]
    
    # Quality metrics
    confidence_score: float = Field(ge=0.0, le=1.0)
    method_used: OrderingMethod
    processing_time_seconds: float = 0.0
    
    # Analysis details
    page_hints_found: int = 0
    section_analysis_success: bool = False
    continuity_analysis_success: bool = False
    
    # Debug information
    debug_info: Dict[str, Any] = Field(default_factory=dict)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

# State Management for LangGraph
class ProcessingState(TypedDict):
    """State schema for LangGraph workflow"""
    # Document information
    document_id: str
    pdf_path: str
    pdf_type: str
    filename: Optional[str]
    
    # Processing stages
    raw_pages: List[Dict[str, Any]]
    fragments: List[Dict[str, Any]]
    
    # Shuffling (for testing)
    shuffle_enabled: bool
    original_order: List[int]
    shuffled_order: List[int]
    
    # Analysis results
    page_hints: Dict[str, int]        # fragment_id -> page_number
    section_analysis: Dict[str, Any]  # Section analysis results
    continuity_scores: Dict[str, float] # Pairwise continuity scores
    structure_analysis: Dict[str, Any] # Document structure analysis
    
    # Decision and results
    selected_strategy: str
    final_order: List[str]
    confidence_score: float
    quality_metrics: Dict[str, float]
    
    # Metadata and debugging
    processing_start_time: float
    step_timings: Dict[str, float]
    debug_info: Dict[str, Any]
    errors: List[str]
    current_step: str

# Tool-specific schemas
class PageHintResult(BaseModel):
    """Result from page hint extraction"""
    success: bool
    page_number: Optional[int] = None
    confidence: float = 0.0
    pattern_type: str = "unknown"
    context: str = ""
    all_candidates: List[Dict[str, Any]] = []

class SectionAnalysisResult(BaseModel):
    """Result from section analysis"""
    chapters: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = []
    subsections: List[Dict[str, Any]] = []
    numbered_lists: List[Dict[str, Any]] = []
    hierarchy_detected: bool = False
    numbering_system: str = "none"
    confidence: float = 0.0

class ContinuityAnalysisResult(BaseModel):
    """Result from continuity analysis"""
    overall_score: float = Field(ge=0.0, le=1.0)
    component_scores: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    factors_analyzed: List[str] = []

class OrderingStrategy(BaseModel):
    """Configuration for ordering strategy"""
    name: str
    priority: int = 1
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)
    fallback_strategy: Optional[str] = None
    
# API Request/Response Models
class ProcessingRequest(BaseModel):
    """Request model for PDF processing"""
    shuffle_for_testing: bool = False
    enable_hybrid_mode: bool = True
    analysis_modes: List[AnalysisType] = Field(default_factory=lambda: list(AnalysisType))
    custom_config: Dict[str, Any] = Field(default_factory=dict)

class ProcessingResponse(BaseModel):
    """Response model for PDF processing"""
    success: bool
    document_info: Dict[str, Any]
    ordering_result: OrderingResult
    analysis_details: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time_seconds: float
    errors: List[str] = []

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "PDF Page Ordering System"
    version: str = "2.0.0"
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

# Utility functions for schema validation
def validate_fragment_order(fragments: List[Fragment], order: List[str]) -> bool:
    """Validate that the ordering contains all fragment IDs"""
    fragment_ids = {f.id for f in fragments}
    order_ids = set(order)
    return fragment_ids == order_ids

def calculate_ordering_confidence(
    fragments: List[Fragment], 
    method: OrderingMethod,
    analysis_results: Dict[str, Any]
) -> float:
    """Calculate confidence score for ordering result"""
    base_confidence = {
        OrderingMethod.PAGE_NUMBERS: 0.95,
        OrderingMethod.SECTIONS: 0.85,
        OrderingMethod.CONTINUITY: 0.75,
        OrderingMethod.HYBRID: 0.80,
        OrderingMethod.FALLBACK: 0.30
    }.get(method, 0.50)
    
    # Adjust based on analysis quality
    if method == OrderingMethod.PAGE_NUMBERS:
        page_coverage = len(analysis_results.get("page_hints", {})) / max(len(fragments), 1)
        base_confidence *= page_coverage
    
    elif method == OrderingMethod.CONTINUITY:
        avg_continuity = sum(analysis_results.get("continuity_scores", {}).values()) / max(len(analysis_results.get("continuity_scores", {})), 1)
        base_confidence = min(base_confidence, avg_continuity + 0.2)
    
    return min(max(base_confidence, 0.0), 1.0)

# Export commonly used types
__all__ = [
    "PDFType", "ProcessingStatus", "OrderingMethod", "AnalysisType",
    "Fragment", "PDFDocument", "OrderingResult", "ProcessingState",
    "PageHintResult", "SectionAnalysisResult", "ContinuityAnalysisResult",
    "ProcessingRequest", "ProcessingResponse", "HealthCheckResponse",
    "validate_fragment_order", "calculate_ordering_confidence"
]