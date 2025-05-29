from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class PDFType(str, Enum):
    TEXT_BASED = "text_based"
    SCANNED = "scanned"
    MIXED = "mixed"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Fragment(BaseModel):
    id: str
    content: str
    start_part: Optional[str] = Field(None, description="First 200 chars of content")
    end_part: Optional[str] = Field(None, description="Last 200 chars of content")
    page_hint: Optional[int] = Field(None, description="Extracted page number")
    confidence_score: float = Field(default=0.0, description="OCR confidence if applicable")
    original_page_number: Optional[int] = Field(None, description="Original PDF page number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def model_post_init(self, _):
        if self.content:
            if len(self.content) > 400:
                self.start_part = self.content[:200]
                self.end_part = self.content[-200:]
            else:
                self.start_part = self.content
                self.end_part = self.content

class PDFDocument(BaseModel):
    file_path: str
    pdf_type: PDFType
    total_pages: int
    fragments: List[Fragment] = []
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: Dict[str, Any] = {}

class OrderingResult(BaseModel):
    document_id: str
    ordered_fragment_ids: List[str]
    confidence_score: float
    method_used: str  # "page_hint" or "llm_continuity"
    processing_time: float
    debug_info: Optional[Dict[str, Any]] = None