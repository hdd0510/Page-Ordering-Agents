"""
Custom exceptions for PDF page ordering system
"""
from typing import Optional, Dict, Any

class PDFOrderingException(Exception):
    """Base exception for PDF ordering system"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }

class ValidationError(PDFOrderingException):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["invalid_value"] = str(value)

class ProcessingError(PDFOrderingException):
    """Raised when PDF processing fails"""
    
    def __init__(self, message: str, stage: Optional[str] = None, pdf_path: Optional[str] = None):
        super().__init__(message, error_code="PROCESSING_ERROR")
        if stage:
            self.details["processing_stage"] = stage
        if pdf_path:
            self.details["pdf_path"] = pdf_path

class ExtractionError(ProcessingError):
    """Raised when text extraction fails"""
    
    def __init__(self, message: str, pdf_path: Optional[str] = None, page_number: Optional[int] = None):
        super().__init__(message, stage="extraction", pdf_path=pdf_path)
        self.error_code = "EXTRACTION_ERROR"
        if page_number is not None:
            self.details["page_number"] = page_number

class OCRError(ExtractionError):
    """Raised when OCR processing fails"""
    
    def __init__(self, message: str, confidence: Optional[float] = None, ocr_engine: Optional[str] = None):
        super().__init__(message)
        self.error_code = "OCR_ERROR"
        if confidence is not None:
            self.details["ocr_confidence"] = confidence
        if ocr_engine:
            self.details["ocr_engine"] = ocr_engine

class AnalysisError(ProcessingError):
    """Raised when analysis fails"""
    
    def __init__(self, message: str, analysis_type: Optional[str] = None, fragment_id: Optional[str] = None):
        super().__init__(message, stage="analysis")
        self.error_code = "ANALYSIS_ERROR"
        if analysis_type:
            self.details["analysis_type"] = analysis_type
        if fragment_id:
            self.details["fragment_id"] = fragment_id

class OrderingError(ProcessingError):
    """Raised when ordering fails"""
    
    def __init__(self, message: str, strategy: Optional[str] = None, fragment_count: Optional[int] = None):
        super().__init__(message, stage="ordering")
        self.error_code = "ORDERING_ERROR"
        if strategy:
            self.details["ordering_strategy"] = strategy
        if fragment_count is not None:
            self.details["fragment_count"] = fragment_count

class WorkflowError(PDFOrderingException):
    """Raised when LangGraph workflow fails"""
    
    def __init__(self, message: str, node_name: Optional[str] = None, workflow_type: Optional[str] = None):
        super().__init__(message, error_code="WORKFLOW_ERROR")
        if node_name:
            self.details["failed_node"] = node_name
        if workflow_type:
            self.details["workflow_type"] = workflow_type

class ConfigurationError(PDFOrderingException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, expected_type: Optional[str] = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        if config_key:
            self.details["config_key"] = config_key
        if expected_type:
            self.details["expected_type"] = expected_type

class TimeoutError(PDFOrderingException):
    """Raised when processing exceeds timeout"""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, elapsed_seconds: Optional[float] = None):
        super().__init__(message, error_code="TIMEOUT_ERROR")
        if timeout_seconds is not None:
            self.details["timeout_seconds"] = timeout_seconds
        if elapsed_seconds is not None:
            self.details["elapsed_seconds"] = elapsed_seconds

class APIError(PDFOrderingException):
    """Raised when external API calls fail"""
    
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message, error_code="API_ERROR")
        if api_name:
            self.details["api_name"] = api_name
        if status_code is not None:
            self.details["status_code"] = status_code

class ToolError(PDFOrderingException):
    """Raised when tool execution fails"""
    
    def __init__(self, message: str, tool_name: Optional[str] = None, tool_version: Optional[str] = None):
        super().__init__(message, error_code="TOOL_ERROR")
        if tool_name:
            self.details["tool_name"] = tool_name
        if tool_version:
            self.details["tool_version"] = tool_version

class ResourceError(PDFOrderingException):
    """Raised when system resources are insufficient"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, required: Optional[str] = None, available: Optional[str] = None):
        super().__init__(message, error_code="RESOURCE_ERROR")
        if resource_type:
            self.details["resource_type"] = resource_type
        if required:
            self.details["required"] = required
        if available:
            self.details["available"] = available

# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAPPING = {
    ValidationError: 400,
    ConfigurationError: 400,
    ProcessingError: 422,
    ExtractionError: 422,
    OCRError: 422,
    AnalysisError: 422,
    OrderingError: 422,
    WorkflowError: 500,
    TimeoutError: 408,
    APIError: 502,
    ToolError: 500,
    ResourceError: 507,
    PDFOrderingException: 500  # Default for base exception
}

def get_http_status_code(exception: Exception) -> int:
    """Get appropriate HTTP status code for exception"""
    for exc_type, status_code in EXCEPTION_STATUS_MAPPING.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500  # Default internal server error

def create_error_response(exception: Exception) -> Dict[str, Any]:
    """Create standardized error response"""
    if isinstance(exception, PDFOrderingException):
        response = exception.to_dict()
    else:
        response = {
            "error": "UnknownError",
            "message": str(exception),
            "error_code": "UNKNOWN_ERROR",
            "details": {}
        }
    
    response["status_code"] = get_http_status_code(exception)
    return response

# Context managers for error handling
class ErrorContext:
    """Context manager for handling errors in specific contexts"""
    
    def __init__(self, context_name: str, reraise_as: type = ProcessingError):
        self.context_name = context_name
        self.reraise_as = reraise_as
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and not isinstance(exc_val, PDFOrderingException):
            # Convert non-PDFOrdering exceptions to our custom exceptions
            raise self.reraise_as(
                f"Error in {self.context_name}: {str(exc_val)}",
                stage=self.context_name
            ) from exc_val
        return False  # Don't suppress the exception

# Utility functions
def handle_api_error(func):
    """Decorator to handle API errors consistently"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, PDFOrderingException):
                raise
            else:
                raise APIError(f"API call failed: {str(e)}") from e
    return wrapper

def validate_and_raise(condition: bool, error_class: type, message: str, **kwargs):
    """Validate condition and raise specific error if false"""
    if not condition:
        raise error_class(message, **kwargs)

# Export all exceptions
__all__ = [
    # Base exception
    "PDFOrderingException",
    
    # Specific exceptions
    "ValidationError",
    "ProcessingError", 
    "ExtractionError",
    "OCRError",
    "AnalysisError",
    "OrderingError",
    "WorkflowError",
    "ConfigurationError",
    "TimeoutError",
    "APIError",
    "ToolError",
    "ResourceError",
    
    # Utilities
    "get_http_status_code",
    "create_error_response",
    "ErrorContext",
    "handle_api_error",
    "validate_and_raise",
    "EXCEPTION_STATUS_MAPPING"
]