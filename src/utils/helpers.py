"""
General utility functions and helpers
"""
import os
import uuid
import hashlib
import tempfile
import shutil
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import time
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_processing_id() -> str:
    """Generate unique processing ID"""
    timestamp = str(int(time.time()))
    random_part = str(uuid.uuid4())[:8]
    return f"proc_{timestamp}_{random_part}"

def generate_document_id() -> str:
    """Generate unique document ID"""
    return str(uuid.uuid4())

def generate_fragment_id(document_id: str, page_number: int) -> str:
    """Generate fragment ID from document ID and page number"""
    return f"{document_id}_page_{page_number:03d}"

def cleanup_temp_file(file_path: str) -> bool:
    """Safely cleanup temporary file"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")
    return False

def create_temp_directory() -> str:
    """Create temporary directory for processing"""
    temp_dir = tempfile.mkdtemp(prefix="pdf_ordering_")
    logger.debug(f"Created temp directory: {temp_dir}")
    return temp_dir

def cleanup_temp_directory(dir_path: str) -> bool:
    """Safely cleanup temporary directory"""
    try:
        if dir_path and os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.debug(f"Cleaned up temp directory: {dir_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to cleanup temp directory {dir_path}: {e}")
    return False

def ensure_directory_exists(dir_path: str) -> bool:
    """Ensure directory exists, create if necessary"""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False

def get_file_hash(file_path: str, hash_algorithm: str = "md5") -> Optional[str]:
    """Calculate file hash"""
    try:
        hash_func = getattr(hashlib, hash_algorithm)()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return None

def get_file_size(file_path: str) -> Optional[int]:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Failed to get size for {file_path}: {e}")
        return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_text_sample(text: str, start_chars: int = 200, end_chars: int = 200) -> Tuple[str, str]:
    """Extract start and end samples from text"""
    if len(text) <= start_chars + end_chars:
        return text, text
    
    start_part = text[:start_chars]
    end_part = text[-end_chars:]
    return start_part, end_part

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    import re
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text_for_analysis(text: str) -> str:
    """Clean text for analysis purposes"""
    # Remove excessive whitespace
    text = normalize_whitespace(text)
    
    # Remove or normalize special characters that might interfere
    import re
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', ' ', text)
    
    return text.strip()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def extract_numbers(text: str) -> List[int]:
    """Extract all numbers from text"""
    import re
    numbers = re.findall(r'\d+', text)
    return [int(n) for n in numbers]

def is_likely_page_number(text: str, number: int) -> bool:
    """Check if a number is likely a page number based on context"""
    text_lower = text.lower()
    
    # Check for page-related keywords
    page_keywords = ["page", "trang", "tr.", "p.", "số trang"]
    has_page_keyword = any(keyword in text_lower for keyword in page_keywords)
    
    # Check if number is in reasonable page range
    reasonable_page = 1 <= number <= 9999
    
    # Check if number appears alone or with minimal context
    words = text.split()
    number_str = str(number)
    number_alone = any(word.strip() == number_str for word in words)
    
    return has_page_keyword and reasonable_page or (number_alone and reasonable_page)

def detect_language(text: str) -> str:
    """Simple language detection for Vietnamese vs English"""
    vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    
    vietnamese_count = sum(1 for char in text.lower() if char in vietnamese_chars)
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return "unknown"
    
    vietnamese_ratio = vietnamese_count / total_chars
    
    if vietnamese_ratio > 0.1:  # More than 10% Vietnamese characters
        return "vietnamese"
    else:
        return "english"

def merge_dicts_deep(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    
    return result

def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def clamp(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def percentage(part: Union[int, float], whole: Union[int, float]) -> float:
    """Calculate percentage safely"""
    return safe_divide(part * 100, whole, 0.0)

def create_progress_tracker(total_items: int, description: str = "Processing") -> 'ProgressTracker':
    """Create a progress tracker"""
    return ProgressTracker(total_items, description)

class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1) -> Dict[str, Any]:
        """Update progress and return current status"""
        self.current_item += increment
        current_time = time.time()
        
        # Only log every 5 seconds or at completion
        if current_time - self.last_update >= 5 or self.current_item >= self.total_items:
            self.last_update = current_time
            
            elapsed = current_time - self.start_time
            progress_ratio = self.current_item / max(self.total_items, 1)
            
            status = {
                "description": self.description,
                "current": self.current_item,
                "total": self.total_items,
                "percentage": progress_ratio * 100,
                "elapsed_seconds": elapsed,
                "items_per_second": safe_divide(self.current_item, elapsed, 0),
                "completed": self.current_item >= self.total_items
            }
            
            if progress_ratio > 0 and not status["completed"]:
                estimated_total = elapsed / progress_ratio
                status["estimated_remaining_seconds"] = estimated_total - elapsed
            
            logger.info(f"{self.description}: {self.current_item}/{self.total_items} ({status['percentage']:.1f}%)")
            
            return status
        
        return {"current": self.current_item, "total": self.total_items}

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
    """Retry function with exponential backoff"""
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    delay = base_delay * (backoff_factor ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {str(e)}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    return wrapper

def timing_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
    
    return wrapper

# Export utility functions
__all__ = [
    "generate_processing_id", "generate_document_id", "generate_fragment_id",
    "cleanup_temp_file", "create_temp_directory", "cleanup_temp_directory",
    "ensure_directory_exists", "get_file_hash", "get_file_size", "format_file_size",
    "format_duration", "truncate_text", "extract_text_sample", "normalize_whitespace",
    "clean_text_for_analysis", "calculate_text_similarity", "extract_numbers",
    "is_likely_page_number", "detect_language", "merge_dicts_deep", "safe_divide",
    "clamp", "percentage", "create_progress_tracker", "ProgressTracker",
    "retry_with_backoff", "timing_decorator"
]