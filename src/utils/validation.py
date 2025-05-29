"""
Validation utilities for PDF processing system
"""
import os
import mimetypes
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_pdf_file(file) -> Dict[str, Any]:
    """
    Validate uploaded PDF file
    
    Args:
        file: UploadFile object from FastAPI
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_info": {}
    }
    
    try:
        # Check filename
        if not file.filename:
            validation_result["errors"].append("No filename provided")
            validation_result["valid"] = False
            return validation_result
        
        validation_result["file_info"]["filename"] = file.filename
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext != '.pdf':
            validation_result["errors"].append(f"Invalid file extension: {file_ext}. Only .pdf files are supported.")
            validation_result["valid"] = False
        
        # Check MIME type if available
        if hasattr(file, 'content_type') and file.content_type:
            validation_result["file_info"]["content_type"] = file.content_type
            
            if file.content_type != 'application/pdf':
                validation_result["warnings"].append(f"Unexpected MIME type: {file.content_type}")
        
        # Additional filename checks
        if len(file.filename) > 255:
            validation_result["errors"].append("Filename too long (max 255 characters)")
            validation_result["valid"] = False
        
        # Check for potentially problematic characters
        problematic_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in file.filename for char in problematic_chars):
            validation_result["warnings"].append("Filename contains potentially problematic characters")
        
    except Exception as e:
        logger.error(f"Error validating PDF file: {e}")
        validation_result["errors"].append(f"Validation error: {str(e)}")
        validation_result["valid"] = False
    
    return validation_result

def validate_file_size(file_size_bytes: int, max_size_mb: int) -> Dict[str, Any]:
    """
    Validate file size against limits
    
    Args:
        file_size_bytes: File size in bytes
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        Dict with validation results
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    validation_result = {
        "valid": file_size_bytes <= max_size_bytes,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_bytes / (1024 * 1024),
        "max_size_mb": max_size_mb,
        "error": None
    }
    
    if not validation_result["valid"]:
        validation_result["error"] = (
            f"File size ({validation_result['file_size_mb']:.1f}MB) "
            f"exceeds maximum allowed size ({max_size_mb}MB)"
        )
    
    return validation_result

def validate_pdf_path(pdf_path: str) -> Dict[str, Any]:
    """
    Validate PDF file path
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_info": {}
    }
    
    try:
        # Check if path exists
        if not os.path.exists(pdf_path):
            validation_result["errors"].append(f"File does not exist: {pdf_path}")
            validation_result["valid"] = False
            return validation_result
        
        # Check if it's a file (not directory)
        if not os.path.isfile(pdf_path):
            validation_result["errors"].append(f"Path is not a file: {pdf_path}")
            validation_result["valid"] = False
            return validation_result
        
        # Get file info
        file_stat = os.stat(pdf_path)
        validation_result["file_info"]["size_bytes"] = file_stat.st_size
        validation_result["file_info"]["modified_time"] = file_stat.st_mtime
        
        # Check file extension
        file_ext = Path(pdf_path).suffix.lower()
        if file_ext != '.pdf':
            validation_result["errors"].append(f"Invalid file extension: {file_ext}")
            validation_result["valid"] = False
        
        # Check file permissions
        if not os.access(pdf_path, os.R_OK):
            validation_result["errors"].append(f"Cannot read file: {pdf_path}")
            validation_result["valid"] = False
        
        # Check if file is empty
        if file_stat.st_size == 0:
            validation_result["errors"].append("File is empty")
            validation_result["valid"] = False
        
        # Basic file header check (PDF magic number)
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    validation_result["warnings"].append("File does not have PDF magic number")
        except Exception as e:
            validation_result["warnings"].append(f"Could not read file header: {e}")
    
    except Exception as e:
        logger.error(f"Error validating PDF path: {e}")
        validation_result["errors"].append(f"Path validation error: {str(e)}")
        validation_result["valid"] = False
    
    return validation_result

def validate_processing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate processing configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "normalized_config": {}
    }
    
    try:
        # Define expected config structure
        config_schema = {
            "shuffle_for_testing": {"type": bool, "default": False},
            "workflow_type": {"type": str, "default": "standard", "choices": ["standard", "testing", "production"]},
            "enable_hybrid_mode": {"type": bool, "default": True},
            "timeout_seconds": {"type": (int, float), "default": 300, "min": 30, "max": 3600},
            "max_file_size_mb": {"type": (int, float), "default": 50, "min": 1, "max": 500},
            "min_ocr_confidence": {"type": (float,), "default": 0.7, "min": 0.0, "max": 1.0}
        }
        
        # Validate and normalize each config key
        for key, schema in config_schema.items():
            value = config.get(key, schema["default"])
            
            # Type validation
            if not isinstance(value, schema["type"]):
                if isinstance(schema["type"], tuple):
                    type_names = " or ".join(t.__name__ for t in schema["type"])
                else:
                    type_names = schema["type"].__name__
                
                validation_result["errors"].append(
                    f"Config '{key}' must be of type {type_names}, got {type(value).__name__}"
                )
                validation_result["valid"] = False
                continue
            
            # Choice validation
            if "choices" in schema and value not in schema["choices"]:
                validation_result["errors"].append(
                    f"Config '{key}' must be one of {schema['choices']}, got '{value}'"
                )
                validation_result["valid"] = False
                continue
            
            # Range validation
            if "min" in schema and value < schema["min"]:
                validation_result["errors"].append(
                    f"Config '{key}' must be >= {schema['min']}, got {value}"
                )
                validation_result["valid"] = False
                continue
                
            if "max" in schema and value > schema["max"]:
                validation_result["errors"].append(
                    f"Config '{key}' must be <= {schema['max']}, got {value}"
                )
                validation_result["valid"] = False
                continue
            
            validation_result["normalized_config"][key] = value
        
        # Check for unknown config keys
        unknown_keys = set(config.keys()) - set(config_schema.keys())
        if unknown_keys:
            validation_result["warnings"].append(f"Unknown config keys: {list(unknown_keys)}")
    
    except Exception as e:
        logger.error(f"Error validating processing config: {e}")
        validation_result["errors"].append(f"Config validation error: {str(e)}")
        validation_result["valid"] = False
    
    return validation_result

def validate_fragment_data(fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate fragment data structure
    
    Args:
        fragments: List of fragment dictionaries
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "fragment_count": len(fragments),
        "statistics": {}
    }
    
    try:
        if not fragments:
            validation_result["errors"].append("No fragments provided")
            validation_result["valid"] = False
            return validation_result
        
        # Required fragment fields
        required_fields = ["id", "content"]
        optional_fields = ["start_part", "end_part", "original_page", "confidence"]
        
        # Statistics
        empty_content_count = 0
        short_content_count = 0
        missing_ids = 0
        duplicate_ids = set()
        fragment_ids = []
        
        for i, fragment in enumerate(fragments):
            # Check if fragment is a dictionary
            if not isinstance(fragment, dict):
                validation_result["errors"].append(f"Fragment {i} is not a dictionary")
                validation_result["valid"] = False
                continue
            
            # Check required fields
            for field in required_fields:
                if field not in fragment:
                    validation_result["errors"].append(f"Fragment {i} missing required field: {field}")
                    validation_result["valid"] = False
                elif field == "id" and not fragment[field]:
                    missing_ids += 1
            
            # Track fragment IDs for duplicate check
            if "id" in fragment and fragment["id"]:
                if fragment["id"] in fragment_ids:
                    duplicate_ids.add(fragment["id"])
                fragment_ids.append(fragment["id"])
            
            # Check content quality
            content = fragment.get("content", "")
            if not content or not content.strip():
                empty_content_count += 1
            elif len(content.strip()) < 50:
                short_content_count += 1
            
            # Validate confidence if present
            if "confidence" in fragment:
                confidence = fragment["confidence"]
                if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                    validation_result["warnings"].append(
                        f"Fragment {i} has invalid confidence value: {confidence}"
                    )
        
        # Update statistics
        validation_result["statistics"] = {
            "total_fragments": len(fragments),
            "empty_content": empty_content_count,
            "short_content": short_content_count,
            "missing_ids": missing_ids,
            "duplicate_ids": len(duplicate_ids),
            "quality_score": 1.0 - (empty_content_count + short_content_count) / len(fragments)
        }
        
        # Add warnings based on statistics
        if empty_content_count > 0:
            validation_result["warnings"].append(f"{empty_content_count} fragments have empty content")
        
        if short_content_count > len(fragments) * 0.5:
            validation_result["warnings"].append(f"{short_content_count} fragments have very short content")
        
        if duplicate_ids:
            validation_result["errors"].append(f"Duplicate fragment IDs found: {list(duplicate_ids)}")
            validation_result["valid"] = False
        
        if missing_ids > 0:
            validation_result["errors"].append(f"{missing_ids} fragments have missing or empty IDs")
            validation_result["valid"] = False
    
    except Exception as e:
        logger.error(f"Error validating fragment data: {e}")
        validation_result["errors"].append(f"Fragment validation error: {str(e)}")
        validation_result["valid"] = False
    
    return validation_result

def validate_ordering_result(ordered_fragment_ids: List[str], original_fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate ordering result against original fragments
    
    Args:
        ordered_fragment_ids: List of ordered fragment IDs
        original_fragments: List of original fragment dictionaries
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "completeness": 0.0,
        "duplicates": []
    }
    
    try:
        # Get original fragment IDs
        original_ids = [f.get("id") for f in original_fragments if f.get("id")]
        original_ids_set = set(original_ids)
        ordered_ids_set = set(ordered_fragment_ids)
        
        # Check completeness
        missing_ids = original_ids_set - ordered_ids_set
        extra_ids = ordered_ids_set - original_ids_set
        
        if missing_ids:
            validation_result["errors"].append(f"Missing fragment IDs in ordering: {list(missing_ids)}")
            validation_result["valid"] = False
        
        if extra_ids:
            validation_result["errors"].append(f"Extra fragment IDs in ordering: {list(extra_ids)}")
            validation_result["valid"] = False
        
        # Check for duplicates in ordering
        if len(ordered_fragment_ids) != len(set(ordered_fragment_ids)):
            duplicates = [id for id in ordered_fragment_ids if ordered_fragment_ids.count(id) > 1]
            validation_result["duplicates"] = list(set(duplicates))
            validation_result["errors"].append(f"Duplicate IDs in ordering: {validation_result['duplicates']}")
            validation_result["valid"] = False
        
        # Calculate completeness score
        if original_ids:
            common_ids = original_ids_set & ordered_ids_set
            validation_result["completeness"] = len(common_ids) / len(original_ids)
        else:
            validation_result["completeness"] = 1.0 if not ordered_fragment_ids else 0.0
        
        # Add warnings for low completeness
        if validation_result["completeness"] < 1.0:
            validation_result["warnings"].append(f"Ordering completeness: {validation_result['completeness']:.1%}")
    
    except Exception as e:
        logger.error(f"Error validating ordering result: {e}")
        validation_result["errors"].append(f"Ordering validation error: {str(e)}")
        validation_result["valid"] = False
    
    return validation_result

def validate_api_request(request_data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """
    Validate API request data for specific endpoint
    
    Args:
        request_data: Request data dictionary
        endpoint: Endpoint name
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Define endpoint-specific validation rules
        endpoint_rules = {
            "process_pdf": {
                "optional_fields": {
                    "shuffle_for_testing": bool,
                    "workflow_type": str,
                    "enable_hybrid_mode": bool
                }
            },
            "analyze_pdf": {
                "optional_fields": {
                    "workflow_type": str
                }
            },
            "batch_process": {
                "optional_fields": {
                    "workflow_type": str,
                    "max_concurrent": int
                }
            }
        }
        
        if endpoint not in endpoint_rules:
            validation_result["warnings"].append(f"No validation rules defined for endpoint: {endpoint}")
            return validation_result
        
        rules = endpoint_rules[endpoint]
        
        # Validate optional fields
        for field, expected_type in rules.get("optional_fields", {}).items():
            if field in request_data:
                value = request_data[field]
                if not isinstance(value, expected_type):
                    validation_result["errors"].append(
                        f"Field '{field}' must be of type {expected_type.__name__}, got {type(value).__name__}"
                    )
                    validation_result["valid"] = False
        
        # Endpoint-specific validations
        if endpoint == "batch_process":
            max_concurrent = request_data.get("max_concurrent", 3)
            if max_concurrent < 1 or max_concurrent > 10:
                validation_result["errors"].append("max_concurrent must be between 1 and 10")
                validation_result["valid"] = False
    
    except Exception as e:
        logger.error(f"Error validating API request: {e}")
        validation_result["errors"].append(f"Request validation error: {str(e)}")
        validation_result["valid"] = False
    
    return validation_result

# Export validation functions
__all__ = [
    "validate_pdf_file",
    "validate_file_size", 
    "validate_pdf_path",
    "validate_processing_config",
    "validate_fragment_data",
    "validate_ordering_result",
    "validate_api_request"
]