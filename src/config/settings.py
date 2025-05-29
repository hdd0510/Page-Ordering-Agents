"""
Configuration management for PDF Page Ordering System
"""
import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "PDF Page Ordering Agent API"
    api_version: str = "2.0.0"
    
    # Google AI Configuration
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash-preview-0417"
    gemini_temperature: float = 0.1
    
    # OCR Configuration
    min_ocr_confidence: float = 0.7
    tesseract_lang: str = "vie+eng"
    ocr_dpi: int = 300
    
    # Processing Configuration
    max_file_size_mb: int = 50
    temp_dir: str = "/tmp"
    enable_shuffling: bool = True
    
    # Performance Configuration
    max_workers: int = 4
    timeout_seconds: int = 300
    cache_ttl_seconds: int = 3600
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Tool Configuration
    page_hint_confidence_threshold: float = 0.8
    continuity_score_threshold: float = 0.3
    section_analysis_enabled: bool = True
    hybrid_mode_enabled: bool = True
    
    # Monitoring Configuration  
    enable_tracing: bool = False
    langsmith_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Tool-specific configurations
TOOL_CONFIGS = {
    "page_hint_extractor": {
        "patterns": {
            "vietnamese": [
                r"(?:trang|trag)\s*([1-9]\d{0,3})",
                r"(?:tr\.|tr)\s*([1-9]\d{0,3})",
                r"-\s*([1-9]\d{0,3})\s*-"
            ],
            "english": [
                r"(?:page|p\.?)\s*([1-9]\d{0,3})",
                r"([1-9]\d{0,3})\s*(?:\/\d+)?$"
            ],
            "standalone": [
                r"(?:\b|_)([1-9]\d{0,3})(?:\b|\s*$)",
                r"^\s*([1-9]\d{0,3})\s*$"
            ]
        },
        "confidence_weights": {
            "vietnamese": 0.9,
            "english": 0.85,
            "standalone": 0.6
        }
    },
    
    "section_analyzer": {
        "patterns": {
            "chapters": [
                r"(?:chương|chapter|phần)\s*([IVX]+|\d+)",
                r"(?:ch\.|chap\.)\s*(\d+)"
            ],
            "sections": [
                r"^(\d+)\.\s*[A-ZÀÁẠẢÃÂẦẤẬẨẪ]",
                r"(?:mục|section)\s*(\d+)"
            ],
            "subsections": [
                r"^(\d+)\.(\d+)\s*[A-Za-z]",
                r"^\s*([a-z])\)\s*[A-ZÀÁẠẢÃÂẦẤẬẨẪ]"
            ]
        }
    },
    
    "continuity_analyzer": {
        "weights": {
            "syntactic": 0.2,
            "semantic": 0.4,
            "lexical": 0.2,
            "structural": 0.2
        },
        "stop_words": {
            "vietnamese": {'và', 'hoặc', 'nhưng', 'tuy nhiên', 'do đó', 'vì vậy'},
            "english": {'the', 'and', 'or', 'but', 'however', 'therefore'}
        },
        "transitions": {
            "vietnamese": ['tuy nhiên', 'do đó', 'vì vậy', 'mặt khác', 'bên cạnh đó'],
            "english": ['however', 'therefore', 'moreover', 'furthermore', 'besides']
        }
    }
}

# Workflow configuration
WORKFLOW_CONFIGS = {
    "default": {
        "shuffle_enabled": False,
        "parallel_analysis": True,
        "fallback_strategies": ["page_hints", "sections", "continuity", "hybrid"]
    },
    "testing": {
        "shuffle_enabled": True,
        "parallel_analysis": False,
        "debug_mode": True
    },
    "production": {
        "shuffle_enabled": False,
        "parallel_analysis": True,
        "enable_caching": True,
        "timeout_multiplier": 1.5
    }
}

def get_tool_config(tool_name: str) -> Dict[str, Any]:
    """Get configuration for specific tool"""
    return TOOL_CONFIGS.get(tool_name, {})

def get_workflow_config(mode: str = "default") -> Dict[str, Any]:
    """Get workflow configuration for specific mode"""
    return WORKFLOW_CONFIGS.get(mode, WORKFLOW_CONFIGS["default"])