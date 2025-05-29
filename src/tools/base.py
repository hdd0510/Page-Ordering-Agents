"""
Base classes for PDF processing tools
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class ToolResult(BaseModel):
    """Standard result format for all tools"""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class BaseTool(ABC):
    """Abstract base class for all processing tools"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._setup()
    
    def _setup(self):
        """Override for tool-specific setup"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @property
    def version(self) -> str:
        """Tool version"""
        return "1.0.0"
    
    @property
    def capabilities(self) -> List[str]:
        """List of tool capabilities"""
        return []
    
    def execute(self, *args, **kwargs) -> ToolResult:
        """Execute tool with timing and error handling"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing {self.name} with args: {args}, kwargs: {kwargs}")
            result = self._execute(*args, **kwargs)
            
            if not isinstance(result, ToolResult):
                # Convert legacy results to ToolResult
                if isinstance(result, dict):
                    result = ToolResult(
                        success=result.get("success", True),
                        data=result,
                        confidence=result.get("confidence", 0.0)
                    )
                else:
                    result = ToolResult(
                        success=True,
                        data={"result": result},
                        confidence=1.0
                    )
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            self.logger.info(f"{self.name} completed in {result.processing_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            self.logger.error(f"{self.name} failed after {error_time:.2f}ms: {str(e)}")
            
            return ToolResult(
                success=False,
                error_message=str(e),
                processing_time_ms=error_time,
                metadata={"error_type": type(e).__name__}
            )
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> Union[ToolResult, Dict[str, Any], Any]:
        """Implement the actual tool logic"""
        pass
    
    def validate_input(self, *args, **kwargs) -> bool:
        """Override for input validation"""
        return True
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback"""
        return self.config.get(key, default)

class ExtractionTool(BaseTool):
    """Base class for text/content extraction tools"""
    
    @property
    def capabilities(self) -> List[str]:
        return ["extraction", "text_processing"]
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        return {
            "length": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.split('\n')),
            "is_empty": not content.strip()
        }

class AnalysisTool(BaseTool):
    """Base class for content analysis tools"""
    
    @property
    def capabilities(self) -> List[str]:
        return ["analysis", "pattern_recognition"]
    
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score for analysis result"""
        # Override in subclasses for specific confidence calculation
        return 0.5

class OrderingTool(BaseTool):
    """Base class for ordering/sorting tools"""
    
    @property 
    def capabilities(self) -> List[str]:
        return ["ordering", "sorting", "ranking"]
    
    def validate_ordering(self, original_ids: List[str], ordered_ids: List[str]) -> bool:
        """Validate that ordering contains all original IDs"""
        return set(original_ids) == set(ordered_ids)
    
    def calculate_ordering_quality(self, ordering_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for ordering result"""
        return {
            "completeness": 1.0,  # All items included
            "consistency": 0.5,   # Override in subclasses
            "confidence": ordering_result.get("confidence", 0.5)
        }

class CompositeEd:
    """Tool that combines multiple other tools"""
    
    def __init__(self, tools: List[BaseTool], config: Optional[Dict[str, Any]] = None):
        self.tools = {tool.name: tool for tool in tools}
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def execute_pipeline(self, input_data: Any, tool_sequence: List[str]) -> List[ToolResult]:
        """Execute a sequence of tools in pipeline fashion"""
        results = []
        current_data = input_data
        
        for tool_name in tool_sequence:
            if tool_name not in self.tools:
                results.append(ToolResult(
                    success=False,
                    error_message=f"Tool '{tool_name}' not found"
                ))
                break
            
            tool = self.tools[tool_name]
            result = tool.execute(current_data)
            results.append(result)
            
            if not result.success:
                self.logger.error(f"Pipeline stopped at {tool_name}: {result.error_message}")
                break
            
            # Pass result data to next tool
            current_data = result.data
        
        return results
    
    def execute_parallel(self, input_data: Any, tool_names: List[str]) -> Dict[str, ToolResult]:
        """Execute multiple tools in parallel"""
        results = {}
        
        for tool_name in tool_names:
            if tool_name in self.tools:
                results[tool_name] = self.tools[tool_name].execute(input_data)
            else:
                results[tool_name] = ToolResult(
                    success=False,
                    error_message=f"Tool '{tool_name}' not found"
                )
        
        return results

class CacheableTool(BaseTool):
    """Base class for tools that support caching"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cache = {}
        self.cache_enabled = self.get_config_value("enable_cache", True)
        self.cache_ttl = self.get_config_value("cache_ttl_seconds", 3600)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from inputs"""
        import hashlib
        key_data = f"{self.name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def execute(self, *args, **kwargs) -> ToolResult:
        """Execute with caching support"""
        if not self.cache_enabled:
            return super().execute(*args, **kwargs)
        
        cache_key = self._get_cache_key(*args, **kwargs)
        
        # Check cache
        if cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.logger.debug(f"Cache hit for {self.name}")
                return cached_result
            else:
                # Expired cache entry
                del self.cache[cache_key]
        
        # Execute and cache result
        result = super().execute(*args, **kwargs)
        if result.success:
            self.cache[cache_key] = (result, time.time())
        
        return result
    
    def clear_cache(self):
        """Clear the tool cache"""
        self.cache.clear()
        self.logger.info(f"Cache cleared for {self.name}")

# Utility functions
def create_tool_registry(tools: List[BaseTool]) -> Dict[str, BaseTool]:
    """Create a registry of tools by name"""
    return {tool.name: tool for tool in tools}

def validate_tool_chain(tools: List[BaseTool], required_capabilities: List[str]) -> bool:
    """Validate that a chain of tools provides required capabilities"""
    available_capabilities = set()
    for tool in tools:
        available_capabilities.update(tool.capabilities)
    
    return all(cap in available_capabilities for cap in required_capabilities)

# Export key classes
__all__ = [
    "ToolResult", "BaseTool", "ExtractionTool", "AnalysisTool", 
    "OrderingTool", "CompositeTool", "CacheableTool",
    "create_tool_registry", "validate_tool_chain"
]