"""
Base node class for LangGraph workflow nodes
"""
from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, Any, Optional
from ..models.schemas import ProcessingState

class BaseNode(ABC):
    """Abstract base class for all workflow nodes"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._step_start_time = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Node name identifier"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Node description"""
        pass
    
    @property
    def version(self) -> str:
        """Node version"""
        return "1.0.0"
    
    @abstractmethod
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Execute the node logic"""
        pass
    
    def validate_input(self, state: ProcessingState) -> bool:
        """Validate input state - override in subclasses"""
        return isinstance(state, dict)
    
    def _log_step_start(self, state: ProcessingState):
        """Log the start of node execution"""
        self._step_start_time = time.time()
        self.logger.info(f"Starting {self.name} for document {state.get('document_id', 'unknown')}")
    
    def _log_step_end(self, state: ProcessingState):
        """Log the end of node execution with timing"""
        if self._step_start_time:
            duration = time.time() - self._step_start_time
            self.logger.info(f"Completed {self.name} in {duration:.3f}s")
            
            # Store timing in state
            step_timings = state.get("step_timings", {})
            step_timings[self.name] = duration
            state["step_timings"] = step_timings
    
    def _handle_error(self, state: ProcessingState, error: Exception, context: str = ""):
        """Handle errors with consistent logging and state updates"""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(f"Error in {self.name}: {error_msg}")
        
        # Add to state errors
        errors = state.get("errors", [])
        errors.append(f"{self.name}: {error_msg}")
        state["errors"] = errors
    
    def _add_warning(self, state: ProcessingState, message: str):
        """Add warning message to state"""
        self.logger.warning(f"{self.name}: {message}")
        
        debug_info = state.get("debug_info", {})
        warnings = debug_info.get("warnings", [])
        warnings.append(f"{self.name}: {message}")
        debug_info["warnings"] = warnings
        state["debug_info"] = debug_info
    
    def _update_progress(self, state: ProcessingState, progress_info: Dict[str, Any]):
        """Update progress information in state"""
        debug_info = state.get("debug_info", {})
        node_progress = debug_info.get("node_progress", {})
        node_progress[self.name] = progress_info
        debug_info["node_progress"] = node_progress
        state["debug_info"] = debug_info

class ConditionalNode(BaseNode):
    """Base class for nodes that make routing decisions"""
    
    @abstractmethod
    def get_next_node(self, state: ProcessingState) -> str:
        """Determine the next node based on state"""
        pass
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Execute conditional logic and set routing decision"""
        self._log_step_start(state)
        
        try:
            next_node = self.get_next_node(state)
            state["_next_node"] = next_node
            self.logger.info(f"{self.name} routing to: {next_node}")
            
        except Exception as e:
            self._handle_error(state, e, "Conditional routing failed")
            state["_next_node"] = "fallback"
        
        finally:
            self._log_step_end(state)
        
        return state

class ValidationNode(BaseNode):
    """Base class for nodes that validate data"""
    
    def __init__(self):
        super().__init__()
        self.validation_rules = self.get_validation_rules()
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Define validation rules for this node"""
        pass
    
    def validate_state(self, state: ProcessingState) -> Dict[str, Any]:
        """Validate state against rules"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "details": {}
        }
        
        for rule_name, rule_config in self.validation_rules.items():
            try:
                result = self._apply_validation_rule(state, rule_name, rule_config)
                validation_results["details"][rule_name] = result
                
                if not result.get("valid", True):
                    validation_results["valid"] = False
                    validation_results["errors"].extend(result.get("errors", []))
                
                validation_results["warnings"].extend(result.get("warnings", []))
                
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Validation rule '{rule_name}' failed: {str(e)}")
        
        return validation_results
    
    def _apply_validation_rule(self, state: ProcessingState, rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single validation rule"""
        # Override in subclasses for specific validation logic
        return {"valid": True, "errors": [], "warnings": []}

class TransformationNode(BaseNode):
    """Base class for nodes that transform data"""
    
    @abstractmethod
    def transform_data(self, input_data: Any, state: ProcessingState) -> Any:
        """Transform input data"""
        pass
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Execute transformation"""
        self._log_step_start(state)
        
        try:
            input_key = getattr(self, 'INPUT_KEY', 'fragments')
            output_key = getattr(self, 'OUTPUT_KEY', 'transformed_data')
            
            input_data = state.get(input_key)
            if input_data is None:
                self._add_warning(state, f"No input data found for key: {input_key}")
                return state
            
            transformed_data = self.transform_data(input_data, state)
            state[output_key] = transformed_data
            
            self.logger.info(f"Transformed {len(input_data) if hasattr(input_data, '__len__') else 'N/A'} items")
            
        except Exception as e:
            self._handle_error(state, e, "Data transformation failed")
        
        finally:
            self._log_step_end(state)
        
        return state

class AggregationNode(BaseNode):
    """Base class for nodes that aggregate results from multiple sources"""
    
    @abstractmethod
    def aggregate_results(self, results: Dict[str, Any], state: ProcessingState) -> Any:
        """Aggregate multiple results"""
        pass
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Execute aggregation"""
        self._log_step_start(state)
        
        try:
            # Collect results from various analysis steps
            results_to_aggregate = {}
            
            for key in ["page_hints", "section_analysis", "continuity_scores"]:
                if key in state:
                    results_to_aggregate[key] = state[key]
            
            if not results_to_aggregate:
                self._add_warning(state, "No results available for aggregation")
                return state
            
            aggregated_result = self.aggregate_results(results_to_aggregate, state)
            state["aggregated_analysis"] = aggregated_result
            
            self.logger.info(f"Aggregated {len(results_to_aggregate)} result sets")
            
        except Exception as e:
            self._handle_error(state, e, "Result aggregation failed")
        
        finally:
            self._log_step_end(state)
        
        return state

class ParallelNode(BaseNode):
    """Base class for nodes that can execute in parallel"""
    
    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.max_workers = max_workers
    
    @abstractmethod
    def get_parallel_tasks(self, state: ProcessingState) -> List[Dict[str, Any]]:
        """Define parallel tasks to execute"""
        pass
    
    @abstractmethod
    def process_task(self, task: Dict[str, Any], state: ProcessingState) -> Any:
        """Process a single task"""
        pass
    
    @abstractmethod
    def combine_results(self, task_results: List[Any], state: ProcessingState) -> Any:
        """Combine results from parallel tasks"""
        pass
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Execute tasks in parallel"""
        self._log_step_start(state)
        
        try:
            import concurrent.futures
            
            tasks = self.get_parallel_tasks(state)
            if not tasks:
                self._add_warning(state, "No parallel tasks defined")
                return state
            
            task_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.process_task, task, state): task 
                    for task in tasks
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        task_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parallel task failed: {e}")
                        task_results.append(None)
            
            # Combine results
            if task_results:
                combined_result = self.combine_results(task_results, state)
                state["parallel_results"] = combined_result
            
            self.logger.info(f"Completed {len(tasks)} parallel tasks")
            
        except Exception as e:
            self._handle_error(state, e, "Parallel execution failed")
        
        finally:
            self._log_step_end(state)
        
        return state

# Node registry for dynamic loading
NODE_REGISTRY = {}

def register_node(node_class):
    """Decorator to register node classes"""
    NODE_REGISTRY[node_class.__name__] = node_class
    return node_class

def get_node_by_name(name: str) -> Optional[BaseNode]:
    """Get node instance by name"""
    if name in NODE_REGISTRY:
        return NODE_REGISTRY[name]()
    return None

def list_available_nodes() -> List[str]:
    """List all registered node names"""
    return list(NODE_REGISTRY.keys())

# Export base classes
__all__ = [
    "BaseNode", "ConditionalNode", "ValidationNode", 
    "TransformationNode", "AggregationNode", "ParallelNode",
    "register_node", "get_node_by_name", "list_available_nodes"
]