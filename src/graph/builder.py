"""
LangGraph workflow builder for PDF page ordering
"""
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, Optional, Callable
import logging
from ..models.schemas import ProcessingState
from ..nodes.extraction_nodes import PDFExtractionNode, FragmentProcessingNode, ShuffleNode
from ..nodes.analysis_nodes import PageHintExtractionNode, SectionAnalysisNode, ContinuityAnalysisNode
from ..nodes.ordering_nodes import (
    OrderingDecisionNode, PageNumberOrderingNode, 
    SectionOrderingNode, ContinuityOrderingNode, HybridOrderingNode
)
from ..config.settings import get_workflow_config

logger = logging.getLogger(__name__)

class WorkflowBuilder:
    """Builder class for creating LangGraph workflows"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_workflow_config("default")
        self.nodes = {}
        self.conditional_edges = {}
        self.linear_edges = []
        self.entry_point = None
        
    def add_node(self, name: str, node_instance, override: bool = False):
        """Add a node to the workflow"""
        if name in self.nodes and not override:
            raise ValueError(f"Node '{name}' already exists. Use override=True to replace.")
        
        self.nodes[name] = node_instance
        logger.debug(f"Added node: {name}")
        return self
    
    def add_edge(self, from_node: str, to_node: str):
        """Add a linear edge between nodes"""
        self.linear_edges.append((from_node, to_node))
        logger.debug(f"Added edge: {from_node} -> {to_node}")
        return self
    
    def add_conditional_edge(self, from_node: str, condition_func: Callable, routing_map: Dict[str, str]):
        """Add a conditional edge with routing logic"""
        self.conditional_edges[from_node] = {
            "condition": condition_func,
            "routing": routing_map
        }
        logger.debug(f"Added conditional edge from {from_node} with {len(routing_map)} routes")
        return self
    
    def set_entry_point(self, node_name: str):
        """Set the workflow entry point"""
        if node_name not in self.nodes:
            raise ValueError(f"Entry point node '{node_name}' not found")
        
        self.entry_point = node_name
        logger.debug(f"Set entry point: {node_name}")
        return self
    
    def build(self) -> StateGraph:
        """Build the complete workflow graph"""
        if not self.entry_point:
            raise ValueError("Entry point must be set before building")
        
        if not self.nodes:
            raise ValueError("At least one node must be added before building")
        
        # Create StateGraph
        workflow = StateGraph(ProcessingState)
        
        # Add all nodes
        for name, node_instance in self.nodes.items():
            workflow.add_node(name, node_instance.execute)
            logger.debug(f"Added node to graph: {name}")
        
        # Set entry point
        workflow.set_entry_point(self.entry_point)
        
        # Add linear edges
        for from_node, to_node in self.linear_edges:
            workflow.add_edge(from_node, to_node)
        
        # Add conditional edges
        for from_node, edge_config in self.conditional_edges.items():
            workflow.add_conditional_edges(
                from_node,
                edge_config["condition"],
                edge_config["routing"]
            )
        
        logger.info(f"Built workflow with {len(self.nodes)} nodes and {len(self.linear_edges)} linear edges")
        return workflow.compile()

class StandardWorkflowBuilder(WorkflowBuilder):
    """Pre-configured builder for standard PDF ordering workflow"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._setup_standard_workflow()
    
    def _setup_standard_workflow(self):
        """Setup the standard PDF ordering workflow"""
        
        # Initialize nodes
        extraction_node = PDFExtractionNode()
        fragment_processing_node = FragmentProcessingNode()
        shuffle_node = ShuffleNode()
        page_hint_node = PageHintExtractionNode()
        section_analysis_node = SectionAnalysisNode()
        continuity_analysis_node = ContinuityAnalysisNode()
        ordering_decision_node = OrderingDecisionNode()
        page_ordering_node = PageNumberOrderingNode()
        section_ordering_node = SectionOrderingNode()
        continuity_ordering_node = ContinuityOrderingNode()
        hybrid_ordering_node = HybridOrderingNode()
        
        # Add nodes to workflow
        self.add_node("extract", extraction_node)
        self.add_node("process_fragments", fragment_processing_node)
        self.add_node("shuffle", shuffle_node)
        self.add_node("extract_page_hints", page_hint_node)
        self.add_node("analyze_sections", section_analysis_node)
        self.add_node("analyze_continuity", continuity_analysis_node)
        self.add_node("decide_ordering", ordering_decision_node)
        self.add_node("page_ordering", page_ordering_node)
        self.add_node("section_ordering", section_ordering_node)
        self.add_node("continuity_ordering", continuity_ordering_node)
        self.add_node("hybrid_ordering", hybrid_ordering_node)
        
        # Set up linear workflow
        self.set_entry_point("extract")
        self.add_edge("extract", "process_fragments")
        self.add_edge("process_fragments", "shuffle")
        self.add_edge("shuffle", "extract_page_hints")
        
        # Parallel analysis (if supported)
        if self.config.get("parallel_analysis", False):
            self._setup_parallel_analysis()
        else:
            self._setup_sequential_analysis()
        
        # Decision point
        self.add_conditional_edge(
            "analyze_continuity",
            self._determine_ordering_strategy,
            {
                "page_ordering": "page_ordering",
                "section_ordering": "section_ordering", 
                "continuity_ordering": "continuity_ordering",
                "hybrid_ordering": "hybrid_ordering"
            }
        )
        
        # Terminal edges
        for ordering_node in ["page_ordering", "section_ordering", "continuity_ordering", "hybrid_ordering"]:
            self.add_edge(ordering_node, END)
    
    def _setup_sequential_analysis(self):
        """Setup sequential analysis workflow"""
        self.add_edge("extract_page_hints", "analyze_sections")
        self.add_edge("analyze_sections", "analyze_continuity")
    
    def _setup_parallel_analysis(self):
        """Setup parallel analysis workflow (simplified - would need custom node)"""
        # For now, use sequential as parallel requires more complex setup
        self._setup_sequential_analysis()
    
    def _determine_ordering_strategy(self, state: ProcessingState) -> str:
        """Determine which ordering strategy to use"""
        try:
            page_hints = state.get("page_hints", {})
            section_analysis = state.get("section_analysis", {})
            continuity_scores = state.get("continuity_scores", {})
            fragments_count = len(state.get("fragments", []))
            
            # Strategy 1: Direct page number ordering
            if page_hints:
                page_coverage = len(page_hints) / max(fragments_count, 1)
                unique_pages = len(set(page_hints.values()))
                
                if page_coverage >= 0.8 and unique_pages == len(page_hints):
                    logger.info("Using page number ordering strategy")
                    return "page_ordering"
            
            # Strategy 2: Section-based ordering
            if section_analysis.get("hierarchy_detected", False):
                continuity_score = section_analysis.get("continuity_score", 0.0)
                if continuity_score >= 0.7:
                    logger.info("Using section-based ordering strategy")
                    return "section_ordering"
            
            # Strategy 3: Hybrid approach
            if self.config.get("hybrid_mode_enabled", True):
                if page_hints or section_analysis.get("hierarchy_detected", False):
                    logger.info("Using hybrid ordering strategy")
                    return "hybrid_ordering"
            
            # Strategy 4: Continuity-based (fallback)
            logger.info("Using continuity-based ordering strategy")
            return "continuity_ordering"
            
        except Exception as e:
            logger.error(f"Error in ordering strategy decision: {e}")
            return "continuity_ordering"

class CustomWorkflowBuilder(WorkflowBuilder):
    """Builder for custom workflows with specific requirements"""
    
    def create_testing_workflow(self):
        """Create workflow optimized for testing with shuffling"""
        # Similar to standard but with forced shuffling and debug nodes
        pass
    
    def create_production_workflow(self):
        """Create workflow optimized for production"""
        # Optimized for speed and reliability
        pass
    
    def create_analysis_only_workflow(self):
        """Create workflow that only does analysis without ordering"""
        pass

class WorkflowFactory:
    """Factory for creating different types of workflows"""
    
    @staticmethod
    def create_standard_workflow(config: Optional[Dict[str, Any]] = None) -> StateGraph:
        """Create standard PDF ordering workflow"""
        builder = StandardWorkflowBuilder(config)
        return builder.build()
    
    @staticmethod
    def create_testing_workflow(config: Optional[Dict[str, Any]] = None) -> StateGraph:
        """Create workflow optimized for testing"""
        test_config = get_workflow_config("testing")
        if config:
            test_config.update(config)
        
        builder = StandardWorkflowBuilder(test_config)
        return builder.build()
    
    @staticmethod
    def create_production_workflow(config: Optional[Dict[str, Any]] = None) -> StateGraph:
        """Create workflow optimized for production"""
        prod_config = get_workflow_config("production")
        if config:
            prod_config.update(config)
        
        builder = StandardWorkflowBuilder(prod_config)
        return builder.build()
    
    @staticmethod
    def create_custom_workflow(
        node_configs: List[Dict[str, Any]], 
        flow_config: Dict[str, Any]
    ) -> StateGraph:
        """Create custom workflow from configuration"""
        builder = CustomWorkflowBuilder()
        
        # Add nodes from config
        for node_config in node_configs:
            # Dynamic node creation would go here
            pass
        
        return builder.build()

def validate_workflow(workflow: StateGraph) -> Dict[str, Any]:
    """Validate workflow configuration"""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "node_count": 0,
        "has_entry_point": False,
        "has_terminal_nodes": False
    }
    
    try:
        # Basic validation logic would go here
        # This is a simplified version
        validation_results["valid"] = True
        validation_results["node_count"] = len(workflow.nodes) if hasattr(workflow, 'nodes') else 0
        
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Workflow validation failed: {str(e)}")
    
    return validation_results

# Export main classes
__all__ = [
    "WorkflowBuilder", "StandardWorkflowBuilder", "CustomWorkflowBuilder",
    "WorkflowFactory", "validate_workflow"
]