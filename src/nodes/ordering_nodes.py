"""
Ordering nodes for LangGraph workflow
"""
import logging
from typing import Dict, Any, List, Tuple
from .base_node import BaseNode, ConditionalNode
from ..models.schemas import ProcessingState, OrderingMethod
from ..tools.ordering.page_number_sorter import PageNumberSorter
from ..tools.ordering.section_sorter import SectionSorter
from ..tools.ordering.continuity_sorter import ContinuitySorter
from ..tools.ordering.hybrid_sorter import HybridSorter
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class OrderingDecisionNode(ConditionalNode):
    """Node that decides which ordering strategy to use"""
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "ordering_decision"
    
    @property
    def description(self) -> str:
        return "Decide which ordering strategy to use based on analysis results"
    
    def get_next_node(self, state: ProcessingState) -> str:
        """Determine the best ordering strategy"""
        try:
            page_hints = state.get("page_hints", {})
            section_analysis = state.get("section_analysis", {})
            continuity_scores = state.get("continuity_scores", {})
            fragments = state.get("fragments", [])
            
            # Calculate strategy scores
            strategy_scores = self._calculate_strategy_scores(
                page_hints, section_analysis, continuity_scores, fragments
            )
            
            # Select best strategy
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            selected_strategy = best_strategy[0]
            
            # Update state with decision info
            state["selected_strategy"] = selected_strategy
            debug_info = state.get("debug_info", {})
            debug_info["ordering_decision"] = {
                "strategy_scores": strategy_scores,
                "selected_strategy": selected_strategy,
                "selection_confidence": best_strategy[1]
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Selected ordering strategy: {selected_strategy} "
                f"(score: {best_strategy[1]:.3f})"
            )
            
            # Map strategy to node name
            strategy_mapping = {
                "page_numbers": "page_ordering",
                "sections": "section_ordering",
                "continuity": "continuity_ordering", 
                "hybrid": "hybrid_ordering"
            }
            
            return strategy_mapping.get(selected_strategy, "continuity_ordering")
            
        except Exception as e:
            self.logger.error(f"Error in ordering decision: {e}")
            return "continuity_ordering"  # Fallback
    
    def _calculate_strategy_scores(
        self,
        page_hints: Dict[str, int],
        section_analysis: Dict[str, Any],
        continuity_scores: Dict[str, float],
        fragments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate scores for each ordering strategy"""
        
        fragments_count = len(fragments)
        scores = {}
        
        # Page number strategy score
        if page_hints:
            page_coverage = len(page_hints) / max(fragments_count, 1)
            unique_pages = len(set(page_hints.values()))
            has_duplicates = unique_pages != len(page_hints)
            
            page_score = page_coverage * 0.8
            if not has_duplicates:
                page_score += 0.2  # Bonus for no duplicates
            
            scores["page_numbers"] = min(page_score, 1.0)
        else:
            scores["page_numbers"] = 0.0
        
        # Section strategy score
        if section_analysis.get("hierarchy_detected", False):
            continuity_score = section_analysis.get("continuity_score", 0.0)
            fragments_with_sections = len(section_analysis.get("fragment_sections", {}))
            section_coverage = fragments_with_sections / max(fragments_count, 1)
            
            section_score = (continuity_score * 0.6) + (section_coverage * 0.4)
            scores["sections"] = section_score
        else:
            scores["sections"] = 0.0
        
        # Continuity strategy score
        if continuity_scores:
            avg_continuity = sum(continuity_scores.values()) / len(continuity_scores)
            high_continuity_pairs = sum(1 for s in continuity_scores.values() if s > 0.7)
            continuity_ratio = high_continuity_pairs / len(continuity_scores)
            
            continuity_score = (avg_continuity * 0.7) + (continuity_ratio * 0.3)
            scores["continuity"] = continuity_score
        else:
            scores["continuity"] = 0.3  # Default baseline
        
        # Hybrid strategy score (combination of above)
        if scores["page_numbers"] > 0.5 or scores["sections"] > 0.5:
            hybrid_components = [s for s in [scores["page_numbers"], scores["sections"]] if s > 0.5]
            hybrid_score = (sum(hybrid_components) / len(hybrid_components)) * 0.9  # Slight penalty for complexity
            scores["hybrid"] = hybrid_score
        else:
            scores["hybrid"] = 0.0
        
        return scores

class PageNumberOrderingNode(BaseNode):
    """Node for ordering fragments by extracted page numbers"""
    
    def __init__(self):
        super().__init__()
        self.page_sorter = PageNumberSorter()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "page_number_ordering"
    
    @property
    def description(self) -> str:
        return "Order fragments using extracted page numbers"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Order fragments by page numbers"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments = state.get("fragments", [])
            page_hints = state.get("page_hints", {})
            
            if not fragments:
                self._add_warning(state, "No fragments available for ordering")
                state["final_order"] = []
                state["confidence_score"] = 0.0
                return state
            
            # Execute page number sorting
            sorting_result = self.page_sorter.execute(fragments, page_hints)
            
            if not sorting_result.success:
                self._handle_error(
                    state,
                    Exception(sorting_result.error_message),
                    "Page number sorting failed"
                )
                # Fallback to original order
                state["final_order"] = [f["id"] for f in fragments]
                state["confidence_score"] = 0.0
                return state
            
            # Update state with results
            ordering_data = sorting_result.data
            state["final_order"] = ordering_data["ordered_fragment_ids"]
            state["confidence_score"] = ordering_data["confidence_score"]
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["page_ordering"] = {
                "fragments_ordered": len(ordering_data["ordered_fragment_ids"]),
                "page_hints_used": len(page_hints),
                "confidence_score": ordering_data["confidence_score"],
                "ordering_method": OrderingMethod.PAGE_NUMBERS,
                "processing_time_ms": sorting_result.processing_time_ms,
                "quality_metrics": ordering_data.get("quality_metrics", {})
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Page number ordering complete: {len(state['final_order'])} fragments, "
                f"confidence: {state['confidence_score']:.3f}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Page number ordering failed")
            # Fallback
            state["final_order"] = [f["id"] for f in state.get("fragments", [])]
            state["confidence_score"] = 0.0
        
        finally:
            self._log_step_end(state)
        
        return state

class SectionOrderingNode(BaseNode):
    """Node for ordering fragments by section structure"""
    
    def __init__(self):
        super().__init__()
        self.section_sorter = SectionSorter()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "section_ordering"
    
    @property
    def description(self) -> str:
        return "Order fragments using document section structure"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Order fragments by sections"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments = state.get("fragments", [])
            section_analysis = state.get("section_analysis", {})
            
            if not fragments:
                self._add_warning(state, "No fragments available for section ordering")
                state["final_order"] = []
                state["confidence_score"] = 0.0
                return state
            
            # Execute section-based sorting
            sorting_result = self.section_sorter.execute(fragments, section_analysis)
            
            if not sorting_result.success:
                self._handle_error(
                    state,
                    Exception(sorting_result.error_message),
                    "Section ordering failed"
                )
                # Fallback
                state["final_order"] = [f["id"] for f in fragments]
                state["confidence_score"] = 0.0
                return state
            
            # Update state
            ordering_data = sorting_result.data
            state["final_order"] = ordering_data["ordered_fragment_ids"]
            state["confidence_score"] = ordering_data["confidence_score"]
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["section_ordering"] = {
                "fragments_ordered": len(ordering_data["ordered_fragment_ids"]),
                "sections_analyzed": len(section_analysis.get("fragment_sections", {})),
                "numbering_system": section_analysis.get("numbering_system", "unknown"),
                "confidence_score": ordering_data["confidence_score"],
                "ordering_method": OrderingMethod.SECTIONS,
                "processing_time_ms": sorting_result.processing_time_ms
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Section ordering complete: {len(state['final_order'])} fragments, "
                f"confidence: {state['confidence_score']:.3f}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Section ordering failed")
            state["final_order"] = [f["id"] for f in state.get("fragments", [])]
            state["confidence_score"] = 0.0
        
        finally:
            self._log_step_end(state)
        
        return state

class ContinuityOrderingNode(BaseNode):
    """Node for ordering fragments based on content continuity"""
    
    def __init__(self):
        super().__init__()
        self.continuity_sorter = ContinuitySorter()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "continuity_ordering"
    
    @property
    def description(self) -> str:
        return "Order fragments using content continuity analysis"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Order fragments by continuity"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments = state.get("fragments", [])
            continuity_scores = state.get("continuity_scores", {})
            
            if not fragments:
                self._add_warning(state, "No fragments available for continuity ordering")
                state["final_order"] = []
                state["confidence_score"] = 0.0
                return state
            
            # Execute continuity-based sorting
            sorting_result = self.continuity_sorter.execute(fragments, continuity_scores)
            
            if not sorting_result.success:
                self._handle_error(
                    state,
                    Exception(sorting_result.error_message),
                    "Continuity ordering failed"
                )
                # Fallback
                state["final_order"] = [f["id"] for f in fragments]
                state["confidence_score"] = 0.3  # Low confidence fallback
                return state
            
            # Update state
            ordering_data = sorting_result.data
            state["final_order"] = ordering_data["ordered_fragment_ids"]
            state["confidence_score"] = ordering_data["confidence_score"]
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["continuity_ordering"] = {
                "fragments_ordered": len(ordering_data["ordered_fragment_ids"]),
                "continuity_pairs_analyzed": len(continuity_scores),
                "average_continuity": sum(continuity_scores.values()) / len(continuity_scores) if continuity_scores else 0.0,
                "confidence_score": ordering_data["confidence_score"],
                "ordering_method": OrderingMethod.CONTINUITY,
                "processing_time_ms": sorting_result.processing_time_ms,
                "chain_quality": ordering_data.get("chain_quality", {})
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Continuity ordering complete: {len(state['final_order'])} fragments, "
                f"confidence: {state['confidence_score']:.3f}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Continuity ordering failed")
            state["final_order"] = [f["id"] for f in state.get("fragments", [])]
            state["confidence_score"] = 0.3
        
        finally:
            self._log_step_end(state)
        
        return state

class HybridOrderingNode(BaseNode):
    """Node for hybrid ordering using multiple strategies"""
    
    def __init__(self):
        super().__init__()
        self.hybrid_sorter = HybridSorter()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "hybrid_ordering"
    
    @property
    def description(self) -> str:
        return "Order fragments using hybrid approach combining multiple strategies"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Order fragments using hybrid approach"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments = state.get("fragments", [])
            
            if not fragments:
                self._add_warning(state, "No fragments available for hybrid ordering")
                state["final_order"] = []
                state["confidence_score"] = 0.0
                return state
            
            # Prepare hybrid input data
            hybrid_input = {
                "page_hints": state.get("page_hints", {}),
                "section_analysis": state.get("section_analysis", {}),
                "continuity_scores": state.get("continuity_scores", {}),
                "quality_metrics": self._calculate_input_quality_metrics(state)
            }
            
            # Execute hybrid sorting
            sorting_result = self.hybrid_sorter.execute(fragments, hybrid_input)
            
            if not sorting_result.success:
                self._handle_error(
                    state,
                    Exception(sorting_result.error_message),
                    "Hybrid ordering failed"
                )
                # Fallback
                state["final_order"] = [f["id"] for f in fragments]
                state["confidence_score"] = 0.5
                return state
            
            # Update state
            ordering_data = sorting_result.data
            state["final_order"] = ordering_data["ordered_fragment_ids"]
            state["confidence_score"] = ordering_data["confidence_score"]
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["hybrid_ordering"] = {
                "fragments_ordered": len(ordering_data["ordered_fragment_ids"]),
                "strategies_used": ordering_data.get("strategies_used", []),
                "strategy_weights": ordering_data.get("strategy_weights", {}),
                "confidence_score": ordering_data["confidence_score"],
                "ordering_method": OrderingMethod.HYBRID,
                "processing_time_ms": sorting_result.processing_time_ms,
                "combination_approach": ordering_data.get("combination_approach", "unknown")
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Hybrid ordering complete: {len(state['final_order'])} fragments, "
                f"confidence: {state['confidence_score']:.3f}, "
                f"strategies: {ordering_data.get('strategies_used', [])}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Hybrid ordering failed")
            state["final_order"] = [f["id"] for f in state.get("fragments", [])]
            state["confidence_score"] = 0.5
        
        finally:
            self._log_step_end(state)
        
        return state
    
    def _calculate_input_quality_metrics(self, state: ProcessingState) -> Dict[str, float]:
        """Calculate quality metrics for hybrid input data"""
        fragments = state.get("fragments", [])
        fragments_count = len(fragments)
        
        if fragments_count == 0:
            return {"overall_quality": 0.0}
        
        metrics = {}
        
        # Page hints quality
        page_hints = state.get("page_hints", {})
        metrics["page_hints_coverage"] = len(page_hints) / fragments_count
        
        # Section analysis quality
        section_analysis = state.get("section_analysis", {})
        fragments_with_sections = len(section_analysis.get("fragment_sections", {}))
        metrics["section_coverage"] = fragments_with_sections / fragments_count
        
        # Continuity analysis quality
        continuity_scores = state.get("continuity_scores", {})
        expected_pairs = fragments_count * (fragments_count - 1) if fragments_count > 1 else 0
        metrics["continuity_coverage"] = len(continuity_scores) / max(expected_pairs, 1)
        
        # Overall quality score
        metrics["overall_quality"] = sum(metrics.values()) / len(metrics)
        
        return metrics

class FallbackOrderingNode(BaseNode):
    """Fallback node when all other ordering strategies fail"""
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "fallback_ordering"
    
    @property
    def description(self) -> str:
        return "Fallback ordering strategy when others fail"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Apply fallback ordering strategy"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments = state.get("fragments", [])
            
            if not fragments:
                state["final_order"] = []
                state["confidence_score"] = 0.0
                return state
            
            # Simple fallback: order by fragment confidence or original order
            fragments_with_confidence = [
                (f, f.get("confidence_score", 0.5)) for f in fragments
            ]
            
            # Sort by confidence (descending) then by original position
            fragments_with_confidence.sort(
                key=lambda x: (-x[1], fragments.index(x[0]))
            )
            
            # Create final order
            final_order = [f[0]["id"] for f in fragments_with_confidence]
            
            state["final_order"] = final_order
            state["confidence_score"] = 0.3  # Low confidence for fallback
            state["selected_strategy"] = "fallback"
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["fallback_ordering"] = {
                "fragments_ordered": len(final_order),
                "ordering_basis": "confidence_and_position",
                "confidence_score": 0.3,
                "ordering_method": OrderingMethod.FALLBACK
            }
            state["debug_info"] = debug_info
            
            self.logger.warning(
                f"Using fallback ordering for {len(final_order)} fragments"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Fallback ordering failed")
            # Ultimate fallback - original order
            state["final_order"] = [f["id"] for f in state.get("fragments", [])]
            state["confidence_score"] = 0.1
        
        finally:
            self._log_step_end(state)
        
        return state