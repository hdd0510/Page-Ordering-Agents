from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, TypedDict, Annotated
from pydantic import BaseModel, Field
from ..models.schemas import Fragment, OrderingResult, ProcessingStatus
from ..tools.page_tools import extract_page_hint_advanced, simple_page_sort, validate_page_sequence
from ..tools.continuity_tools import find_best_continuation
import time
import logging

logger = logging.getLogger(__name__)

# Define state schema for StateGraph
class OrderingState(TypedDict):
    document_id: str
    fragments: List[Fragment]
    start_time: float
    result: OrderingResult
    validation: Dict[str, Any]
    error: str

class PageOrderingAgent:
    def __init__(self):
        self.graph = self._build_graph()
        logger.info("Page Ordering Agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the page ordering workflow graph."""
        
        def preprocess_fragments(state: OrderingState) -> OrderingState:
            """Extract page hints from all fragments."""
            logger.info("Preprocessing fragments for page hints...")
            
            processed_fragments = []
            for fragment in state["fragments"]:
                # Ensure fragment has proper start_part and end_part
                if not fragment.start_part or not fragment.end_part:
                    fragment.model_post_init(None)
                
                processed_fragment = extract_page_hint_advanced(fragment)
                processed_fragments.append(processed_fragment)
            
            state["fragments"] = processed_fragments
            state["validation"] = validate_page_sequence(processed_fragments)
            
            logger.info(f"Processed {len(processed_fragments)} fragments")
            return state
        
        def sort_by_page_hints(state: OrderingState) -> OrderingState:
            """Sort fragments using extracted page hints."""
            logger.info("Sorting by page hints...")
            
            try:
                ordered_ids = simple_page_sort(state["fragments"])
                state["result"] = OrderingResult(
                    document_id=state["document_id"],
                    ordered_fragment_ids=ordered_ids,
                    confidence_score=0.95,  # High confidence for page-based sorting
                    method_used="page_hint",
                    processing_time=time.time() - state["start_time"],
                    debug_info=state["validation"]
                )
                logger.info("Successfully sorted by page hints")
            except Exception as e:
                logger.error(f"Page hint sorting failed: {str(e)}")
                state["error"] = str(e)
            
            return state
        
        def sort_by_continuity(state: OrderingState) -> OrderingState:
            """Sort fragments using LLM continuity analysis."""
            logger.info("Sorting by content continuity...")
            
            fragments = state["fragments"].copy()
            if not fragments:
                state["result"] = OrderingResult(
                    document_id=state["document_id"],
                    ordered_fragment_ids=[],
                    confidence_score=0.0,
                    method_used="llm_continuity",
                    processing_time=time.time() - state["start_time"]
                )
                return state
            
            # Start with first fragment (or could use highest confidence)
            ordered_fragments = [fragments.pop(0)]
            confidence_scores = []
            
            # Greedy approach: find best continuation at each step
            while fragments:
                current = ordered_fragments[-1]
                result = find_best_continuation(current, fragments)
                
                if result["best_fragment"] and result["score"] > 0.3:  # Threshold for good continuity
                    ordered_fragments.append(result["best_fragment"])
                    fragments.remove(result["best_fragment"])
                    confidence_scores.append(result["score"])
                else:
                    # If no good continuation found, add remaining fragments
                    ordered_fragments.extend(fragments)
                    confidence_scores.extend([0.1] * len(fragments))
                    break
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            state["result"] = OrderingResult(
                document_id=state["document_id"],
                ordered_fragment_ids=[f.id for f in ordered_fragments],
                confidence_score=avg_confidence,
                method_used="llm_continuity",
                processing_time=time.time() - state["start_time"],
                debug_info={"individual_scores": confidence_scores}
            )
            
            logger.info(f"Completed continuity sorting with avg confidence: {avg_confidence:.2f}")
            return state
        
        def needs_llm_processing(state: OrderingState) -> str:
            """Determine if LLM processing is needed."""
            validation = state.get("validation", {})
            
            if not validation.get("has_all_pages", False):
                logger.info("Missing page numbers - using LLM continuity")
                return "llm_sort"
            
            if validation.get("has_duplicates", False):
                logger.info("Duplicate page numbers - using LLM continuity") 
                return "llm_sort"
            
            logger.info("Valid page sequence - using page hint sorting")
            return "page_sort"
        
        # Build the graph with schema
        workflow = StateGraph(OrderingState)
        
        # Add nodes
        workflow.add_node("preprocess", preprocess_fragments)
        workflow.add_node("page_sort", sort_by_page_hints)
        workflow.add_node("llm_sort", sort_by_continuity)
        
        # Set entry point
        workflow.set_entry_point("preprocess")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "preprocess",
            needs_llm_processing,
            {
                "page_sort": "page_sort",
                "llm_sort": "llm_sort"
            }
        )
        
        # Add terminal edges
        workflow.add_edge("page_sort", END)
        workflow.add_edge("llm_sort", END)
        
        return workflow.compile()
    
    def process_document(self, document_id: str, fragments: List[Fragment]) -> OrderingResult:
        """Process document and return ordering result."""
        logger.info(f"Processing document: {document_id} with {len(fragments)} fragments")
        
        initial_state = {
            "document_id": document_id,
            "fragments": fragments,
            "start_time": time.time(),
            "result": None,
            "validation": {},
            "error": None
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            if final_state.get("error"):
                logger.error(f"Processing error: {final_state['error']}")
                # Return fallback result
                return OrderingResult(
                    document_id=document_id,
                    ordered_fragment_ids=[f.id for f in fragments],
                    confidence_score=0.0,
                    method_used="fallback",
                    processing_time=time.time() - initial_state["start_time"],
                    debug_info={"error": final_state["error"]}
                )
            
            return final_state["result"]
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return OrderingResult(
                document_id=document_id,
                ordered_fragment_ids=[f.id for f in fragments],  # Fallback order
                confidence_score=0.0,
                method_used="fallback",
                processing_time=time.time() - initial_state["start_time"],
                debug_info={"error": str(e)}
            )