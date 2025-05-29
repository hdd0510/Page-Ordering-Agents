"""
Analysis nodes for LangGraph workflow
"""
import logging
from typing import Dict, Any
from .base_node import BaseNode
from ..models.schemas import ProcessingState, Fragment
from ..tools.analysis.page_hint_extractor import PageHintExtractor
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class PageHintExtractionNode(BaseNode):
    """Node for extracting page hints from fragments"""
    
    def __init__(self):
        super().__init__()
        self.extractor = PageHintExtractor()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "page_hint_extraction"
    
    @property
    def description(self) -> str:
        return "Extract page numbers from document fragments"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Extract page hints from all fragments"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments_data = state.get("fragments", [])
            if not fragments_data:
                self._add_warning(state, "No fragments available for page hint extraction")
                return state
            
            page_hints = {}
            successful_extractions = 0
            
            for fragment_data in fragments_data:
                # Convert dict to Fragment object
                fragment = Fragment(
                    id=fragment_data["id"],
                    content=fragment_data["content"],
                    start_part=fragment_data.get("start_part"),
                    end_part=fragment_data.get("end_part")
                )
                
                # Extract page hint
                result = self.extractor.execute(fragment)
                
                if result.success and result.data.get("page_number"):
                    page_hints[fragment.id] = result.data["page_number"]
                    successful_extractions += 1
                    
                    # Store additional metadata
                    fragment_data["page_hint_confidence"] = result.confidence
                    fragment_data["page_hint_context"] = result.data.get("context", "")
            
            # Update state
            state["page_hints"] = page_hints
            
            # Validate page sequence
            validation_result = self.extractor.validate_page_sequence(page_hints)
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["page_hint_extraction"] = {
                "fragments_processed": len(fragments_data),
                "successful_extractions": successful_extractions,
                "extraction_rate": successful_extractions / len(fragments_data) if fragments_data else 0.0,
                "validation": validation_result,
                "unique_pages_found": len(set(page_hints.values())) if page_hints else 0
            }
            state["debug_info"] = debug_info
            
            # Log results
            self.logger.info(
                f"Page hint extraction: {successful_extractions}/{len(fragments_data)} successful, "
                f"validation: {'passed' if validation_result['valid'] else 'failed'}"
            )
            
            if not validation_result["valid"]:
                for issue in validation_result["issues"]:
                    self._add_warning(state, f"Page sequence issue: {issue}")
            
        except Exception as e:
            self._handle_error(state, e, "Page hint extraction failed")
        
        finally:
            self._log_step_end(state)
        
        return state

class SectionAnalysisNode(BaseNode):
    """Node for analyzing document sections and structure"""
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "section_analysis"
    
    @property
    def description(self) -> str:
        return "Analyze document sections, chapters, and hierarchical structure"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Analyze sections in document fragments"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments_data = state.get("fragments", [])
            if not fragments_data:
                self._add_warning(state, "No fragments available for section analysis")
                return state
            
            # Section patterns (moved from config for clarity)
            section_patterns = {
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
                ],
                "numbered_lists": [
                    r"^\s*(\d+)\.\s*[^\d]",
                    r"^\s*\((\d+)\)\s*[^\d]"
                ]
            }
            
            section_analysis = {
                "fragment_sections": {},
                "global_sections": {category: [] for category in section_patterns.keys()},
                "hierarchy_detected": False,
                "numbering_system": "none",
                "continuity_score": 0.0
            }
            
            # Analyze each fragment
            for fragment_data in fragments_data:
                fragment_id = fragment_data["id"]
                content = fragment_data["content"]
                
                fragment_sections = self._analyze_fragment_sections(content, section_patterns)
                
                if any(fragment_sections.values()):
                    section_analysis["fragment_sections"][fragment_id] = fragment_sections
                    
                    # Add to global sections
                    for category, sections in fragment_sections.items():
                        section_analysis["global_sections"][category].extend(sections)
            
            # Determine overall structure
            section_analysis["hierarchy_detected"] = any(
                len(sections) > 0 for sections in section_analysis["global_sections"].values()
            )
            
            section_analysis["numbering_system"] = self._determine_numbering_system(
                section_analysis["global_sections"]
            )
            
            section_analysis["continuity_score"] = self._calculate_section_continuity(
                section_analysis["global_sections"]
            )
            
            # Update state
            state["section_analysis"] = section_analysis
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["section_analysis"] = {
                "fragments_with_sections": len(section_analysis["fragment_sections"]),
                "total_sections_found": sum(
                    len(sections) for sections in section_analysis["global_sections"].values()
                ),
                "hierarchy_detected": section_analysis["hierarchy_detected"],
                "numbering_system": section_analysis["numbering_system"]
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Section analysis: {len(section_analysis['fragment_sections'])} fragments with sections, "
                f"system: {section_analysis['numbering_system']}, "
                f"hierarchy: {section_analysis['hierarchy_detected']}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Section analysis failed")
        
        finally:
            self._log_step_end(state)
        
        return state
    
    def _analyze_fragment_sections(self, content: str, patterns: Dict[str, list]) -> Dict[str, list]:
        """Analyze sections in a single fragment"""
        import re
        
        fragment_sections = {}
        lines = content.split('\n')
        
        for category, pattern_list in patterns.items():
            found_sections = []
            
            for pattern in pattern_list:
                for line_num, line in enumerate(lines):
                    matches = re.findall(pattern, line, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        section_info = {
                            "number": self._extract_section_number(match),
                            "line_number": line_num,
                            "context": line.strip(),
                            "pattern": pattern
                        }
                        
                        if section_info["number"] is not None:
                            found_sections.append(section_info)
            
            fragment_sections[category] = found_sections
        
        return fragment_sections
    
    def _extract_section_number(self, match) -> Any:
        """Extract numeric value from section match"""
        if isinstance(match, tuple):
            # Handle multi-group matches
            for group in match:
                if group and (group.isdigit() or self._is_roman_numeral(group)):
                    return int(group) if group.isdigit() else self._roman_to_int(group)
        elif isinstance(match, str):
            if match.isdigit():
                return int(match)
            elif self._is_roman_numeral(match):
                return self._roman_to_int(match)
        
        return None
    
    def _is_roman_numeral(self, text: str) -> bool:
        """Check if text is a Roman numeral"""
        return bool(re.match(r'^[IVX]+$', text.upper()))
    
    def _roman_to_int(self, roman: str) -> Optional[int]:
        """Convert Roman numeral to integer"""
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        roman = roman.upper()
        
        if not all(c in roman_map for c in roman):
            return None
        
        total = 0
        prev_value = 0
        
        for char in reversed(roman):
            value = roman_map[char]
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        return total
    
    def _determine_numbering_system(self, global_sections: Dict[str, list]) -> str:
        """Determine the primary numbering system"""
        if global_sections["chapters"]:
            return "chapter_based"
        elif global_sections["sections"]:
            return "section_based"
        elif global_sections["subsections"]:
            return "hierarchical"
        elif global_sections["numbered_lists"]:
            return "list_based"
        else:
            return "none"
    
    def _calculate_section_continuity(self, global_sections: Dict[str, list]) -> float:
        """Calculate how continuous the section numbering is"""
        primary_sections = []
        
        # Find primary section type
        for category in ["chapters", "sections", "subsections", "numbered_lists"]:
            if global_sections[category]:
                primary_sections = [s["number"] for s in global_sections[category] if s["number"]]
                break
        
        if len(primary_sections) < 2:
            return 0.5  # Insufficient data
        
        # Calculate continuity score
        sorted_sections = sorted(set(primary_sections))
        expected_gaps = len(sorted_sections) - 1
        actual_gaps = sorted_sections[-1] - sorted_sections[0]
        
        if actual_gaps == 0:
            return 1.0
        
        continuity = expected_gaps / actual_gaps
        return min(continuity, 1.0)

class ContinuityAnalysisNode(BaseNode):
    """Node for analyzing content continuity between fragments"""
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    @property
    def name(self) -> str:
        return "continuity_analysis"
    
    @property
    def description(self) -> str:
        return "Analyze text continuity between document fragments"
    
    def execute(self, state: ProcessingState) -> ProcessingState:
        """Analyze continuity between all fragment pairs"""
        state["current_step"] = self.name
        self._log_step_start(state)
        
        try:
            fragments_data = state.get("fragments", [])
            if len(fragments_data) < 2:
                self._add_warning(state, "Need at least 2 fragments for continuity analysis")
                state["continuity_scores"] = {}
                return state
            
            continuity_scores = {}
            total_pairs = 0
            analyzed_pairs = 0
            
            # Calculate pairwise continuity scores
            for i, frag_a in enumerate(fragments_data):
                for j, frag_b in enumerate(fragments_data):
                    if i != j:
                        total_pairs += 1
                        score_key = f"{frag_a['id']}_{frag_b['id']}"
                        
                        try:
                            score = self._calculate_continuity_score(
                                frag_a.get("end_part", ""),
                                frag_b.get("start_part", "")
                            )
                            continuity_scores[score_key] = score
                            analyzed_pairs += 1
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to calculate continuity for {score_key}: {e}")
                            continuity_scores[score_key] = 0.0
            
            # Update state
            state["continuity_scores"] = continuity_scores
            
            # Calculate statistics
            if continuity_scores:
                scores_list = list(continuity_scores.values())
                avg_score = sum(scores_list) / len(scores_list)
                max_score = max(scores_list)
                min_score = min(scores_list)
                high_continuity_pairs = sum(1 for s in scores_list if s > 0.7)
            else:
                avg_score = max_score = min_score = 0.0
                high_continuity_pairs = 0
            
            # Update debug info
            debug_info = state.get("debug_info", {})
            debug_info["continuity_analysis"] = {
                "total_pairs": total_pairs,
                "analyzed_pairs": analyzed_pairs,
                "average_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "high_continuity_pairs": high_continuity_pairs,
                "analysis_success_rate": analyzed_pairs / total_pairs if total_pairs > 0 else 0.0
            }
            state["debug_info"] = debug_info
            
            self.logger.info(
                f"Continuity analysis: {analyzed_pairs}/{total_pairs} pairs analyzed, "
                f"avg score: {avg_score:.3f}, high continuity pairs: {high_continuity_pairs}"
            )
            
        except Exception as e:
            self._handle_error(state, e, "Continuity analysis failed")
        
        finally:
            self._log_step_end(state)
        
        return state
    
    def _calculate_continuity_score(self, end_text: str, start_text: str) -> float:
        """Calculate continuity score between two text fragments"""
        if not end_text or not start_text:
            return 0.0
        
        # Multi-factor continuity analysis
        syntactic_score = self._analyze_syntactic_continuity(end_text, start_text)
        semantic_score = self._analyze_semantic_continuity(end_text, start_text)
        lexical_score = self._analyze_lexical_continuity(end_text, start_text)
        
        # Weighted combination
        weights = {"syntactic": 0.3, "semantic": 0.4, "lexical": 0.3}
        overall_score = (
            syntactic_score * weights["syntactic"] +
            semantic_score * weights["semantic"] +
            lexical_score * weights["lexical"]
        )
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _analyze_syntactic_continuity(self, end_text: str, start_text: str) -> float:
        """Analyze grammatical/syntactic flow"""
        import re
        
        end_clean = end_text.strip()
        start_clean = start_text.strip()
        
        # Check for sentence continuation
        if end_clean.endswith((',', ';', 'và', 'hoặc', 'nhưng', 'that', 'which', 'who')):
            if start_clean and start_clean[0].islower():
                return 0.8
        
        # Check for incomplete sentences
        if not re.search(r'[.!?:;]$', end_clean):
            return 0.6
        
        # Check for transitional elements
        transition_starts = ['however', 'therefore', 'moreover', 'tuy nhiên', 'do đó', 'ngoài ra']
        if any(start_clean.lower().startswith(trans.lower()) for trans in transition_starts):
            return 0.7
        
        return 0.3
    
    def _analyze_semantic_continuity(self, end_text: str, start_text: str) -> float:
        """Analyze semantic/topical continuity"""
        # Simplified semantic analysis using word overlap
        
        end_words = set(end_text.lower().split())
        start_words = set(start_text.lower().split())
        
        # Remove stop words
        stop_words = {
            'và', 'hoặc', 'nhưng', 'tuy nhiên', 'do đó', 'vì vậy',
            'the', 'and', 'or', 'but', 'however', 'therefore', 'this', 'that'
        }
        
        end_words -= stop_words
        start_words -= stop_words
        
        if not end_words or not start_words:
            return 0.3
        
        # Calculate Jaccard similarity
        intersection = len(end_words & start_words)
        union = len(end_words | start_words)
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # Boost for exact word repetitions at boundaries
        end_last_words = end_text.split()[-3:]
        start_first_words = start_text.split()[:3]
        
        boundary_overlap = len(set(end_last_words) & set(start_first_words))
        boundary_bonus = boundary_overlap * 0.1
        
        return min(jaccard_score + boundary_bonus, 1.0)
    
    def _analyze_lexical_continuity(self, end_text: str, start_text: str) -> float:
        """Analyze lexical cohesion and transitional phrases"""
        
        # Check for explicit transitions
        strong_transitions = [
            'however', 'therefore', 'moreover', 'furthermore', 'consequently',
            'tuy nhiên', 'do đó', 'vì vậy', 'mặt khác', 'bên cạnh đó', 'hơn nữa'
        ]
        
        start_lower = start_text.lower().strip()
        for transition in strong_transitions:
            if start_lower.startswith(transition.lower()):
                return 0.9
        
        # Check for pronoun references
        pronouns = ['this', 'that', 'these', 'those', 'it', 'they', 'such', 'điều này', 'điều đó', 'như vậy']
        if any(start_text.lower().strip().startswith(pron.lower()) for pron in pronouns):
            return 0.7
        
        # Check for continued numbering/listing
        import re
        end_numbers = re.findall(r'(\d+)\.', end_text)
        start_numbers = re.findall(r'(\d+)\.', start_text)
        
        if end_numbers and start_numbers:
            try:
                if int(start_numbers[0]) == int(end_numbers[-1]) + 1:
                    return 0.8
            except (ValueError, IndexError):
                pass
        
        return 0.3