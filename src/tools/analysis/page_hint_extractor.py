"""
Advanced page hint extraction tool
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from ..base import AnalysisTool, ToolResult
from ...config.settings import get_tool_config
from ...models.schemas import Fragment, PageHintResult

class PageHintExtractor(AnalysisTool):
    """Advanced page number extraction with multiple strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.tool_config = get_tool_config("page_hint_extractor")
        self.patterns = self.tool_config.get("patterns", {})
        self.confidence_weights = self.tool_config.get("confidence_weights", {})
    
    @property
    def name(self) -> str:
        return "page_hint_extractor"
    
    @property
    def description(self) -> str:
        return "Extract page numbers from text using multiple patterns and strategies"
    
    @property
    def capabilities(self) -> List[str]:
        return super().capabilities + ["page_number_extraction", "pattern_matching"]
    
    def _execute(self, fragment: Fragment, position: str = "both") -> ToolResult:
        """
        Extract page hints from fragment
        position: 'start', 'end', 'both'
        """
        if not isinstance(fragment, Fragment):
            return ToolResult(
                success=False,
                error_message="Input must be a Fragment object"
            )
        
        candidates = self._extract_candidate_text(fragment, position)
        results = self._analyze_candidates(candidates)
        
        if results:
            best_result = max(results, key=lambda x: x["confidence"])
            page_hint_result = PageHintResult(
                success=True,
                page_number=best_result["page_number"],
                confidence=best_result["confidence"],
                pattern_type=best_result["pattern_type"],
                context=best_result["context"],
                all_candidates=results
            )
        else:
            page_hint_result = PageHintResult(
                success=False,
                confidence=0.0,
                all_candidates=[]
            )
        
        return ToolResult(
            success=page_hint_result.success,
            data=page_hint_result.dict(),
            confidence=page_hint_result.confidence,
            metadata={
                "candidates_found": len(results),
                "position_analyzed": position
            }
        )
    
    def _extract_candidate_text(self, fragment: Fragment, position: str) -> List[str]:
        """Extract candidate text regions for analysis"""
        candidates = []
        
        # Get text content
        content = fragment.content or ""
        start_part = fragment.start_part or ""
        end_part = fragment.end_part or ""
        
        lines = content.split('\n')
        
        if position in ['end', 'both']:
            # Last few lines
            candidates.extend(lines[-3:])
            if end_part:
                candidates.append(end_part)
        
        if position in ['start', 'both']:
            # First few lines
            candidates.extend(lines[:3])
            if start_part:
                candidates.append(start_part)
        
        # Remove empty candidates
        return [c.strip() for c in candidates if c.strip()]
    
    def _analyze_candidates(self, candidates: List[str]) -> List[Dict[str, Any]]:
        """Analyze candidate text for page numbers"""
        results = []
        
        pattern_categories = {
            "vietnamese": self.patterns.get("vietnamese", []),
            "english": self.patterns.get("english", []),
            "standalone": self.patterns.get("standalone", [])
        }
        
        for candidate in candidates:
            for category, pattern_list in pattern_categories.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, candidate.lower().strip(), re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            page_num = int(match)
                            if 1 <= page_num <= 9999:  # Reasonable page range
                                confidence = self._calculate_pattern_confidence(
                                    candidate, pattern, category
                                )
                                
                                results.append({
                                    "page_number": page_num,
                                    "confidence": confidence,
                                    "pattern_type": category,
                                    "context": candidate,
                                    "pattern": pattern
                                })
                        except (ValueError, TypeError):
                            continue
        
        # Remove duplicates and sort by confidence
        unique_results = []
        seen_pages = set()
        
        for result in sorted(results, key=lambda x: x["confidence"], reverse=True):
            page_num = result["page_number"]
            if page_num not in seen_pages:
                unique_results.append(result)
                seen_pages.add(page_num)
        
        return unique_results
    
    def _calculate_pattern_confidence(self, context: str, pattern: str, category: str) -> float:
        """Calculate confidence score for page hint match"""
        base_confidence = self.confidence_weights.get(category, 0.5)
        
        # Contextual adjustments
        adjustments = 0.0
        
        # Boost for explicit page indicators
        page_indicators = ["trang", "page", "tr.", "p.", "số trang"]
        if any(indicator in context.lower() for indicator in page_indicators):
            adjustments += 0.1
        
        # Boost for page-like formatting
        if re.search(r'-\s*\d+\s*-', context):  # "- 5 -" format
            adjustments += 0.15
        
        # Penalty for noisy context
        if len(context.split()) > 10:
            adjustments -= 0.1
        
        # Penalty for context with many numbers
        number_count = len(re.findall(r'\d+', context))
        if number_count > 3:
            adjustments -= 0.05 * (number_count - 3)
        
        # Boost for end-of-line positioning
        if context.strip().endswith(tuple(re.findall(r'\d+', context))):
            adjustments += 0.05
        
        final_confidence = base_confidence + adjustments
        return min(max(final_confidence, 0.0), 1.0)
    
    def extract_from_multiple_fragments(self, fragments: List[Fragment]) -> Dict[str, PageHintResult]:
        """Extract page hints from multiple fragments"""
        results = {}
        
        for fragment in fragments:
            result = self.execute(fragment)
            if result.success:
                results[fragment.id] = PageHintResult(**result.data)
            else:
                results[fragment.id] = PageHintResult(
                    success=False,
                    confidence=0.0
                )
        
        return results
    
    def validate_page_sequence(self, page_hints: Dict[str, int]) -> Dict[str, Any]:
        """Validate extracted page sequence"""
        if not page_hints:
            return {
                "valid": False,
                "issues": ["No page hints found"],
                "coverage": 0.0
            }
        
        page_numbers = list(page_hints.values())
        issues = []
        
        # Check for duplicates
        if len(set(page_numbers)) != len(page_numbers):
            duplicates = [p for p in set(page_numbers) if page_numbers.count(p) > 1]
            issues.append(f"Duplicate pages: {duplicates}")
        
        # Check for sequence gaps
        sorted_pages = sorted(set(page_numbers))
        gaps = []
        for i in range(len(sorted_pages) - 1):
            if sorted_pages[i+1] - sorted_pages[i] > 1:
                gap_start = sorted_pages[i] + 1
                gap_end = sorted_pages[i+1] - 1
                gaps.extend(range(gap_start, gap_end + 1))
        
        if gaps:
            issues.append(f"Missing pages: {gaps}")
        
        # Check for reasonable page range
        if sorted_pages:
            if sorted_pages[0] > 50:  # Unusual to start after page 50
                issues.append(f"High starting page number: {sorted_pages[0]}")
            
            if max(sorted_pages) > 1000:  # Very large document
                issues.append(f"Unusually high page number: {max(sorted_pages)}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "coverage": len(page_hints),
            "page_range": (min(sorted_pages), max(sorted_pages)) if sorted_pages else None,
            "sequence_gaps": gaps
        }

# Convenience function for direct usage
def extract_page_hint(fragment: Fragment, position: str = "both") -> PageHintResult:
    """Direct function to extract page hint from fragment"""
    extractor = PageHintExtractor()
    result = extractor.execute(fragment, position)
    
    if result.success:
        return PageHintResult(**result.data)
    else:
        return PageHintResult(
            success=False,
            confidence=0.0,
            error_message=result.error_message
        )