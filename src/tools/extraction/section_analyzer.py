"""
Section analyzer for document structure analysis
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

from ..base import AnalysisTool, ToolResult
from ...models.schemas import Fragment
from ...config.settings import get_tool_config

logger = logging.getLogger(__name__)

class SectionAnalyzer(AnalysisTool):
    """Analyze document sections, chapters, and hierarchical structure"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.tool_config = get_tool_config("section_analyzer")
        self.patterns = self._initialize_patterns()
        self.min_section_confidence = self.get_config_value("min_section_confidence", 0.6)
        
    @property
    def name(self) -> str:
        return "section_analyzer"
    
    @property
    def description(self) -> str:
        return "Analyze document sections, chapters, and hierarchical structure"
    
    @property
    def capabilities(self) -> List[str]:
        return super().capabilities + ["section_detection", "hierarchy_analysis", "numbering_systems"]
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize section detection patterns"""
        
        # Get patterns from config or use defaults
        config_patterns = self.tool_config.get("patterns", {})
        
        default_patterns = {
            "chapters": [
                r"(?:chương|chapter|phần|part)\s*([IVX]+|\d+)",
                r"(?:ch\.|chap\.)\s*(\d+)",
                r"^(chương|chapter)\s+([IVX]+|\d+)[\s\.:]\s*([^\n]+)",
                r"^\s*([IVX]+|\d+)\.\s*([A-ZÀÁẠẢÃÂẦẤẬẨẪ][^\n]{10,})"
            ],
            
            "sections": [
                r"^(\d+)\.\s*([A-ZÀÁẠẢÃÂẦẤẬẨẪ][^\n]+)",
                r"(?:mục|section|tiết)\s*(\d+)",
                r"^([A-Z])\.\s*([A-ZÀÁẠẢÃÂẦẤẬẨẪ][^\n]+)",
                r"^\s*(\d+)\s*[\.\-\)]\s*([A-ZÀÁẠẢÃÂẦẤẬẨẪ][^\n]{5,})"
            ],
            
            "subsections": [
                r"^(\d+)\.(\d+)\s*([A-Za-zÀ-ỹ][^\n]+)",
                r"^\s*([a-z])\)\s*([A-ZÀÁẠẢÃÂẦẤẬẨẪ][^\n]+)",
                r"^(\d+)\.(\d+)\.(\d+)\s*([A-Za-zÀ-ỹ][^\n]+)",
                r"^\s*[\-\*\+]\s*([A-ZÀÁẠẢÃÂẦẤẬẨẪ][^\n]{10,})"
            ],
            
            "numbered_lists": [
                r"^\s*(\d+)\.\s*([^\d][^\n]+)",
                r"^\s*\((\d+)\)\s*([^\n]+)",
                r"^\s*(\d+)\)\s*([^\n]+)",
                r"^\s*[\-\*\+]\s*([^\n]{5,})"
            ],
            
            "appendices": [
                r"(?:phụ lục|appendix|annex)\s*([A-Z]|\d+)",
                r"^(phụ lục|appendix)\s+([A-Z]|\d+)[\s\.:]\s*([^\n]+)"
            ],
            
            "references": [
                r"(?:tài liệu tham khảo|references|bibliography)",
                r"(?:nguồn tham khảo|cited works|works cited)"
            ]
        }
        
        # Merge config patterns with defaults
        for category, patterns in default_patterns.items():
            if category in config_patterns:
                patterns.extend(config_patterns[category])
        
        return config_patterns if config_patterns else default_patterns
    
    def _execute(self, fragments: List[Fragment]) -> ToolResult:
        """
        Analyze sections in document fragments
        
        Args:
            fragments: List of document fragments to analyze
            
        Returns:
            ToolResult with section analysis
        """
        try:
            if not fragments:
                return ToolResult(
                    success=False,
                    error_message="No fragments provided for section analysis"
                )
            
            # Analyze each fragment for sections
            fragment_sections = {}
            global_sections = defaultdict(list)
            
            for fragment in fragments:
                section_data = self._analyze_fragment_sections(fragment)
                
                if any(section_data.values()):
                    fragment_sections[fragment.id] = section_data
                    
                    # Add to global sections
                    for category, sections in section_data.items():
                        global_sections[category].extend(sections)
            
            # Analyze overall document structure
            structure_analysis = self._analyze_document_structure(global_sections, fragments)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_section_confidence(global_sections, fragments)
            
            result_data = {
                "fragment_sections": fragment_sections,
                "global_sections": dict(global_sections),
                "structure_analysis": structure_analysis,
                "confidence_scores": confidence_scores,
                "hierarchy_detected": structure_analysis["has_hierarchy"],
                "numbering_system": structure_analysis["primary_numbering_system"],
                "continuity_score": structure_analysis["section_continuity_score"]
            }
            
            overall_confidence = confidence_scores.get("overall", 0.0)
            
            return ToolResult(
                success=len(fragment_sections) > 0,
                data=result_data,
                confidence=overall_confidence,
                metadata={
                    "fragments_with_sections": len(fragment_sections),
                    "total_sections_found": sum(len(sections) for sections in global_sections.values()),
                    "analysis_method": "pattern_matching"
                }
            )
            
        except Exception as e:
            logger.error(f"Section analysis failed: {str(e)}")
            return ToolResult(
                success=False,
                error_message=f"Section analysis failed: {str(e)}"
            )
    
    def _analyze_fragment_sections(self, fragment: Fragment) -> Dict[str, List[Dict]]:
        """Analyze sections within a single fragment"""
        
        content = fragment.content
        if not content:
            return {category: [] for category in self.patterns.keys()}
        
        lines = content.split('\n')
        fragment_sections = defaultdict(list)
        
        for category, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                matches = self._find_pattern_matches(pattern, lines, category)
                fragment_sections[category].extend(matches)
        
        # Remove duplicates and sort
        for category in fragment_sections:
            fragment_sections[category] = self._deduplicate_sections(fragment_sections[category])
        
        return dict(fragment_sections)
    
    def _find_pattern_matches(self, pattern: str, lines: List[str], category: str) -> List[Dict]:
        """Find pattern matches in text lines"""
        matches = []
        
        for line_num, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            try:
                regex_matches = re.finditer(pattern, line_clean, re.IGNORECASE | re.MULTILINE)
                
                for match in regex_matches:
                    section_info = self._extract_section_info(match, line_clean, line_num, category)
                    if section_info:
                        matches.append(section_info)
                        
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
        
        return matches
    
    def _extract_section_info(self, match, line: str, line_num: int, category: str) -> Optional[Dict]:
        """Extract section information from regex match"""
        
        try:
            groups = match.groups()
            if not groups:
                return None
            
            # Parse section number/identifier
            section_number = self._parse_section_number(groups[0])
            
            # Extract section title if available
            section_title = ""
            if len(groups) > 1:
                section_title = groups[1].strip()
            elif len(line) > match.end():
                # Try to extract title from rest of line
                remaining_text = line[match.end():].strip()
                if remaining_text and len(remaining_text) > 3:
                    section_title = remaining_text[:100]  # Limit title length
            
            # Calculate match confidence
            confidence = self._calculate_match_confidence(match, line, category)
            
            return {
                "number": section_number,
                "title": section_title,
                "line_number": line_num,
                "context": line,
                "category": category,
                "confidence": confidence,
                "pattern_used": match.re.pattern,
                "position_in_line": match.start()
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract section info: {e}")
            return None
    
    def _parse_section_number(self, number_str: str) -> Optional[Any]:
        """Parse section number from string"""
        
        number_str = number_str.strip()
        
        # Try integer first
        if number_str.isdigit():
            return int(number_str)
        
        # Try Roman numerals
        roman_value = self._roman_to_int(number_str)
        if roman_value is not None:
            return roman_value
        
        # Try letter (A, B, C, etc.)
        if len(number_str) == 1 and number_str.isalpha():
            return number_str.upper()
        
        # Return as string if can't parse
        return number_str
    
    def _roman_to_int(self, roman: str) -> Optional[int]:
        """Convert Roman numeral to integer"""
        roman_map = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        
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
        
        return total if total > 0 else None
    
    def _calculate_match_confidence(self, match, line: str, category: str) -> float:
        """Calculate confidence score for pattern match"""
        
        base_confidence = 0.5
        
        # Position-based confidence
        if match.start() == 0:  # Starts at beginning of line
            base_confidence += 0.2
        elif match.start() <= 5:  # Near beginning
            base_confidence += 0.1
        
        # Category-specific adjustments
        category_multipliers = {
            "chapters": 1.2,
            "sections": 1.1,
            "subsections": 1.0,
            "numbered_lists": 0.9,
            "appendices": 1.1,
            "references": 1.0
        }
        
        base_confidence *= category_multipliers.get(category, 1.0)
        
        # Context-based adjustments
        line_lower = line.lower()
        
        # Boost for section keywords
        section_keywords = ["chương", "chapter", "phần", "part", "mục", "section", "tiết"]
        if any(keyword in line_lower for keyword in section_keywords):
            base_confidence += 0.1
        
        # Penalty for very long lines (likely false positives)
        if len(line) > 200:
            base_confidence -= 0.1
        
        # Boost for lines that look like headers (short, title case)
        if len(line) < 100 and any(c.isupper() for c in line):
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _deduplicate_sections(self, sections: List[Dict]) -> List[Dict]:
        """Remove duplicate sections and sort"""
        
        if not sections:
            return []
        
        # Group by number to find duplicates
        by_number = defaultdict(list)
        for section in sections:
            by_number[section["number"]].append(section)
        
        # Keep highest confidence section for each number
        unique_sections = []
        for number, section_group in by_number.items():
            best_section = max(section_group, key=lambda s: s["confidence"])
            unique_sections.append(best_section)
        
        # Sort by number
        def sort_key(section):
            num = section["number"]
            if isinstance(num, int):
                return (0, num)
            elif isinstance(num, str) and len(num) == 1 and num.isalpha():
                return (1, ord(num))
            else:
                return (2, str(num))
        
        return sorted(unique_sections, key=sort_key)
    
    def _analyze_document_structure(self, global_sections: Dict, fragments: List[Fragment]) -> Dict[str, Any]:
        """Analyze overall document structure"""
        
        structure = {
            "has_hierarchy": False,
            "hierarchy_levels": [],
            "primary_numbering_system": "none",
            "section_distribution": {},
            "section_continuity_score": 0.0,
            "structural_patterns": []
        }
        
        # Analyze hierarchy
        non_empty_categories = [cat for cat, sections in global_sections.items() if sections]
        
        if len(non_empty_categories) >= 2:
            structure["has_hierarchy"] = True
            structure["hierarchy_levels"] = non_empty_categories
        
        # Determine primary numbering system
        structure["primary_numbering_system"] = self._determine_numbering_system(global_sections)
        
        # Section distribution analysis
        for category, sections in global_sections.items():
            if sections:
                structure["section_distribution"][category] = {
                    "count": len(sections),
                    "avg_confidence": sum(s["confidence"] for s in sections) / len(sections),
                    "number_types": self._analyze_number_types(sections)
                }
        
        # Calculate continuity score
        structure["section_continuity_score"] = self._calculate_section_continuity(global_sections)
        
        # Identify structural patterns
        structure["structural_patterns"] = self._identify_structural_patterns(global_sections)
        
        return structure
    
    def _determine_numbering_system(self, global_sections: Dict) -> str:
        """Determine the primary numbering system used"""
        
        # Priority order for numbering systems
        priority_categories = ["chapters", "sections", "subsections", "numbered_lists"]
        
        for category in priority_categories:
            if category in global_sections and global_sections[category]:
                return f"{category}_based"
        
        # Check for other categories
        for category, sections in global_sections.items():
            if sections:
                return f"{category}_based"
        
        return "none"
    
    def _analyze_number_types(self, sections: List[Dict]) -> Dict[str, int]:
        """Analyze types of numbers used in sections"""
        
        type_counts = {"integer": 0, "roman": 0, "letter": 0, "other": 0}
        
        for section in sections:
            number = section["number"]
            
            if isinstance(number, int):
                type_counts["integer"] += 1
            elif isinstance(number, str):
                if len(number) == 1 and number.isalpha():
                    type_counts["letter"] += 1
                elif self._roman_to_int(number) is not None:
                    type_counts["roman"] += 1
                else:
                    type_counts["other"] += 1
        
        return type_counts
    
    def _calculate_section_continuity(self, global_sections: Dict) -> float:
        """Calculate how continuous the section numbering is"""
        
        # Find the most prominent section type
        primary_sections = None
        max_count = 0
        
        for category, sections in global_sections.items():
            if len(sections) > max_count:
                max_count = len(sections)
                primary_sections = sections
        
        if not primary_sections or len(primary_sections) < 2:
            return 0.5  # Neutral score
        
        # Extract numeric values
        numbers = []
        for section in primary_sections:
            number = section["number"]
            if isinstance(number, int):
                numbers.append(number)
            elif isinstance(number, str) and len(number) == 1 and number.isalpha():
                numbers.append(ord(number.upper()) - ord('A') + 1)
        
        if len(numbers) < 2:
            return 0.5
        
        # Calculate continuity
        numbers.sort()
        gaps = []
        for i in range(len(numbers) - 1):
            gap = numbers[i + 1] - numbers[i]
            gaps.append(gap)
        
        # Ideal gap is 1 (consecutive numbering)
        avg_gap = sum(gaps) / len(gaps)
        
        if avg_gap <= 1.5:
            return 0.9  # Very continuous
        elif avg_gap <= 3.0:
            return 0.7  # Moderately continuous
        else:
            return 0.3  # Poor continuity
    
    def _identify_structural_patterns(self, global_sections: Dict) -> List[str]:
        """Identify common structural patterns"""
        
        patterns = []
        
        # Check for standard academic structure
        if ("chapters" in global_sections and global_sections["chapters"] and
            "sections" in global_sections and global_sections["sections"]):
            patterns.append("academic_structure")
        
        # Check for hierarchical numbering (1.1, 1.2, etc.)
        if "subsections" in global_sections:
            subsections = global_sections["subsections"]
            if any("." in str(s["number"]) for s in subsections):
                patterns.append("hierarchical_numbering")
        
        # Check for appendices
        if "appendices" in global_sections and global_sections["appendices"]:
            patterns.append("has_appendices")
        
        # Check for references section
        if "references" in global_sections and global_sections["references"]:
            patterns.append("has_references")
        
        # Check for mixed numbering systems
        number_systems = set()
        for sections in global_sections.values():
            for section in sections:
                number = section["number"]
                if isinstance(number, int):
                    number_systems.add("numeric")
                elif isinstance(number, str) and self._roman_to_int(number):
                    number_systems.add("roman")
                elif isinstance(number, str) and len(number) == 1 and number.isalpha():
                    number_systems.add("alphabetic")
        
        if len(number_systems) > 1:
            patterns.append("mixed_numbering")
        
        return patterns
    
    def _calculate_section_confidence(self, global_sections: Dict, fragments: List[Fragment]) -> Dict[str, float]:
        """Calculate confidence scores for section analysis"""
        
        confidence_scores = {}
        
        # Overall confidence based on section coverage
        fragments_with_sections = sum(1 for sections in global_sections.values() for _ in sections)
        coverage = fragments_with_sections / max(len(fragments), 1)
        confidence_scores["coverage"] = min(coverage, 1.0)
        
        # Category-specific confidences
        for category, sections in global_sections.items():
            if sections:
                avg_confidence = sum(s["confidence"] for s in sections) / len(sections)
                confidence_scores[category] = avg_confidence
        
        # Overall confidence (weighted average)
        if confidence_scores:
            weights = {"coverage": 0.4}
            category_weight = 0.6 / len([k for k in confidence_scores.keys() if k != "coverage"])
            
            total_confidence = confidence_scores["coverage"] * weights["coverage"]
            for category, conf in confidence_scores.items():
                if category != "coverage":
                    total_confidence += conf * category_weight
            
            confidence_scores["overall"] = total_confidence
        else:
            confidence_scores["overall"] = 0.0
        
        return confidence_scores