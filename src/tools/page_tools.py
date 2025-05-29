import re
from typing import List, Optional
from ..models.schemas import Fragment
import logging

logger = logging.getLogger(__name__)

# Enhanced regex for Vietnamese page patterns
PAGE_PATTERNS = [
    r"(?:trang|trag|page|p\.?)\s*([1-9]\d{0,3})",  # "trang 5", "page 10"
    r"(?:\b|_)([1-9]\d{0,3})(?:\b|\s*$)",          # standalone numbers
    r"([1-9]\d{0,3})\s*(?:\/\d+)?$",               # "5/10" format
    r"-\s*([1-9]\d{0,3})\s*-",                     # "- 5 -" format
]

def extract_page_hint_advanced(fragment: Fragment) -> Fragment:
    """Advanced page number extraction with multiple patterns."""
    text_candidates = [
        fragment.end_part.split()[-3:] if fragment.end_part else [],  # Last 3 words
        fragment.start_part.split()[:3] if fragment.start_part else [], # First 3 words
        fragment.content.split()[-5:]    # Last 5 words of full content
    ]
    
    for candidate_list in text_candidates:
        if not candidate_list:
            continue
            
        candidate_text = ' '.join(candidate_list).lower()
        
        for pattern in PAGE_PATTERNS:
            matches = re.findall(pattern, candidate_text, re.IGNORECASE)
            if matches:
                try:
                    page_num = int(matches[0])
                    if 1 <= page_num <= 9999:  # Reasonable page range
                        fragment.page_hint = page_num
                        logger.debug(f"Found page hint {page_num} for fragment {fragment.id}")
                        return fragment
                except ValueError:
                    continue
    
    return fragment

def validate_page_sequence(fragments: List[Fragment]) -> dict:
    """Validate page sequence and detect issues."""
    page_hints = [f.page_hint for f in fragments if f.page_hint is not None]
    
    result = {
        "has_all_pages": len(page_hints) == len(fragments),
        "has_duplicates": len(set(page_hints)) != len(page_hints),
        "missing_pages": [],
        "duplicate_pages": [],
        "sequence_gaps": []
    }
    
    if page_hints:
        sorted_pages = sorted(page_hints)
        
        # Check for duplicates
        seen = set()
        for page in page_hints:
            if page in seen:
                result["duplicate_pages"].append(page)
            seen.add(page)
        
        # Check for gaps in sequence
        for i in range(len(sorted_pages) - 1):
            if sorted_pages[i+1] - sorted_pages[i] > 1:
                missing_range = list(range(sorted_pages[i] + 1, sorted_pages[i+1]))
                result["missing_pages"].extend(missing_range)
    
    logger.info(f"Page validation: {result}")
    return result

def simple_page_sort(fragments: List[Fragment]) -> List[str]:
    """Sort fragments by page hint if valid sequence exists."""
    validation = validate_page_sequence(fragments)
    
    if not validation["has_all_pages"]:
        raise ValueError("Some fragments missing page numbers")
    
    if validation["has_duplicates"]:
        raise ValueError(f"Duplicate pages found: {validation['duplicate_pages']}")
    
    # Sort by page_hint
    sorted_fragments = sorted(fragments, key=lambda f: f.page_hint or 0)
    result = [f.id for f in sorted_fragments]
    logger.info(f"Sorted fragments by page hints: {result}")
    return result