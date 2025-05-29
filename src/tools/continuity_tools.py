from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
import re
import logging
import json
import os
from ..models.schemas import Fragment

logger = logging.getLogger(__name__)

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-0417",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
except Exception as e:
    logger.warning(f"Failed to initialize LLM: {str(e)}")
    llm = None

def continuity_score_detailed(a_end: str, b_start: str, language: str = "vietnamese") -> dict:
    """
    Calculate detailed continuity score between two text fragments.
    Returns score and reasoning.
    """
    if not llm:
        logger.warning("LLM not available, returning default score")
        return {"score": 0.5, "reasoning": "LLM not available", "raw_response": ""}
    
    prompt_template = {
        "vietnamese": """
Đánh giá mức độ liền mạch khi nối hai đoạn văn bản:

Đoạn A kết thúc: ```{a_end}```
Đoạn B bắt đầu: ```{b_start}```

Hãy đánh giá theo các tiêu chí:
1. Ngữ pháp và cú pháp (0-0.3)
2. Ngữ nghĩa và logic nội dung (0-0.4) 
3. Từ khóa và chủ đề liên quan (0-0.3)

Trả về JSON format:
{{"score": 0.85, "reasoning": "Lý do chi tiết..."}}
""",
        "english": """
Evaluate text continuity between two fragments:

Fragment A ends: ```{a_end}```
Fragment B starts: ```{b_start}```

Rate based on:
1. Grammar and syntax flow (0-0.3)
2. Semantic and logical continuity (0-0.4)
3. Related keywords and topics (0-0.3)

Return JSON format:
{{"score": 0.85, "reasoning": "Detailed explanation..."}}
"""
    }
    
    prompt = prompt_template.get(language, prompt_template["english"]).format(
        a_end=a_end[-300:] if a_end else "", 
        b_start=b_start[:300] if b_start else ""
    )
    
    try:
        response = llm.predict(prompt)
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "score": float(result.get("score", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "raw_response": response
            }
    except Exception as e:
        logger.error(f"Continuity scoring failed: {str(e)}")
    
    return {"score": 0.0, "reasoning": "Scoring failed", "raw_response": ""}

def find_best_continuation(current_fragment: Fragment, candidate_fragments: List[Fragment]) -> dict:
    """Find the best continuation among candidate fragments."""
    if not candidate_fragments:
        return {"best_fragment": None, "score": 0.0, "reasoning": "No candidates"}
    
    best_score = -1
    best_candidate = None
    best_reasoning = ""
    
    for candidate in candidate_fragments:
        result = continuity_score_detailed(
            current_fragment.end_part or "", 
            candidate.start_part or ""
        )
        
        if result["score"] > best_score:
            best_score = result["score"]
            best_candidate = candidate
            best_reasoning = result["reasoning"]
    
    logger.info(f"Best continuation score: {best_score}")
    return {
        "best_fragment": best_candidate,
        "score": best_score,
        "reasoning": best_reasoning
    }