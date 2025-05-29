from PIL import Image
import pytesseract
from typing import List, Tuple
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Preprocess image to improve OCR accuracy."""
    try:
        # Convert PIL to OpenCV
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply preprocessing
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Noise removal
        denoised = cv2.medianBlur(gray, 5)
        
        # Thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL
        processed_img = Image.fromarray(thresh)
        return processed_img
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image

def ocr_image_with_confidence(image: Image.Image, lang: str = 'vie+eng') -> Tuple[str, float]:
    """Perform OCR on image and return text with confidence score."""
    try:
        # Preprocess image
        processed_img = preprocess_image_for_ocr(image)
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(processed_img, lang=lang, output_type=pytesseract.Output.DICT)
        
        # Extract text and calculate average confidence
        text_parts = []
        confidences = []
        
        for i, word in enumerate(data['text']):
            if word.strip():
                text_parts.append(word)
                confidences.append(data['conf'][i])
        
        text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return text, avg_confidence / 100.0  # Convert to 0-1 scale
        
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return "", 0.0

def batch_ocr_images(images: List[Image.Image], lang: str = 'vie+eng') -> List[Tuple[str, float]]:
    """Perform OCR on multiple images."""
    results = []
    for i, img in enumerate(images):
        logger.info(f"Processing image {i+1}/{len(images)}")
        text, confidence = ocr_image_with_confidence(img, lang)
        results.append((text, confidence))
    return results