"""
OCR processing tool using Triton inference server
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import tempfile
import os
import io
import tritonclient.grpc as grpcclient

from ..base import ExtractionTool, ToolResult
from ...config.settings import get_settings

logger = logging.getLogger(__name__)

class OCRProcessor(ExtractionTool):
    """OCR processing using Triton inference server"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.settings = get_settings()
        self.language = self.get_config_value("language", self.settings.tesseract_lang)
        self.dpi = self.get_config_value("dpi", self.settings.ocr_dpi)
        self.min_confidence = self.get_config_value("min_confidence", self.settings.min_ocr_confidence)
        self.preprocess_enabled = self.get_config_value("preprocess_enabled", True)
        self.triton_url = self.get_config_value("triton_url", "10.124.68.252:8001")
        self.triton_model_name = self.get_config_value("triton_model_name", "OCR")
        
        # Test Triton availability
        self._test_triton_connection()
    
    @property
    def name(self) -> str:
        return "ocr_processor"
    
    @property
    def description(self) -> str:
        return "Process scanned PDF pages using OCR with Triton inference server"
    
    @property
    def capabilities(self) -> List[str]:
        return super().capabilities + ["ocr_processing", "image_preprocessing", "confidence_scoring"]
    
    def _test_triton_connection(self):
        """Test if Triton server is available"""
        try:
            client = grpcclient.InferenceServerClient(url=self.triton_url)
            logger.info(f"Triton server at {self.triton_url} is available")
        except Exception as e:
            logger.error(f"Triton server not available: {e}")
            raise RuntimeError(f"Triton inference server not found at {self.triton_url}")
    
    def _execute(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> ToolResult:
        """
        Process PDF pages with OCR
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: Specific pages to process (None for all)
            
        Returns:
            ToolResult with OCR results
        """
        try:
            # Convert PDF to images
            images_result = self._pdf_to_images(pdf_path, page_numbers)
            if not images_result.success:
                return images_result
            
            images_data = images_result.data["images"]
            
            # Process each image with OCR
            ocr_results = []
            total_confidence = 0.0
            successful_pages = 0
            
            for img_data in images_data:
                page_num = img_data["page_number"]
                image = img_data["image"]
                
                logger.info(f"Processing page {page_num} with OCR...")
                
                # Preprocess image if enabled
                if self.preprocess_enabled:
                    processed_image = self._preprocess_image(image)
                else:
                    processed_image = image
                
                # Convert PIL image to numpy array for Triton
                img_array = np.array(processed_image)
                if len(img_array.shape) == 2:
                    # Convert grayscale to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif img_array.shape[2] == 4:
                    # Convert RGBA to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                # Perform OCR with Triton
                ocr_result = self._ocr_image_triton(img_array, page_num)
                
                if ocr_result["success"]:
                    ocr_results.append(ocr_result)
                    total_confidence += ocr_result["confidence"]
                    successful_pages += 1
                else:
                    # Still add failed results for completeness
                    ocr_results.append({
                        "page_number": page_num,
                        "text": "",
                        "confidence": 0.0,
                        "success": False,
                        "error": ocr_result.get("error", "OCR failed")
                    })
            
            # Calculate overall confidence
            avg_confidence = total_confidence / max(successful_pages, 1)
            
            # Prepare result
            result_data = {
                "ocr_results": ocr_results,
                "total_pages_processed": len(images_data),
                "successful_pages": successful_pages,
                "failed_pages": len(images_data) - successful_pages,
                "average_confidence": avg_confidence,
                "language_used": self.language,
                "dpi_used": self.dpi,
                "preprocessing_enabled": self.preprocess_enabled
            }
            
            return ToolResult(
                success=successful_pages > 0,
                data=result_data,
                confidence=avg_confidence,
                metadata={
                    "total_pages": len(images_data),
                    "success_rate": successful_pages / len(images_data) if images_data else 0,
                    "ocr_engine": "triton"
                }
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return ToolResult(
                success=False,
                error_message=f"OCR processing failed: {str(e)}"
            )
    
    def _pdf_to_images(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> ToolResult:
        """Convert PDF pages to images"""
        try:
            doc = fitz.open(pdf_path)
            images_data = []
            
            # Determine which pages to process
            if page_numbers is None:
                pages_to_process = range(doc.page_count)
            else:
                pages_to_process = [p - 1 for p in page_numbers if 1 <= p <= doc.page_count]  # Convert to 0-based
            
            for page_idx in pages_to_process:
                page = doc[page_idx]
                
                # Convert page to image
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # Scale to desired DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                images_data.append({
                    "page_number": page_idx + 1,
                    "image": image,
                    "width": pix.width,
                    "height": pix.height
                })
            
            doc.close()
            
            return ToolResult(
                success=True,
                data={"images": images_data},
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"PDF to images conversion failed: {str(e)}")
            return ToolResult(
                success=False,
                error_message=f"PDF to images failed: {str(e)}"
            )
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy"""
        try:
            # Convert PIL to OpenCV
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            # Apply preprocessing techniques
            processed = self._apply_preprocessing_pipeline(gray)
            
            # Convert back to PIL
            return Image.fromarray(processed)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return image
    
    def _apply_preprocessing_pipeline(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply comprehensive preprocessing pipeline"""
        
        # 1. Noise reduction
        denoised = cv2.medianBlur(gray_image, 3)
        
        # 2. Adaptive thresholding for better text/background separation
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Optional: Deskewing (basic rotation correction)
        # This is computationally expensive, so only apply if needed
        if self.get_config_value("enable_deskewing", False):
            cleaned = self._deskew_image(cleaned)
        
        return cleaned
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Basic deskewing to correct slight rotations"""
        try:
            # Find lines using HoughLines
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle
                angles = []
                for line in lines[:20]:  # Use top 20 lines
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    if angle > 45:
                        angle = angle - 90
                    angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only rotate if angle is significant
                    if abs(median_angle) > 0.5:
                        # Rotate image
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        return rotated
            
            return image
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            return image
    
    def _ocr_image_triton(self, image: np.ndarray, page_number: int) -> Dict[str, Any]:
        """Perform OCR on preprocessed image using Triton inference server"""
        try:
            # Call Triton OCR
            output_names = ['word_list', 'box_list', 'full_text']
            
            try:
                ocr_outputs = self._call_triton_ocr(image, self.triton_url, self.triton_model_name, output_names=output_names)
                
                # Process OCR text
                if len(ocr_outputs) > 2 and ocr_outputs[2] is not None:
                    full_text_np = ocr_outputs[2]
                    extracted_text = self._process_ocr_text(full_text_np)
                else:
                    logger.warning(f"Full text output not available for page {page_number}")
                    extracted_text = ""
                
                # For simplicity, we'll assign a default confidence score
                # In a real implementation, you might want to derive this from the OCR model
                confidence = 0.85  # Default confidence
                
                result = {
                    "page_number": page_number,
                    "text": extracted_text,
                    "confidence": confidence,
                    "word_count": len(extracted_text.split()) if extracted_text else 0,
                    "char_count": len(extracted_text) if extracted_text else 0,
                    "meets_threshold": confidence >= self.min_confidence,
                    "success": confidence >= self.min_confidence and len(extracted_text.strip()) > 0,
                    "language_detected": self._detect_language_simple(extracted_text)
                }
                
                if not result["meets_threshold"]:
                    result["warning"] = f"Low confidence: {confidence*100:.1f}% < {self.min_confidence*100:.1f}%"
                
                return result
                
            except Exception as e:
                logger.error(f"Triton OCR call failed for page {page_number}: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"OCR failed for page {page_number}: {str(e)}")
            return {
                "page_number": page_number,
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _call_triton_ocr(self, image_inp: np.ndarray, triton_url: str, model_name='OCR', 
                        input_name='input', output_names=['word_list', 'box_list', 'full_text'], 
                        data_type='UINT8'):
        """Calls Triton OCR service."""
        infer_input = grpcclient.InferInput(input_name, image_inp.shape, data_type)
        infer_input.set_data_from_numpy(image_inp)
        
        try:
            client = grpcclient.InferenceServerClient(url=triton_url)
        except Exception as e:
            logger.error(f"Failed to create Triton client for URL {triton_url}: {e}")
            raise
            
        outputs_req = [grpcclient.InferRequestedOutput(output_name) for output_name in output_names]
        
        try:
            response = client.infer(model_name=model_name, inputs=[infer_input], outputs=outputs_req)
            output_data = [response.as_numpy(output_name) for output_name in output_names]
            return output_data
        except Exception as e:
            logger.error(f"Error during Triton inference for model {model_name}: {e}")
            raise
    
    def _process_ocr_text(self, full_text_np_array: np.ndarray) -> str:
        """Decodes OCR text output."""
        if isinstance(full_text_np_array, np.ndarray) and full_text_np_array.size > 0:
            item = full_text_np_array.flat[0]
            if isinstance(item, bytes):
                try:
                    return item.decode('utf-8')
                except UnicodeDecodeError:
                    return f"Error decoding text (bytes: {item[:100]}...)"
            elif isinstance(item, np.bytes_): # Handle numpy.bytes_
                try:
                    return item.decode('utf-8')
                except UnicodeDecodeError:
                    return f"Error decoding text (np.bytes_: {item[:100]}...)"
            else:
                # If it's already a string or other type, convert to string
                return str(item)
        return "[No text data or not a recognized byte string]"
    
    def _detect_language_simple(self, text: str) -> str:
        """Simple language detection for Vietnamese vs English"""
        if not text:
            return "unknown"
        
        vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        
        vietnamese_count = sum(1 for char in text.lower() if char in vietnamese_chars)
        total_alpha_chars = sum(1 for char in text if char.isalpha())
        
        if total_alpha_chars == 0:
            return "unknown"
        
        vietnamese_ratio = vietnamese_count / total_alpha_chars
        
        if vietnamese_ratio > 0.1:  # More than 10% Vietnamese characters
            return "vietnamese"
        else:
            return "english"
    
    def process_single_image(self, image_path: str) -> ToolResult:
        """Process a single image file with OCR"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Preprocess if enabled  
            if self.preprocess_enabled:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image
            
            # Convert PIL image to numpy array for Triton
            img_array = np.array(processed_image)
            if len(img_array.shape) == 2:
                # Convert grayscale to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                # Convert RGBA to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Perform OCR with Triton
            ocr_result = self._ocr_image_triton(img_array, 1)
            
            return ToolResult(
                success=ocr_result["success"],
                data=ocr_result,
                confidence=ocr_result["confidence"]
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Single image OCR failed: {str(e)}"
            )
    
    def batch_process_images(self, image_paths: List[str]) -> ToolResult:
        """Process multiple image files"""
        results = []
        successful_count = 0
        
        for i, img_path in enumerate(image_paths):
            result = self.process_single_image(img_path)
            
            if result.success:
                successful_count += 1
                
            results.append({
                "image_path": img_path,
                "result": result.data,
                "success": result.success
            })
        
        avg_confidence = sum(r["result"].get("confidence", 0) for r in results) / len(results) if results else 0
        
        return ToolResult(
            success=successful_count > 0,
            data={
                "batch_results": results,
                "total_processed": len(image_paths),
                "successful": successful_count,
                "failed": len(image_paths) - successful_count
            },
            confidence=avg_confidence
        )