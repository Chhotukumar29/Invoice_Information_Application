#!/usr/bin/env python3
"""
OCR Engine for Invoice Extraction System

This module provides a unified interface for multiple OCR engines:
- Tesseract
- PaddleOCR
- EasyOCR
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logger import setup_logger, LoggerMixin

logger = setup_logger(__name__)

class OCREngine(LoggerMixin):
    """Unified OCR engine interface"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OCR engine with configuration"""
        super().__init__()
        self.config = config
        self.engine_name = config.get('engine', 'paddleocr')
        self.language = config.get('language', 'en')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.enable_gpu = config.get('enable_gpu', True)
        
        # Initialize the selected OCR engine
        self.engine = self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected OCR engine"""
        try:
            if self.engine_name == 'tesseract':
                return TesseractEngine(self.config)
            elif self.engine_name == 'paddleocr':
                return PaddleOCREngine(self.config)
            elif self.engine_name == 'easyocr':
                return EasyOCREngine(self.config)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.engine_name}")
        except Exception as e:
            self.log_error(f"Error initializing OCR engine {self.engine_name}: {e}")
            # Fallback to Tesseract
            self.log_warning("Falling back to Tesseract OCR")
            return TesseractEngine(self.config)
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text from image with bounding boxes and confidence scores"""
        try:
            results = self.engine.extract_text(image)
            
            # Filter results by confidence threshold
            filtered_results = [
                result for result in results 
                if result.get('confidence', 0) >= self.confidence_threshold
            ]
            
            self.log_info(f"Extracted {len(filtered_results)} text regions with confidence >= {self.confidence_threshold}")
            return filtered_results
            
        except Exception as e:
            self.log_error(f"Error extracting text: {e}")
            return []
    
    def extract_text_simple(self, image: np.ndarray) -> str:
        """Extract simple text string from image"""
        try:
            results = self.extract_text(image)
            text = " ".join([result['text'] for result in results])
            return text
        except Exception as e:
            self.log_error(f"Error extracting simple text: {e}")
            return ""

class TesseractEngine(LoggerMixin):
    """Tesseract OCR engine implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Tesseract engine"""
        super().__init__()
        self.config = config
        self.language = config.get('language', 'en')
        
        try:
            import pytesseract
            self.tesseract = pytesseract
            # Set Tesseract path if specified
            if 'tesseract_path' in config:
                self.tesseract.pytesseract.tesseract_cmd = config['tesseract_path']
        except ImportError:
            raise ImportError("pytesseract not installed. Install with: pip install pytesseract")
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using Tesseract OCR"""
        try:
            # Get detailed OCR data
            data = self.tesseract.image_to_data(image, lang=self.language, output_type=self.tesseract.Output.DICT)
            
            results = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                # Filter out empty text and low confidence
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    result = {
                        'text': data['text'][i].strip(),
                        'confidence': float(data['conf'][i]) / 100.0,
                        'bbox': (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        )
                    }
                    results.append(result)
            
            self.log_info(f"Tesseract extracted {len(results)} text regions")
            return results
            
        except Exception as e:
            self.log_error(f"Error in Tesseract OCR: {e}")
            return []

class PaddleOCREngine(LoggerMixin):
    """PaddleOCR engine implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PaddleOCR engine"""
        super().__init__()
        self.config = config
        self.language = config.get('language', 'en')
        self.enable_gpu = config.get('enable_gpu', True)
        
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                use_gpu=self.enable_gpu
            )
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr")
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using PaddleOCR"""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                from PIL import Image
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Perform OCR
            results = self.ocr.ocr(pil_image, cls=True)
            
            extracted_results = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line
                        
                        result = {
                            'text': text.strip(),
                            'confidence': float(confidence),
                            'bbox': bbox
                        }
                        extracted_results.append(result)
            
            self.log_info(f"PaddleOCR extracted {len(extracted_results)} text regions")
            return extracted_results
            
        except Exception as e:
            self.log_error(f"Error in PaddleOCR: {e}")
            return []

class EasyOCREngine(LoggerMixin):
    """EasyOCR engine implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EasyOCR engine"""
        super().__init__()
        self.config = config
        self.language = config.get('language', 'en')
        self.enable_gpu = config.get('enable_gpu', True)
        
        try:
            import easyocr
            self.reader = easyocr.Reader([self.language], gpu=self.enable_gpu)
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR"""
        try:
            # Perform OCR
            results = self.reader.readtext(image)
            
            extracted_results = []
            
            for (bbox, text, confidence) in results:
                result = {
                    'text': text.strip(),
                    'confidence': float(confidence),
                    'bbox': bbox
                }
                extracted_results.append(result)
            
            self.log_info(f"EasyOCR extracted {len(extracted_results)} text regions")
            return extracted_results
            
        except Exception as e:
            self.log_error(f"Error in EasyOCR: {e}")
            return []

class OCRProcessor(LoggerMixin):
    """Advanced OCR processing with post-processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OCR processor"""
        super().__init__()
        self.ocr_engine = OCREngine(config)
        self.config = config
    
    def extract_with_layout_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text with layout analysis"""
        try:
            # Extract text regions
            text_regions = self.ocr_engine.extract_text(image)
            
            # Analyze layout
            layout_analysis = self._analyze_layout(text_regions, image.shape)
            
            # Group text by regions
            grouped_text = self._group_text_by_regions(text_regions, layout_analysis)
            
            return {
                'text_regions': text_regions,
                'layout_analysis': layout_analysis,
                'grouped_text': grouped_text,
                'full_text': self.ocr_engine.extract_text_simple(image)
            }
            
        except Exception as e:
            self.log_error(f"Error in layout analysis: {e}")
            return {
                'text_regions': [],
                'layout_analysis': {},
                'grouped_text': {},
                'full_text': ""
            }
    
    def _analyze_layout(self, text_regions: List[Dict[str, Any]], image_shape: tuple) -> Dict[str, Any]:
        """Analyze document layout"""
        try:
            height, width = image_shape[:2]
            
            # Define regions (header, body, footer)
            header_threshold = height * 0.2
            footer_threshold = height * 0.8
            
            layout = {
                'header': [],
                'body': [],
                'footer': [],
                'left_column': [],
                'right_column': []
            }
            
            for region in text_regions:
                bbox = region['bbox']
                center_y = (bbox[1] + bbox[3]) / 2
                center_x = (bbox[0] + bbox[2]) / 2
                
                # Categorize by vertical position
                if center_y < header_threshold:
                    layout['header'].append(region)
                elif center_y > footer_threshold:
                    layout['footer'].append(region)
                else:
                    layout['body'].append(region)
                
                # Categorize by horizontal position
                if center_x < width / 2:
                    layout['left_column'].append(region)
                else:
                    layout['right_column'].append(region)
            
            return layout
            
        except Exception as e:
            self.log_error(f"Error analyzing layout: {e}")
            return {}
    
    def _group_text_by_regions(self, text_regions: List[Dict[str, Any]], layout: Dict[str, Any]) -> Dict[str, str]:
        """Group text by layout regions"""
        try:
            grouped = {}
            
            for region_name, regions in layout.items():
                # Sort regions by position (top to bottom, left to right)
                sorted_regions = sorted(regions, key=lambda x: (x['bbox'][1], x['bbox'][0]))
                
                # Combine text
                text = " ".join([region['text'] for region in sorted_regions])
                grouped[region_name] = text
            
            return grouped
            
        except Exception as e:
            self.log_error(f"Error grouping text: {e}")
            return {}
    
    def extract_key_value_pairs(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract key-value pairs from document"""
        try:
            # Get layout analysis
            layout_result = self.extract_with_layout_analysis(image)
            
            # Extract potential key-value pairs
            key_value_pairs = []
            
            # Look for patterns like "Key: Value" or "Key Value"
            text_regions = layout_result['text_regions']
            
            for region in text_regions:
                text = region['text']
                
                # Look for colon-separated pairs
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        if key and value:
                            key_value_pairs.append({
                                'key': key,
                                'value': value,
                                'confidence': region['confidence'],
                                'bbox': region['bbox']
                            })
                
                # Look for common invoice field patterns
                field_patterns = {
                    'invoice_number': r'(?i)(invoice\s*#?\s*|inv\s*#?\s*)(\d+)',
                    'invoice_date': r'(?i)(date|invoice\s*date)\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    'total_amount': r'(?i)(total|amount\s*due|grand\s*total)\s*[:.]?\s*(\$?\d+\.?\d*)',
                    'due_date': r'(?i)(due\s*date|payment\s*due)\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
                }
                
                for field_name, pattern in field_patterns.items():
                    import re
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if len(match) == 2:
                            key_value_pairs.append({
                                'key': match[0].strip(),
                                'value': match[1].strip(),
                                'field_type': field_name,
                                'confidence': region['confidence'],
                                'bbox': region['bbox']
                            })
            
            self.log_info(f"Extracted {len(key_value_pairs)} key-value pairs")
            return key_value_pairs
            
        except Exception as e:
            self.log_error(f"Error extracting key-value pairs: {e}")
            return []

def create_ocr_engine(config: Dict[str, Any]) -> OCREngine:
    """Factory function to create OCR engine"""
    return OCREngine(config)

def create_ocr_processor(config: Dict[str, Any]) -> OCRProcessor:
    """Factory function to create OCR processor"""
    return OCRProcessor(config) 