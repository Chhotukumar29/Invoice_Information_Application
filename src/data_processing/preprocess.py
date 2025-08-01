#!/usr/bin/env python3
"""
Data Preprocessing for Invoice Extraction System

This module handles preprocessing of invoice images including:
- Image preprocessing and enhancement
- OCR text extraction
- Layout analysis
- Data preparation for training
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm
import re
from datetime import datetime
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger, LoggerMixin
from extraction.ocr_engine import OCREngine

logger = setup_logger(__name__)

class ImagePreprocessor(LoggerMixin):
    """Handles image preprocessing and enhancement"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the image preprocessor"""
        super().__init__()
        self.config = ConfigLoader(config_path).load_config()
        self.max_size = tuple(self.config['data']['max_image_size'])
        self.supported_formats = self.config['data']['supported_formats']
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Resize if necessary
            if image.shape[:2] != self.max_size:
                resized = cv2.resize(thresh, self.max_size, interpolation=cv2.INTER_AREA)
            else:
                resized = thresh
            
            self.log_info(f"Preprocessed image: {image_path}")
            return resized
            
        except Exception as e:
            self.log_error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply additional image enhancement techniques"""
        try:
            # Apply morphological operations
            kernel = np.ones((1, 1), np.uint8)
            enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
            # Apply Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.log_error(f"Error enhancing image: {e}")
            return image
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges in the image"""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(image, 50, 150)
            return edges
        except Exception as e:
            self.log_error(f"Error detecting edges: {e}")
            return image
    
    def find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """Find contours in the image"""
        try:
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except Exception as e:
            self.log_error(f"Error finding contours: {e}")
            return []

class TextExtractor(LoggerMixin):
    """Handles text extraction from images"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the text extractor"""
        super().__init__()
        self.config = ConfigLoader(config_path).load_config()
        self.ocr_engine = OCREngine(self.config['ocr'])
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text from image with bounding boxes"""
        try:
            # Extract text using OCR engine
            results = self.ocr_engine.extract_text(image)
            
            self.log_info(f"Extracted {len(results)} text regions")
            return results
            
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

class FieldExtractor(LoggerMixin):
    """Extracts specific fields from invoice text"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the field extractor"""
        super().__init__()
        self.config = ConfigLoader(config_path).load_config()
        self.field_configs = self.config['fields']
    
    def extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number from text"""
        try:
            patterns = self.field_configs['invoice_number']['patterns']
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    invoice_number = matches[0]
                    self.log_info(f"Extracted invoice number: {invoice_number}")
                    return invoice_number
            
            self.log_warning("No invoice number found")
            return None
            
        except Exception as e:
            self.log_error(f"Error extracting invoice number: {e}")
            return None
    
    def extract_invoice_date(self, text: str) -> Optional[str]:
        """Extract invoice date from text"""
        try:
            patterns = self.field_configs['invoice_date']['patterns']
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    date_str = matches[0]
                    # Try to parse and standardize date format
                    try:
                        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                        standardized_date = parsed_date.strftime("%Y-%m-%d")
                        self.log_info(f"Extracted invoice date: {standardized_date}")
                        return standardized_date
                    except ValueError:
                        # Try other date formats
                        for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                            try:
                                parsed_date = datetime.strptime(date_str, fmt)
                                standardized_date = parsed_date.strftime("%Y-%m-%d")
                                self.log_info(f"Extracted invoice date: {standardized_date}")
                                return standardized_date
                            except ValueError:
                                continue
                        
                        # Return original if parsing fails
                        return date_str
            
            self.log_warning("No invoice date found")
            return None
            
        except Exception as e:
            self.log_error(f"Error extracting invoice date: {e}")
            return None
    
    def extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract line items from text"""
        try:
            line_items = []
            
            # Split text into lines
            lines = text.split('\n')
            
            # Look for line item patterns
            item_patterns = self.field_configs['line_items']['patterns']
            
            for line in lines:
                # Check if line contains item information
                if any(pattern.lower() in line.lower() for pattern in item_patterns):
                    # Extract item details
                    item = self._parse_line_item(line)
                    if item:
                        line_items.append(item)
            
            self.log_info(f"Extracted {len(line_items)} line items")
            return line_items
            
        except Exception as e:
            self.log_error(f"Error extracting line items: {e}")
            return []
    
    def _parse_line_item(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line item"""
        try:
            # Simple parsing - can be enhanced with more sophisticated NLP
            parts = line.split()
            
            if len(parts) < 2:
                return None
            
            # Try to extract quantity and price
            item = {
                'description': ' '.join(parts[:-2]) if len(parts) > 2 else parts[0],
                'quantity': self._extract_number(parts[-2]),
                'price': self._extract_number(parts[-1])
            }
            
            return item
            
        except Exception as e:
            self.log_error(f"Error parsing line item: {e}")
            return None
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text"""
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[^\d.]', '', text)
            if cleaned:
                return float(cleaned)
            return None
        except ValueError:
            return None
    
    def extract_total_amount(self, text: str) -> Optional[float]:
        """Extract total amount from text"""
        try:
            patterns = self.field_configs['total_amount']['patterns']
            
            for pattern in patterns:
                # Look for total amount patterns
                matches = re.findall(rf"{pattern}.*?(\d+\.?\d*)", text, re.IGNORECASE)
                if matches:
                    amount = float(matches[0])
                    self.log_info(f"Extracted total amount: {amount}")
                    return amount
            
            self.log_warning("No total amount found")
            return None
            
        except Exception as e:
            self.log_error(f"Error extracting total amount: {e}")
            return None

class DataPreprocessor(LoggerMixin):
    """Main data preprocessing class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data preprocessor"""
        super().__init__()
        self.config = ConfigLoader(config_path).load_config()
        self.image_preprocessor = ImagePreprocessor(config_path)
        self.text_extractor = TextExtractor(config_path)
        self.field_extractor = FieldExtractor(config_path)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_dataset(self, input_dir: str = None) -> None:
        """Preprocess entire dataset"""
        if input_dir is None:
            input_dir = self.raw_dir
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")
        
        self.log_info(f"Starting preprocessing of dataset in: {input_path}")
        
        # Find all image files
        image_files = []
        for format in self.config['data']['supported_formats']:
            image_files.extend(input_path.rglob(f"*.{format}"))
        
        self.log_info(f"Found {len(image_files)} image files")
        
        # Process each image
        processed_data = []
        
        for image_file in tqdm(image_files, desc="Preprocessing images"):
            try:
                processed_item = self.preprocess_single_image(str(image_file))
                if processed_item:
                    processed_data.append(processed_item)
            except Exception as e:
                self.log_error(f"Error processing {image_file}: {e}")
        
        # Save processed data
        self._save_processed_data(processed_data)
        
        self.log_info(f"Preprocessing completed. Processed {len(processed_data)} images")
    
    def preprocess_single_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Preprocess a single image and extract information"""
        try:
            # Preprocess image
            preprocessed_image = self.image_preprocessor.preprocess_image(image_path)
            enhanced_image = self.image_preprocessor.enhance_image(preprocessed_image)
            
            # Extract text
            text_results = self.text_extractor.extract_text(enhanced_image)
            full_text = self.text_extractor.extract_text_simple(enhanced_image)
            
            # Extract fields
            invoice_number = self.field_extractor.extract_invoice_number(full_text)
            invoice_date = self.field_extractor.extract_invoice_date(full_text)
            line_items = self.field_extractor.extract_line_items(full_text)
            total_amount = self.field_extractor.extract_total_amount(full_text)
            
            # Create processed item
            processed_item = {
                'image_path': image_path,
                'preprocessed_image_path': str(self.processed_dir / f"{Path(image_path).stem}_processed.png"),
                'text_results': text_results,
                'full_text': full_text,
                'extracted_fields': {
                    'invoice_number': invoice_number,
                    'invoice_date': invoice_date,
                    'line_items': line_items,
                    'total_amount': total_amount
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Save preprocessed image
            cv2.imwrite(processed_item['preprocessed_image_path'], enhanced_image)
            
            return processed_item
            
        except Exception as e:
            self.log_error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def _save_processed_data(self, processed_data: List[Dict[str, Any]]) -> None:
        """Save processed data to files"""
        try:
            # Save as JSON
            json_path = self.processed_dir / "processed_data.json"
            with open(json_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            # Save as CSV for analysis
            csv_data = []
            for item in processed_data:
                csv_item = {
                    'image_path': item['image_path'],
                    'invoice_number': item['extracted_fields']['invoice_number'],
                    'invoice_date': item['extracted_fields']['invoice_date'],
                    'total_amount': item['extracted_fields']['total_amount'],
                    'num_line_items': len(item['extracted_fields']['line_items']),
                    'processing_timestamp': item['processing_timestamp']
                }
                csv_data.append(csv_item)
            
            csv_path = self.processed_dir / "processed_data.csv"
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            
            self.log_info(f"Saved processed data to {json_path} and {csv_path}")
            
        except Exception as e:
            self.log_error(f"Error saving processed data: {e}")
    
    def create_training_data(self) -> None:
        """Create training data from processed data"""
        try:
            # Load processed data
            json_path = self.processed_dir / "processed_data.json"
            if not json_path.exists():
                raise FileNotFoundError("Processed data not found. Run preprocessing first.")
            
            with open(json_path, 'r') as f:
                processed_data = json.load(f)
            
            # Create training examples
            training_data = []
            
            for item in processed_data:
                # Create positive examples for each field
                if item['extracted_fields']['invoice_number']:
                    training_data.append({
                        'text': item['full_text'],
                        'field': 'invoice_number',
                        'value': item['extracted_fields']['invoice_number'],
                        'label': 1
                    })
                
                if item['extracted_fields']['invoice_date']:
                    training_data.append({
                        'text': item['full_text'],
                        'field': 'invoice_date',
                        'value': item['extracted_fields']['invoice_date'],
                        'label': 1
                    })
                
                # Create negative examples
                training_data.append({
                    'text': item['full_text'],
                    'field': 'random_field',
                    'value': 'none',
                    'label': 0
                })
            
            # Save training data
            training_path = self.processed_dir / "training_data.json"
            with open(training_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            self.log_info(f"Created training data with {len(training_data)} examples")
            
        except Exception as e:
            self.log_error(f"Error creating training data: {e}")

def main():
    """Main function to run preprocessing"""
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_dataset()
    preprocessor.create_training_data()

if __name__ == "__main__":
    main() 