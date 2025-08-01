#!/usr/bin/env python3
"""
Invoice Extraction System

Main module for extracting invoice information from images:
- Invoice number
- Invoice date
- Line items
- Total amount
- Key-value pairs
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger, LoggerMixin
from data_processing.preprocess import ImagePreprocessor, TextExtractor, FieldExtractor
from extraction.ocr_engine import OCREngine, OCRProcessor
from model.train import FieldExtractionModel

logger = setup_logger(__name__)

class InvoiceExtractor(LoggerMixin):
    """Main invoice extraction class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize invoice extractor"""
        super().__init__()
        self.config = ConfigLoader(config_path).load_config()
        
        # Initialize components
        self.image_preprocessor = ImagePreprocessor(config_path)
        self.text_extractor = TextExtractor(config_path)
        self.field_extractor = FieldExtractor(config_path)
        self.ocr_processor = OCRProcessor(self.config['ocr'])
        
        # Load trained models if available
        self.models = self._load_trained_models()
        
        # Set device
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    def _load_trained_models(self) -> Dict[str, FieldExtractionModel]:
        """Load trained models for field extraction"""
        models = {}
        models_dir = Path("models/trained")
        
        if not models_dir.exists():
            self.log_warning("No trained models found. Using rule-based extraction only.")
            return models
        
        try:
            # Look for model files
            model_files = list(models_dir.glob("*.pt"))
            
            for model_file in model_files:
                # Extract field name from filename
                field_name = model_file.stem.split('_')[0]
                
                # Create model
                model = FieldExtractionModel(
                    self.config['model']['backbone'],
                    self.config['model']['num_classes'],
                    self.config['model']['dropout']
                )
                
                # Load weights
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                models[field_name] = model
                self.log_info(f"Loaded trained model for field: {field_name}")
            
        except Exception as e:
            self.log_error(f"Error loading trained models: {e}")
        
        return models
    
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract invoice information from image"""
        try:
            self.log_info(f"Processing image: {image_path}")
            
            # Preprocess image
            preprocessed_image = self.image_preprocessor.preprocess_image(image_path)
            enhanced_image = self.image_preprocessor.enhance_image(preprocessed_image)
            
            # Extract text using OCR
            text_results = self.text_extractor.extract_text(enhanced_image)
            full_text = self.text_extractor.extract_text_simple(enhanced_image)
            
            # Extract fields using rule-based approach
            extracted_fields = self._extract_fields_rule_based(full_text)
            
            # Extract fields using trained models
            model_extracted_fields = self._extract_fields_with_models(full_text)
            
            # Combine results
            final_fields = self._combine_extraction_results(extracted_fields, model_extracted_fields)
            
            # Extract key-value pairs
            key_value_pairs = self.ocr_processor.extract_key_value_pairs(enhanced_image)
            
            # Create extraction result
            result = {
                'image_path': image_path,
                'processing_timestamp': datetime.now().isoformat(),
                'extracted_fields': final_fields,
                'key_value_pairs': key_value_pairs,
                'full_text': full_text,
                'text_regions': text_results,
                'confidence_scores': self._calculate_confidence_scores(final_fields, key_value_pairs)
            }
            
            self.log_info(f"Extraction completed for {image_path}")
            return result
            
        except Exception as e:
            self.log_error(f"Error extracting from image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def _extract_fields_rule_based(self, text: str) -> Dict[str, Any]:
        """Extract fields using rule-based approach"""
        try:
            fields = {}
            
            # Extract invoice number
            invoice_number = self.field_extractor.extract_invoice_number(text)
            if invoice_number:
                fields['invoice_number'] = {
                    'value': invoice_number,
                    'method': 'rule_based',
                    'confidence': 0.8
                }
            
            # Extract invoice date
            invoice_date = self.field_extractor.extract_invoice_date(text)
            if invoice_date:
                fields['invoice_date'] = {
                    'value': invoice_date,
                    'method': 'rule_based',
                    'confidence': 0.8
                }
            
            # Extract line items
            line_items = self.field_extractor.extract_line_items(text)
            if line_items:
                fields['line_items'] = {
                    'value': line_items,
                    'method': 'rule_based',
                    'confidence': 0.7
                }
            
            # Extract total amount
            total_amount = self.field_extractor.extract_total_amount(text)
            if total_amount:
                fields['total_amount'] = {
                    'value': total_amount,
                    'method': 'rule_based',
                    'confidence': 0.8
                }
            
            return fields
            
        except Exception as e:
            self.log_error(f"Error in rule-based extraction: {e}")
            return {}
    
    def _extract_fields_with_models(self, text: str) -> Dict[str, Any]:
        """Extract fields using trained models"""
        try:
            fields = {}
            
            for field_name, model in self.models.items():
                try:
                    # Tokenize text
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.config['model']['backbone'])
                    
                    # Prepare input
                    inputs = tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.config['model']['max_length'],
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs['logits']
                        probabilities = torch.softmax(logits, dim=1)
                        prediction = torch.argmax(logits, dim=1)
                        confidence = torch.max(probabilities, dim=1)[0].item()
                    
                    # If model predicts positive class (1), extract the field
                    if prediction.item() == 1 and confidence > 0.5:
                        # Use rule-based extraction for the actual value
                        if field_name == 'invoice_number':
                            value = self.field_extractor.extract_invoice_number(text)
                        elif field_name == 'invoice_date':
                            value = self.field_extractor.extract_invoice_date(text)
                        elif field_name == 'line_items':
                            value = self.field_extractor.extract_line_items(text)
                        elif field_name == 'total_amount':
                            value = self.field_extractor.extract_total_amount(text)
                        else:
                            value = None
                        
                        if value:
                            fields[field_name] = {
                                'value': value,
                                'method': 'model_based',
                                'confidence': confidence
                            }
                
                except Exception as e:
                    self.log_error(f"Error extracting {field_name} with model: {e}")
            
            return fields
            
        except Exception as e:
            self.log_error(f"Error in model-based extraction: {e}")
            return {}
    
    def _combine_extraction_results(self, rule_based: Dict[str, Any], model_based: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and model-based extraction results"""
        try:
            combined = {}
            
            # Start with rule-based results
            for field_name, field_data in rule_based.items():
                combined[field_name] = field_data
            
            # Override with model-based results if confidence is higher
            for field_name, field_data in model_based.items():
                if field_name in combined:
                    # Compare confidence scores
                    if field_data['confidence'] > combined[field_name]['confidence']:
                        combined[field_name] = field_data
                else:
                    # Add new field
                    combined[field_name] = field_data
            
            return combined
            
        except Exception as e:
            self.log_error(f"Error combining extraction results: {e}")
            return rule_based
    
    def _calculate_confidence_scores(self, fields: Dict[str, Any], key_value_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall confidence scores"""
        try:
            confidence_scores = {}
            
            # Calculate field-specific confidence
            for field_name, field_data in fields.items():
                confidence_scores[field_name] = field_data.get('confidence', 0.0)
            
            # Calculate overall confidence
            if fields:
                overall_confidence = sum(field_data.get('confidence', 0.0) for field_data in fields.values()) / len(fields)
                confidence_scores['overall'] = overall_confidence
            else:
                confidence_scores['overall'] = 0.0
            
            return confidence_scores
            
        except Exception as e:
            self.log_error(f"Error calculating confidence scores: {e}")
            return {}
    
    def extract_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Extract invoice information from all images in directory"""
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            # Find all image files
            image_files = []
            for format in self.config['data']['supported_formats']:
                image_files.extend(directory.rglob(f"*.{format}"))
            
            self.log_info(f"Found {len(image_files)} image files in {directory_path}")
            
            # Process each image
            results = []
            for image_file in image_files:
                result = self.extract_from_image(str(image_file))
                results.append(result)
            
            return results
            
        except Exception as e:
            self.log_error(f"Error extracting from directory {directory_path}: {e}")
            return []
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None) -> None:
        """Save extraction results"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"extraction_results_{timestamp}.json"
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.log_info(f"Saved extraction results to {output_path}")
            
        except Exception as e:
            self.log_error(f"Error saving results: {e}")
    
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate extraction report"""
        try:
            report = {
                'total_images': len(results),
                'successful_extractions': 0,
                'failed_extractions': 0,
                'field_extraction_stats': {},
                'average_confidence': 0.0,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            total_confidence = 0.0
            field_counts = {}
            
            for result in results:
                if 'error' not in result:
                    report['successful_extractions'] += 1
                    
                    # Count extracted fields
                    for field_name, field_data in result.get('extracted_fields', {}).items():
                        if field_name not in field_counts:
                            field_counts[field_name] = 0
                        field_counts[field_name] += 1
                    
                    # Calculate confidence
                    confidence_scores = result.get('confidence_scores', {})
                    if 'overall' in confidence_scores:
                        total_confidence += confidence_scores['overall']
                else:
                    report['failed_extractions'] += 1
            
            # Calculate averages
            if report['successful_extractions'] > 0:
                report['average_confidence'] = total_confidence / report['successful_extractions']
            
            report['field_extraction_stats'] = field_counts
            
            return report
            
        except Exception as e:
            self.log_error(f"Error generating report: {e}")
            return {}

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Invoice Extraction System")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--report", action="store_true", help="Generate extraction report")
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = InvoiceExtractor(args.config)
        
        # Process input
        if os.path.isfile(args.input):
            # Single image
            results = [extractor.extract_from_image(args.input)]
        elif os.path.isdir(args.input):
            # Directory of images
            results = extractor.extract_from_directory(args.input)
        else:
            raise ValueError(f"Input path does not exist: {args.input}")
        
        # Save results
        if args.output:
            extractor.save_results(results, args.output)
        else:
            extractor.save_results(results)
        
        # Generate report if requested
        if args.report:
            report = extractor.generate_report(results)
            print("\nExtraction Report:")
            print(json.dumps(report, indent=2))
        
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        print(f"\nExtraction completed: {successful}/{len(results)} images processed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 