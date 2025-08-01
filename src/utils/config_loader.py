#!/usr/bin/env python3
"""
Configuration Loader for Invoice Extraction System

This module handles loading and validating configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Loads and validates configuration files"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the configuration loader"""
        self.config_path = Path(config_path)
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate configuration
            self._validate_config(config)
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        required_sections = [
            'data', 'model', 'ocr', 'fields', 'training', 
            'data_sources', 'pdf_processing', 'logging', 'output'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        if 'raw_dir' not in config['data']:
            raise ValueError("Missing 'raw_dir' in data configuration")
        
        # Validate model configuration
        if 'architecture' not in config['model']:
            raise ValueError("Missing 'architecture' in model configuration")
        
        # Validate OCR configuration
        if 'engine' not in config['ocr']:
            raise ValueError("Missing 'engine' in OCR configuration")
        
        # Validate fields configuration
        required_fields = ['invoice_number', 'invoice_date', 'line_items', 'total_amount']
        for field in required_fields:
            if field not in config['fields']:
                raise ValueError(f"Missing required field configuration: {field}")
    
    def save_config(self, config: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving configuration: {e}")
    
    def get_field_config(self, field_name: str) -> Dict[str, Any]:
        """Get configuration for a specific field"""
        config = self.load_config()
        
        if field_name not in config['fields']:
            raise ValueError(f"Field '{field_name}' not found in configuration")
        
        return config['fields'][field_name]
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        config = self.load_config()
        return config['model']
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR configuration"""
        config = self.load_config()
        return config['ocr']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        config = self.load_config()
        return config['training'] 