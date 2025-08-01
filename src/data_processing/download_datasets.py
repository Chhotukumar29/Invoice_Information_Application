#!/usr/bin/env python3
"""
Dataset Downloader for Invoice Extraction System

This module downloads invoice datasets from various sources including:
- Kaggle datasets
- Hugging Face datasets
- GitHub repositories
- PDF to image conversion
"""

import os
import sys
import yaml
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import subprocess
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DatasetDownloader:
    """Downloads invoice datasets from various sources"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the dataset downloader"""
        self.config = ConfigLoader(config_path).load_config()
        self.data_dir = Path(self.config['data']['raw_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_datasets(self) -> None:
        """Download datasets from Kaggle"""
        logger.info("Downloading Kaggle datasets...")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            datasets = self.config['data_sources']['kaggle']['datasets']
            
            for dataset in datasets:
                try:
                    logger.info(f"Downloading Kaggle dataset: {dataset}")
                    api.dataset_download_files(dataset, path=self.data_dir / "kaggle", unzip=True)
                    logger.info(f"Successfully downloaded {dataset}")
                except Exception as e:
                    logger.error(f"Failed to download Kaggle dataset {dataset}: {e}")
                    
        except ImportError:
            logger.warning("Kaggle API not available. Install with: pip install kaggle")
        except Exception as e:
            logger.error(f"Error downloading Kaggle datasets: {e}")
    
    def download_huggingface_datasets(self) -> None:
        """Download datasets from Hugging Face"""
        logger.info("Downloading Hugging Face datasets...")
        
        try:
            from datasets import load_dataset
            from huggingface_hub import login
            
            # Try to login if token is available
            try:
                login()
            except:
                logger.info("No Hugging Face token provided, proceeding without authentication")
            
            datasets = self.config['data_sources']['huggingface']['datasets']
            
            for dataset_name in datasets:
                try:
                    logger.info(f"Downloading Hugging Face dataset: {dataset_name}")
                    
                    # Load dataset
                    dataset = load_dataset(dataset_name)
                    
                    # Save to local directory
                    output_dir = self.data_dir / "huggingface" / dataset_name.replace("/", "_")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as JSON files
                    for split_name, split_data in dataset.items():
                        output_file = output_dir / f"{split_name}.json"
                        split_data.to_json(str(output_file))
                    
                    logger.info(f"Successfully downloaded {dataset_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to download Hugging Face dataset {dataset_name}: {e}")
                    
        except ImportError:
            logger.warning("Hugging Face datasets not available. Install with: pip install datasets")
        except Exception as e:
            logger.error(f"Error downloading Hugging Face datasets: {e}")
    
    def download_github_datasets(self) -> None:
        """Download datasets from GitHub repositories"""
        logger.info("Downloading GitHub datasets...")
        
        repositories = self.config['data_sources']['github']['repositories']
        
        for repo_url in repositories:
            try:
                logger.info(f"Downloading from GitHub: {repo_url}")
                
                # Extract repository name
                repo_name = repo_url.split("/")[-1]
                output_dir = self.data_dir / "github" / repo_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Clone repository
                subprocess.run([
                    "git", "clone", repo_url, str(output_dir)
                ], check=True)
                
                logger.info(f"Successfully downloaded {repo_name}")
                
            except Exception as e:
                logger.error(f"Failed to download GitHub repository {repo_url}: {e}")
    
    def download_sample_invoice_datasets(self) -> None:
        """Download sample invoice datasets from public sources"""
        logger.info("Downloading sample invoice datasets...")
        
        # Sample invoice datasets URLs
        sample_datasets = [
            {
                "name": "invoice_samples",
                "url": "https://github.com/username/invoice-samples/archive/refs/heads/main.zip",
                "type": "zip"
            },
            {
                "name": "document_understanding",
                "url": "https://huggingface.co/datasets/microsoft/layoutlm-document-understanding",
                "type": "huggingface"
            }
        ]
        
        for dataset in sample_datasets:
            try:
                logger.info(f"Downloading sample dataset: {dataset['name']}")
                
                if dataset['type'] == 'zip':
                    self._download_zip_file(dataset['url'], dataset['name'])
                elif dataset['type'] == 'huggingface':
                    self.download_huggingface_datasets()
                    
            except Exception as e:
                logger.error(f"Failed to download sample dataset {dataset['name']}: {e}")
    
    def _download_zip_file(self, url: str, name: str) -> None:
        """Download and extract a zip file"""
        output_dir = self.data_dir / "samples" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        zip_path = output_dir / f"{name}.zip"
        
        with open(zip_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {name}"):
                f.write(chunk)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Remove zip file
        zip_path.unlink()
        
        logger.info(f"Successfully downloaded and extracted {name}")
    
    def convert_pdfs_to_images(self, pdf_dir: str = None) -> None:
        """Convert PDF files to images"""
        logger.info("Converting PDF files to images...")
        
        if pdf_dir is None:
            pdf_dir = self.data_dir / "pdfs"
        
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.warning(f"PDF directory {pdf_dir} does not exist")
            return
        
        output_dir = Path(self.config['pdf_processing']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from pdf2image import convert_from_path
            
            pdf_files = list(pdf_dir.glob("*.pdf"))
            
            for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
                try:
                    # Convert PDF to images
                    images = convert_from_path(
                        pdf_file,
                        dpi=self.config['pdf_processing']['dpi'],
                        fmt=self.config['pdf_processing']['format'],
                        thread_count=self.config['pdf_processing']['thread_count']
                    )
                    
                    # Save images
                    base_name = pdf_file.stem
                    for i, image in enumerate(images):
                        output_path = output_dir / f"{base_name}_page_{i+1}.png"
                        image.save(output_path, self.config['pdf_processing']['format'])
                    
                    logger.info(f"Converted {pdf_file.name} to {len(images)} images")
                    
                except Exception as e:
                    logger.error(f"Failed to convert {pdf_file}: {e}")
                    
        except ImportError:
            logger.warning("pdf2image not available. Install with: pip install pdf2image")
        except Exception as e:
            logger.error(f"Error converting PDFs: {e}")
    
    def create_sample_invoices(self) -> None:
        """Create sample invoice images for testing"""
        logger.info("Creating sample invoice images...")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random
            
            output_dir = self.data_dir / "samples" / "generated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sample invoice templates
            templates = [
                {
                    "name": "simple_invoice",
                    "fields": ["Invoice #", "Date", "Item", "Quantity", "Price", "Total"]
                },
                {
                    "name": "detailed_invoice", 
                    "fields": ["Invoice Number", "Invoice Date", "Due Date", "Item Description", "Qty", "Unit Price", "Amount"]
                }
            ]
            
            for template in templates:
                for i in range(5):  # Create 5 samples per template
                    # Create image
                    img = Image.new('RGB', (800, 600), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    # Add text
                    y_position = 50
                    for field in template['fields']:
                        text = f"{field}: {self._generate_sample_value(field)}"
                        draw.text((50, y_position), text, fill='black')
                        y_position += 30
                    
                    # Save image
                    output_path = output_dir / f"{template['name']}_{i+1}.png"
                    img.save(output_path)
                    
            logger.info("Created sample invoice images")
            
        except Exception as e:
            logger.error(f"Error creating sample invoices: {e}")
    
    def _generate_sample_value(self, field: str) -> str:
        """Generate sample values for invoice fields"""
        import random
        from datetime import datetime, timedelta
        
        if "invoice" in field.lower() and "number" in field.lower():
            return f"INV-{random.randint(10000, 99999)}"
        elif "date" in field.lower():
            date = datetime.now() - timedelta(days=random.randint(0, 30))
            return date.strftime("%Y-%m-%d")
        elif "item" in field.lower():
            items = ["Product A", "Service B", "Item C", "Product D"]
            return random.choice(items)
        elif "quantity" in field.lower() or "qty" in field.lower():
            return str(random.randint(1, 10))
        elif "price" in field.lower():
            return f"${random.uniform(10.0, 1000.0):.2f}"
        elif "total" in field.lower():
            return f"${random.uniform(100.0, 5000.0):.2f}"
        else:
            return f"Sample {field}"
    
    def download_all_datasets(self) -> None:
        """Download all available datasets"""
        logger.info("Starting dataset download process...")
        
        # Download from all sources
        self.download_kaggle_datasets()
        self.download_huggingface_datasets()
        self.download_github_datasets()
        self.download_sample_invoice_datasets()
        
        # Convert PDFs if available
        self.convert_pdfs_to_images()
        
        # Create sample invoices
        self.create_sample_invoices()
        
        logger.info("Dataset download process completed!")

def main():
    """Main function to download datasets"""
    downloader = DatasetDownloader()
    downloader.download_all_datasets()

if __name__ == "__main__":
    main() 