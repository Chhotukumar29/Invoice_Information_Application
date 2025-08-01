# Invoice Extraction System Documentation

## Overview

The Invoice Extraction System is a scalable, machine learning-based solution for extracting key information from invoice images. The system can extract invoice numbers, dates, line items, and total amounts using both rule-based and model-based approaches.

## Features

### Core Capabilities
- **Invoice Number Extraction**: Extract unique invoice identifiers using pattern matching and ML models
- **Invoice Date Extraction**: Extract and standardize invoice dates in various formats
- **Line Items Extraction**: Extract product/service details with quantities and prices
- **Total Amount Extraction**: Extract total amounts and payment information
- **Key-Value Pair Extraction**: Extract structured data from unstructured invoice images

### Scalable Architecture
- **Modular Design**: Easy to extend with new field types
- **Multiple OCR Engines**: Support for Tesseract, PaddleOCR, and EasyOCR
- **Model Training Pipeline**: Train custom models for specific domains
- **Configuration-Driven**: All parameters configurable via YAML files

## Installation

### Prerequisites
- Python 3.8 or higher
- Linux/macOS/Windows
- Git

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd Master_India

# Run setup script
python setup.py

# Test the installation
python test_extraction.py
```

### Manual Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (Linux)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils

# Create directories
mkdir -p data/{raw,processed,samples} models/{trained,configs} logs output
```

## Usage

### Basic Usage

#### Extract from Single Image
```bash
python src/extraction/extract.py --input path/to/invoice.jpg --output results.json
```

#### Extract from Directory
```bash
python src/extraction/extract.py --input path/to/invoice/directory --output results.json --report
```

#### Using Python API
```python
from src.extraction.extract import InvoiceExtractor

# Initialize extractor
extractor = InvoiceExtractor()

# Extract from single image
result = extractor.extract_from_image("invoice.jpg")
print(result)

# Extract from directory
results = extractor.extract_from_directory("invoice_folder/")
extractor.save_results(results, "output.json")
```

### Advanced Usage

#### Custom Configuration
```python
from src.utils.config_loader import ConfigLoader

# Load custom configuration
config = ConfigLoader("custom_config.yaml").load_config()

# Initialize with custom config
extractor = InvoiceExtractor("custom_config.yaml")
```

#### Training Custom Models
```python
from src.model.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train field-specific models
results = trainer.train_field_specific_models()

# Train with transfer learning
from src.model.train import ScalableModelTrainer
scalable_trainer = ScalableModelTrainer()
model = scalable_trainer.train_with_transfer_learning("base_model.pt", "new_field")
```

## Configuration

### Main Configuration File (`config.yaml`)

```yaml
# Data Configuration
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  max_image_size: [1024, 1024]
  supported_formats: ["jpg", "jpeg", "png", "tiff"]

# Model Configuration
model:
  architecture: "transformer"
  backbone: "microsoft/layoutlm-base-uncased"
  learning_rate: 2e-5
  batch_size: 8
  num_epochs: 10

# OCR Configuration
ocr:
  engine: "paddleocr"  # Options: tesseract, paddleocr, easyocr
  language: "en"
  confidence_threshold: 0.7

# Field Extraction Configuration
fields:
  invoice_number:
    patterns: ["INV-\\d+", "INVOICE\\s*#\\s*\\d+"]
    required: true
    confidence_threshold: 0.8
```

### Adding New Fields

To add a new field for extraction:

1. **Update Configuration**:
```yaml
fields:
  new_field:
    patterns: ["PATTERN1", "PATTERN2"]
    required: true
    confidence_threshold: 0.8
```

2. **Add Extraction Logic**:
```python
# In src/data_processing/preprocess.py
def extract_new_field(self, text: str) -> Optional[str]:
    patterns = self.field_configs['new_field']['patterns']
    # Add your extraction logic here
    return extracted_value
```

3. **Train Model** (optional):
```python
# The system will automatically train models for new fields
trainer = ModelTrainer()
results = trainer.train_field_specific_models()
```

## Data Sources

### Supported Sources
- **Kaggle Datasets**: Invoice and document understanding datasets
- **Hugging Face**: Pre-trained models and datasets
- **GitHub**: Open-source invoice datasets
- **PDF Conversion**: Convert PDF invoices to images

### Adding Custom Data Sources

```python
from src.data_processing.download_datasets import DatasetDownloader

class CustomDatasetDownloader(DatasetDownloader):
    def download_custom_source(self):
        # Add your custom download logic
        pass

# Update configuration
config['data_sources']['custom'] = {
    'datasets': ['your_dataset_name'],
    'api_key_required': False
}
```

## Model Architecture

### Supported Models
- **LayoutLM**: Microsoft's layout-aware language model
- **BERT**: Bidirectional Encoder Representations from Transformers
- **Custom Models**: PyTorch-based custom architectures

### Training Pipeline

1. **Data Preprocessing**:
   - Image enhancement and normalization
   - OCR text extraction
   - Field annotation and labeling

2. **Model Training**:
   - Transfer learning from pre-trained models
   - Multi-task learning for multiple fields
   - Hyperparameter optimization

3. **Evaluation**:
   - Accuracy, precision, recall, F1-score
   - Confidence scoring
   - Cross-validation

### Scalable Training Features

#### Transfer Learning
```python
from src.model.train import ScalableModelTrainer

trainer = ScalableModelTrainer()
model = trainer.train_with_transfer_learning(
    base_model_path="models/trained/base_model.pt",
    target_field="new_field"
)
```

#### Multi-Task Learning
```python
# Train a single model for multiple fields
model = trainer.train_multi_task_model([
    'invoice_number', 'invoice_date', 'total_amount'
])
```

## OCR Engines

### Supported Engines

#### Tesseract
- **Pros**: Fast, lightweight, good for clean text
- **Cons**: Struggles with complex layouts
- **Best for**: Simple, well-formatted invoices

#### PaddleOCR
- **Pros**: High accuracy, good with complex layouts
- **Cons**: Slower, larger model size
- **Best for**: Complex, multi-column invoices

#### EasyOCR
- **Pros**: Easy to use, good accuracy
- **Cons**: Moderate speed
- **Best for**: General-purpose OCR

### Switching OCR Engines
```yaml
# In config.yaml
ocr:
  engine: "paddleocr"  # Change to "tesseract" or "easyocr"
```

## Output Format

### JSON Output Structure
```json
{
  "image_path": "path/to/invoice.jpg",
  "processing_timestamp": "2024-01-01T12:00:00",
  "extracted_fields": {
    "invoice_number": {
      "value": "INV-12345",
      "method": "model_based",
      "confidence": 0.95
    },
    "invoice_date": {
      "value": "2024-01-01",
      "method": "rule_based",
      "confidence": 0.88
    },
    "line_items": {
      "value": [
        {
          "description": "Product A",
          "quantity": 2,
          "price": 25.00
        }
      ],
      "method": "rule_based",
      "confidence": 0.75
    },
    "total_amount": {
      "value": 50.00,
      "method": "model_based",
      "confidence": 0.92
    }
  },
  "key_value_pairs": [
    {
      "key": "Invoice Number",
      "value": "INV-12345",
      "confidence": 0.95,
      "bbox": [100, 50, 200, 80]
    }
  ],
  "confidence_scores": {
    "invoice_number": 0.95,
    "invoice_date": 0.88,
    "total_amount": 0.92,
    "overall": 0.88
  }
}
```

## Performance Optimization

### GPU Acceleration
```yaml
# Enable GPU for training and inference
training:
  device: "cuda"  # Use "cpu" for CPU-only

ocr:
  enable_gpu: true
```

### Batch Processing
```python
# Process multiple images efficiently
extractor = InvoiceExtractor()
results = extractor.extract_from_directory("invoice_folder/")
```

### Memory Optimization
```yaml
# Reduce memory usage
model:
  batch_size: 4  # Reduce for limited memory
  max_length: 256  # Reduce sequence length
```

## Troubleshooting

### Common Issues

#### OCR Not Working
```bash
# Check Tesseract installation
tesseract --version

# Install missing dependencies
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

#### Model Training Fails
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Use CPU if GPU not available
config['training']['device'] = 'cpu'
```

#### Memory Issues
```yaml
# Reduce batch size and model size
model:
  batch_size: 2
  max_length: 128
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python src/extraction/extract.py --input invoice.jpg --verbose
```

## API Reference

### Core Classes

#### InvoiceExtractor
Main class for invoice information extraction.

```python
class InvoiceExtractor:
    def __init__(self, config_path: str = "config.yaml")
    def extract_from_image(self, image_path: str) -> Dict[str, Any]
    def extract_from_directory(self, directory_path: str) -> List[Dict[str, Any]]
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None)
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]
```

#### ModelTrainer
Class for training field extraction models.

```python
class ModelTrainer:
    def __init__(self, config_path: str = "config.yaml")
    def train_field_specific_models(self) -> Dict[str, Any]
    def evaluate_model(self, model: FieldExtractionModel, test_data: List[Dict[str, Any]]) -> Dict[str, Any]
```

#### DataPreprocessor
Class for preprocessing invoice images and data.

```python
class DataPreprocessor:
    def __init__(self, config_path: str = "config.yaml")
    def preprocess_dataset(self, input_dir: str = None)
    def create_training_data(self)
```

## Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd Master_India

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black src/
```

### Adding New Features

1. **Create Feature Branch**:
```bash
git checkout -b feature/new-feature
```

2. **Implement Feature**:
   - Add new modules in `src/`
   - Update configuration in `config.yaml`
   - Add tests in `tests/`

3. **Submit Pull Request**:
   - Include documentation
   - Add tests
   - Update README if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

## Acknowledgments

- Microsoft LayoutLM for document understanding
- PaddleOCR for OCR capabilities
- Hugging Face for transformer models
- OpenCV for image processing 