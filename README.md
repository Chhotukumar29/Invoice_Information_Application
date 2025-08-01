# Advanced Invoice Extraction System

A comprehensive Streamlit application for extracting invoice information with advanced ML features, model training, evaluation, and data visualization.

## Features

- **Multi-format image support** (PNG, JPG, TIFF, BMP)
- **Advanced OCR processing** with multiple engines (Gemini API, Tesseract, EasyOCR, PaddleOCR)
- **Confidence scoring** for extracted fields
- **Data visualization** and analytics
- **Model training simulation**
- **Performance dashboard**
- **Clear records functionality**
- **Extract key information**:
  - Invoice number and date
  - Sender and receiver details
  - Email addresses and phone numbers
  - Addresses and transaction IDs
  - Line items with quantities and prices
  - Total amounts and tax information

## Environment Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root with your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your actual API key
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Get Your Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key and paste it in your `.env` file

### 4. Run the Application
```bash
streamlit run app.py
```

## Security Notes
- Never commit your `.env` file to version control
- The `.env.example` file is provided as a template
- Your API key is automatically loaded from environment variables

## Usage

1. **Upload invoice images** using the file uploader
2. **Click "Extract Information"** to process the images
3. **View extracted data** in clean tables
4. **Download results** as JSON file

## Extracted Information

The app extracts the following from your invoices:

- **Invoice Details**: Number, date, total amount
- **Contact Information**: Sender/Receiver names, emails, phones, addresses
- **Financial Data**: Transaction IDs, tax rates, line items
- **Items & Quantities**: Product descriptions, quantities, unit prices, totals

## File Structure

```
Master_India/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── invoice_results.json   # Extracted results (auto-generated)
└── README.md             # This file
```

## Requirements

- Python 3.8+
- Streamlit
- Pillow (PIL)
- Pandas
- OpenCV

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`   
