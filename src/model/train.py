#!/usr/bin/env python3
"""
Model Training for Invoice Extraction System

This module handles training of models for invoice field extraction:
- Transformer-based models (LayoutLM, BERT)
- Custom field extraction models
- Scalable training pipeline
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, LayoutLMForSequenceClassification,
    LayoutLMConfig, TrainingArguments, Trainer
)
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import random
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger, LoggerMixin

logger = setup_logger(__name__)

class InvoiceDataset(Dataset):
    """Dataset for invoice field extraction"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        """Initialize dataset"""
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

class FieldExtractionModel(nn.Module):
    """Custom model for field extraction"""
    
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        """Initialize model"""
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained model
        if 'layoutlm' in model_name.lower():
            self.backbone = LayoutLMForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes
            )
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass"""
        if 'layoutlm' in self.model_name.lower():
            return self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                return {'loss': loss, 'logits': logits}
            else:
                return {'logits': logits}

class ModelTrainer(LoggerMixin):
    """Main model trainer class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model trainer"""
        super().__init__()
        self.config = ConfigLoader(config_path).load_config()
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        
        # Set device
        self.device = torch.device(self.training_config['device'] if torch.cuda.is_available() else 'cpu')
        self.log_info(f"Using device: {self.device}")
        
        # Create output directories
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs_dir = Path("models/configs")
        self.configs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load training and validation data"""
        try:
            processed_dir = Path(self.config['data']['processed_dir'])
            training_file = processed_dir / "training_data.json"
            
            if not training_file.exists():
                raise FileNotFoundError(f"Training data not found: {training_file}")
            
            with open(training_file, 'r') as f:
                data = json.load(f)
            
            # Split data
            random.shuffle(data)
            split_idx = int(len(data) * self.config['data']['train_split'])
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            self.log_info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
            return train_data, val_data
            
        except Exception as e:
            self.log_error(f"Error loading training data: {e}")
            raise
    
    def create_model(self, field_name: str = None) -> FieldExtractionModel:
        """Create model for field extraction"""
        try:
            model_name = self.model_config['backbone']
            num_classes = self.model_config['num_classes']
            dropout = self.model_config['dropout']
            
            model = FieldExtractionModel(model_name, num_classes, dropout)
            model.to(self.device)
            
            self.log_info(f"Created model: {model_name} for field: {field_name}")
            return model
            
        except Exception as e:
            self.log_error(f"Error creating model: {e}")
            raise
    
    def create_tokenizer(self) -> AutoTokenizer:
        """Create tokenizer"""
        try:
            model_name = self.model_config['backbone']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add special tokens if needed
            special_tokens = ['[INVOICE_NUMBER]', '[INVOICE_DATE]', '[LINE_ITEM]', '[TOTAL]']
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            
            self.log_info(f"Created tokenizer: {model_name}")
            return tokenizer
            
        except Exception as e:
            self.log_error(f"Error creating tokenizer: {e}")
            raise
    
    def train_model(self, model: FieldExtractionModel, train_data: List[Dict[str, Any]], 
                   val_data: List[Dict[str, Any]], field_name: str = "general") -> Dict[str, Any]:
        """Train the model"""
        try:
            # Create tokenizer
            tokenizer = self.create_tokenizer()
            
            # Create datasets
            train_dataset = InvoiceDataset(train_data, tokenizer, self.model_config['max_length'])
            val_dataset = InvoiceDataset(val_data, tokenizer, self.model_config['max_length'])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True,
                num_workers=self.training_config['num_workers']
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                num_workers=self.training_config['num_workers']
            )
            
            # Setup optimizer and scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.model_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
            
            # Training loop
            num_epochs = self.model_config['num_epochs']
            best_val_loss = float('inf')
            training_history = []
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        
                        outputs = model(**batch)
                        loss = outputs['loss']
                        logits = outputs['logits']
                        
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        predictions = torch.argmax(logits, dim=1)
                        val_correct += (predictions == batch['labels']).sum().item()
                        val_total += batch['labels'].size(0)
                
                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy
                })
                
                self.log_info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model(model, tokenizer, field_name, epoch + 1, avg_val_loss)
            
            return {
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'final_val_accuracy': val_accuracy
            }
            
        except Exception as e:
            self.log_error(f"Error training model: {e}")
            raise
    
    def save_model(self, model: FieldExtractionModel, tokenizer: AutoTokenizer, 
                   field_name: str, epoch: int, val_loss: float) -> None:
        """Save trained model"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{field_name}_model_epoch_{epoch}_loss_{val_loss:.4f}_{timestamp}"
            
            # Save model
            model_path = self.models_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save tokenizer
            tokenizer_path = self.models_dir / f"{model_name}_tokenizer"
            tokenizer.save_pretrained(tokenizer_path)
            
            # Save model config
            config_path = self.configs_dir / f"{model_name}_config.json"
            model_config = {
                'field_name': field_name,
                'epoch': epoch,
                'val_loss': val_loss,
                'timestamp': timestamp,
                'model_path': str(model_path),
                'tokenizer_path': str(tokenizer_path)
            }
            
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            self.log_info(f"Saved model: {model_name}")
            
        except Exception as e:
            self.log_error(f"Error saving model: {e}")
    
    def train_field_specific_models(self) -> Dict[str, Any]:
        """Train models for specific fields"""
        try:
            # Load training data
            train_data, val_data = self.load_training_data()
            
            # Get field configurations
            fields = self.config['fields']
            training_results = {}
            
            for field_name, field_config in fields.items():
                if field_config.get('required', False):
                    self.log_info(f"Training model for field: {field_name}")
                    
                    # Create model for this field
                    model = self.create_model(field_name)
                    
                    # Filter data for this field
                    field_train_data = [item for item in train_data if item.get('field') == field_name]
                    field_val_data = [item for item in val_data if item.get('field') == field_name]
                    
                    if field_train_data and field_val_data:
                        # Train model
                        result = self.train_model(model, field_train_data, field_val_data, field_name)
                        training_results[field_name] = result
                        
                        self.log_info(f"Completed training for {field_name}")
                    else:
                        self.log_warning(f"Insufficient data for field: {field_name}")
            
            # Save training summary
            summary_path = self.models_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(training_results, f, indent=2)
            
            self.log_info("Completed training for all fields")
            return training_results
            
        except Exception as e:
            self.log_error(f"Error training field-specific models: {e}")
            raise
    
    def evaluate_model(self, model: FieldExtractionModel, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            tokenizer = self.create_tokenizer()
            test_dataset = InvoiceDataset(test_data, tokenizer, self.model_config['max_length'])
            test_loader = DataLoader(test_dataset, batch_size=self.training_config['batch_size'])
            
            model.eval()
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating model"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = model(**batch)
                    logits = outputs['logits']
                    
                    preds = torch.argmax(logits, dim=1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(batch['labels'].cpu().numpy())
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            self.log_info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.log_error(f"Error evaluating model: {e}")
            raise

class ScalableModelTrainer(ModelTrainer):
    """Scalable model trainer with advanced features"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize scalable trainer"""
        super().__init__(config_path)
    
    def train_with_transfer_learning(self, base_model_path: str, target_field: str) -> FieldExtractionModel:
        """Train model using transfer learning"""
        try:
            # Load base model
            base_model = self.create_model()
            base_model.load_state_dict(torch.load(base_model_path))
            
            # Create new model for target field
            target_model = self.create_model(target_field)
            
            # Copy weights from base model (except classifier)
            for name, param in base_model.named_parameters():
                if 'classifier' not in name:
                    target_model.get_parameter(name).data.copy_(param.data)
            
            self.log_info(f"Applied transfer learning from {base_model_path} to {target_field}")
            return target_model
            
        except Exception as e:
            self.log_error(f"Error in transfer learning: {e}")
            raise
    
    def train_multi_task_model(self, fields: List[str]) -> FieldExtractionModel:
        """Train a multi-task model for multiple fields"""
        try:
            # Create multi-task model
            model = self.create_model("multi_task")
            
            # Load data for all fields
            train_data, val_data = self.load_training_data()
            
            # Combine data from all fields
            combined_train_data = []
            combined_val_data = []
            
            for field in fields:
                field_train = [item for item in train_data if item.get('field') == field]
                field_val = [item for item in val_data if item.get('field') == field]
                
                combined_train_data.extend(field_train)
                combined_val_data.extend(field_val)
            
            # Train multi-task model
            result = self.train_model(model, combined_train_data, combined_val_data, "multi_task")
            
            self.log_info(f"Trained multi-task model for fields: {fields}")
            return model
            
        except Exception as e:
            self.log_error(f"Error training multi-task model: {e}")
            raise

def main():
    """Main function to run training"""
    trainer = ModelTrainer()
    results = trainer.train_field_specific_models()
    
    # Print summary
    print("\nTraining Summary:")
    for field_name, result in results.items():
        print(f"{field_name}: Best Val Loss: {result['best_val_loss']:.4f}, "
              f"Final Accuracy: {result['final_val_accuracy']:.4f}")

if __name__ == "__main__":
    main() 