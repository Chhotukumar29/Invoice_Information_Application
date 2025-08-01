#!/usr/bin/env python3
"""
Logger Setup for Invoice Extraction System

This module provides consistent logging configuration across the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os

def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent configuration
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger
    
    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class LoggerMixin:
    """Mixin class to add logging capabilities to other classes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logger(self.__class__.__name__)
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

def setup_project_logging(config: dict) -> None:
    """
    Set up project-wide logging based on configuration
    
    Args:
        config: Configuration dictionary containing logging settings
    """
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', 'logs/invoice_extraction.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    level = log_config.get('level', 'INFO')
    format_string = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    logging.info("Project logging configured successfully")

class ProgressLogger:
    """Logger with progress bar capabilities"""
    
    def __init__(self, name: str = "ProgressLogger"):
        self.logger = setup_logger(name)
        self.progress_bars = {}
    
    def log_progress(self, message: str, current: int, total: int, bar_id: str = "default") -> None:
        """Log progress with percentage"""
        percentage = (current / total) * 100
        self.logger.info(f"{message}: {current}/{total} ({percentage:.1f}%)")
    
    def start_progress(self, message: str, total: int, bar_id: str = "default") -> None:
        """Start a progress tracking session"""
        self.logger.info(f"Starting: {message} (Total: {total})")
        self.progress_bars[bar_id] = {"total": total, "current": 0, "message": message}
    
    def update_progress(self, increment: int = 1, bar_id: str = "default") -> None:
        """Update progress for a specific bar"""
        if bar_id in self.progress_bars:
            self.progress_bars[bar_id]["current"] += increment
            current = self.progress_bars[bar_id]["current"]
            total = self.progress_bars[bar_id]["total"]
            message = self.progress_bars[bar_id]["message"]
            self.log_progress(message, current, total, bar_id)
    
    def finish_progress(self, bar_id: str = "default") -> None:
        """Finish a progress tracking session"""
        if bar_id in self.progress_bars:
            message = self.progress_bars[bar_id]["message"]
            self.logger.info(f"Completed: {message}")
            del self.progress_bars[bar_id] 