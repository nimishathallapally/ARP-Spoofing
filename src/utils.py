"""
Utility functions for the ARP spoofing detection system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import yaml

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (None = console only)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")


def save_metrics(metrics: Dict, output_path: str):
    """
    Save performance metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save metrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    metrics_serializable = convert_types(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")


def load_metrics(metrics_path: str) -> Dict:
    """
    Load performance metrics from JSON file.
    
    Args:
        metrics_path: Path to metrics file
        
    Returns:
        Metrics dictionary
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def print_header(title: str, width: int = 60, char: str = '='):
    """
    Print a formatted header.
    
    Args:
        title: Header title
        width: Total width
        char: Character to use for border
    """
    print('\n' + char * width)
    print(title.center(width))
    print(char * width)


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.
    
    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """
    Format a number with fixed decimals.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{value:.{decimals}f}"


def create_directory_structure(base_path: str):
    """
    Create project directory structure.
    
    Args:
        base_path: Base project path
    """
    base_path = Path(base_path)
    
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'outputs/plots',
        'outputs/logs',
        'outputs/reports',
        'config',
        'scripts',
        'src',
        'tests',
        'docs',
        'notebooks'
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure at {base_path}")


def get_color_code(alert_level: str) -> str:
    """
    Get ANSI color code for alert level.
    
    Args:
        alert_level: Alert level (SAFE, MEDIUM, HIGH, CRITICAL)
        
    Returns:
        ANSI color code
    """
    colors = {
        'SAFE': '\033[92m',      # Green
        'MEDIUM': '\033[93m',    # Yellow
        'HIGH': '\033[91m',      # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    return colors.get(alert_level, colors['RESET'])


def print_colored(text: str, alert_level: str):
    """
    Print colored text based on alert level.
    
    Args:
        text: Text to print
        alert_level: Alert level for color
    """
    color = get_color_code(alert_level)
    reset = get_color_code('RESET')
    print(f"{color}{text}{reset}")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary of class -> weight
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
    return weights


def balance_dataset(X: np.ndarray, y: np.ndarray, method: str = 'undersample') -> tuple:
    """
    Balance dataset using undersampling or oversampling.
    
    Args:
        X: Features
        y: Labels
        method: 'undersample' or 'oversample'
        
    Returns:
        Tuple of (balanced_X, balanced_y)
    """
    unique, counts = np.unique(y, return_counts=True)
    
    if method == 'undersample':
        # Undersample to minority class size
        min_count = counts.min()
        balanced_X = []
        balanced_y = []
        
        for cls in unique:
            cls_indices = np.where(y == cls)[0]
            selected_indices = np.random.choice(cls_indices, size=min_count, replace=False)
            balanced_X.append(X[selected_indices])
            balanced_y.append(y[selected_indices])
        
        balanced_X = np.vstack(balanced_X)
        balanced_y = np.concatenate(balanced_y)
        
    elif method == 'oversample':
        # Oversample to majority class size
        max_count = counts.max()
        balanced_X = []
        balanced_y = []
        
        for cls in unique:
            cls_indices = np.where(y == cls)[0]
            selected_indices = np.random.choice(cls_indices, size=max_count, replace=True)
            balanced_X.append(X[selected_indices])
            balanced_y.append(y[selected_indices])
        
        balanced_X = np.vstack(balanced_X)
        balanced_y = np.concatenate(balanced_y)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Shuffle
    shuffle_indices = np.random.permutation(len(balanced_y))
    balanced_X = balanced_X[shuffle_indices]
    balanced_y = balanced_y[shuffle_indices]
    
    return balanced_X, balanced_y


def save_json(data: Dict, output_path: str):
    """
    Save dictionary data to JSON file.
    Alias for save_metrics with generic naming.
    
    Args:
        data: Dictionary to save
        output_path: Path to save JSON file
    """
    save_metrics(data, output_path)


if __name__ == "__main__":
    # Test utilities
    setup_logging(level='INFO')
    logger.info("Utilities module loaded successfully")
    
    # Test color printing
    for level in ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']:
        print_colored(f"This is a {level} alert", level)
