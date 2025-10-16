#!/usr/bin/env python3
"""
Evaluate Model Script
=====================
Evaluates a saved ARP spoofing detector model on test data.

Usage:
    python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl
    python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl --data data/raw/test_data.csv
    python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl --config config/config.yaml

Author: ARP Spoofing Detection Team
Date: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

from src.detector import ARPSpoofingDetector
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.utils import setup_logging, load_config, save_json

# Configure logging
logger = logging.getLogger(__name__)


def print_header(text: str, width: int = 70, char: str = '='):
    """Print a formatted header."""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def print_metrics_table(metrics: dict):
    """Print metrics in a formatted table."""
    print("\nPerformance Metrics:")
    print("-" * 40)
    print(f"{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Accuracy':<20} {metrics['accuracy']:>14.2%}")
    print(f"{'Precision':<20} {metrics['precision']:>14.2%}")
    print(f"{'Recall':<20} {metrics['recall']:>14.2%}")
    print(f"{'F1-Score':<20} {metrics['f1_score']:>14.2%}")
    if 'roc_auc' in metrics:
        print(f"{'ROC AUC':<20} {metrics['roc_auc']:>14.4f}")
    print("-" * 40)


def print_confusion_matrix(cm: np.ndarray, labels: list = None):
    """Print confusion matrix in a formatted way."""
    if labels is None:
        labels = ['Normal', 'Attack']
    
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"{'':>15} {'Predicted':^20}")
    print(f"{'':>15} {labels[0]:>10} {labels[1]:>10}")
    print("-" * 40)
    print(f"{'Actual':<8} {labels[0]:>6} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"{'':>8} {labels[1]:>6} {cm[1][0]:>10} {cm[1][1]:>10}")
    print("-" * 40)


def evaluate_on_test_data(detector: ARPSpoofingDetector, 
                          X_test: np.ndarray, 
                          y_test: np.ndarray) -> dict:
    """
    Evaluate detector on test data.
    
    Args:
        detector: Trained detector
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    predictions = []
    probabilities = []
    
    for i in range(len(X_test)):
        result = detector.detect(X_test[i])
        predictions.append(1 if result['label'] == 'arp_spoofing' else 0)
        probabilities.append(result['probability'])
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'classification_report': classification_report(
            y_test, predictions, 
            target_names=['normal', 'arp_spoofing'],
            output_dict=True
        )
    }
    
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate ARP Spoofing Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate saved model (uses datasets from config)
  python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl
  
  # Evaluate with specific test data file
  python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl --data data/raw/test.csv
  
  # Evaluate with custom config
  python scripts/evaluate_model.py --model models/saved_models/arp_spoofing_detector.pkl --config config/eval_config.yaml
        """
    )
    
    parser.add_argument('--model', '-m', 
                       type=str,
                       default='models/saved_models/arp_spoofing_detector.pkl',
                       help='Path to saved detector model (default: models/saved_models/arp_spoofing_detector.pkl)')
    
    parser.add_argument('--data', '-d',
                       type=str,
                       default=None,
                       help='Path to specific test data CSV file (optional)')
    
    parser.add_argument('--config', '-c',
                       type=str,
                       default='config/config.yaml',
                       help='Path to configuration file (default: config/config.yaml)')
    
    parser.add_argument('--output', '-o',
                       type=str,
                       default=None,
                       help='Path to save evaluation results JSON (optional)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print_header("ARP SPOOFING DETECTION - MODEL EVALUATION")
    
    # ===== STEP 1: LOAD CONFIGURATION =====
    print_header("STEP 1: LOAD CONFIGURATION", width=70, char='-')
    
    try:
        config = load_config(args.config)
        logger.info(f"✓ Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {str(e)}")
        return 1
    
    # ===== STEP 2: LOAD MODEL =====
    print_header("STEP 2: LOAD MODEL", width=70, char='-')
    
    try:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        
        detector = ARPSpoofingDetector.load(args.model)
        logger.info(f"✓ Loaded model from: {args.model}")
        logger.info(f"  Model type: {detector.model_name}")
        logger.info(f"  Features: {len(detector.feature_names)}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        return 1
    
    # ===== STEP 3: LOAD TEST DATA =====
    print_header("STEP 3: LOAD TEST DATA", width=70, char='-')
    
    try:
        if args.data:
            # Load specific test file
            logger.info(f"Loading test data from: {args.data}")
            df = pd.read_csv(args.data)
            logger.info(f"  Loaded {len(df)} samples with {len(df.columns)} columns")
        else:
            # Load and combine datasets from config
            logger.info("Loading datasets from config...")
            data_dir = config['data']['raw_data_path']
            data_loader = DataLoader(data_dir)
            df = data_loader.load_all_datasets(
                filenames=config['data']['dataset_files'],
                select_best=config['data'].get('select_best_datasets', False),
                top_n=config['data'].get('top_n_datasets', 3),
                balance_classes=config['data'].get('balance_classes', True)
            )
            logger.info(f"✓ Loaded combined dataset: {df.shape}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load test data: {str(e)}")
        return 1
    
    # ===== STEP 4: PREPARE DATA =====
    print_header("STEP 4: PREPARE DATA", width=70, char='-')
    
    try:
        # Prepare features using the same pipeline
        engineer = FeatureEngineer()
        
        # Check if 'label' column exists (case-insensitive)
        label_col = None
        for col in df.columns:
            if col.lower() == 'label':
                label_col = col
                break
        
        if label_col is None:
            logger.error("✗ Test data must have 'label' column")
            return 1
        
        # Prepare data with same settings as training
        X_train, X_test, y_train, y_test, selected_features = engineer.prepare_data(
            df,
            target_column=label_col,
            feature_selection_method=config['features']['selection_method'],
            n_features=config['features']['n_features'],
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state']
        )
        
        # For evaluation, we'll use the test set
        X_prepared = X_test
        y = y_test
        
        # Ensure features match what the detector expects
        if set(selected_features) != set(detector.feature_names):
            logger.warning("Feature mismatch detected. Features may not align perfectly.")
            logger.warning(f"  Model expects: {len(detector.feature_names)} features")
            logger.warning(f"  Data provides: {len(selected_features)} features")
        
        logger.info(f"✓ Prepared data: {X_prepared.shape}")
        logger.info(f"✓ Test samples: {len(y)}")
        logger.info(f"✓ Features: {len(selected_features)}")
        
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {str(e)}")
        return 1
    
    # ===== STEP 5: EVALUATE MODEL =====
    print_header("STEP 5: EVALUATE MODEL", width=70, char='-')
    
    try:
        metrics = evaluate_on_test_data(detector, X_prepared, y)
        logger.info("✓ Evaluation complete")
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {str(e)}")
        return 1
    
    # ===== STEP 6: DISPLAY RESULTS =====
    print_header("EVALUATION RESULTS")
    
    # Print metrics table
    print_metrics_table(metrics)
    
    # Print confusion matrix
    print_confusion_matrix(np.array(metrics['confusion_matrix']))
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 60)
    report = metrics['classification_report']
    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 60)
    for class_name in ['normal', 'arp_spoofing']:
        if class_name in report:
            print(f"{class_name:<20} {report[class_name]['precision']:>12.4f} "
                  f"{report[class_name]['recall']:>12.4f} {report[class_name]['f1-score']:>12.4f}")
    print("-" * 60)
    
    # ===== STEP 7: SAVE RESULTS =====
    if args.output:
        print_header("STEP 7: SAVE RESULTS", width=70, char='-')
        
        try:
            # Add metadata
            results = {
                'timestamp': datetime.now().isoformat(),
                'model_path': args.model,
                'model_type': detector.model_name,
                'test_data_path': args.data if args.data else 'combined_datasets',
                'test_samples': len(y),
                'features': len(detector.feature_names),
                'metrics': {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1_score']),
                    'roc_auc': float(metrics['roc_auc'])
                },
                'confusion_matrix': metrics['confusion_matrix'],
                'classification_report': metrics['classification_report']
            }
            
            save_json(results, args.output)
            logger.info(f"✓ Results saved to: {args.output}")
            
        except Exception as e:
            logger.error(f"✗ Failed to save results: {str(e)}")
            return 1
    
    # ===== SUMMARY =====
    print_header("EVALUATION COMPLETE")
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 47)
    print(f"{'Model':<30} {detector.model_name:>15}")
    print(f"{'Test Samples':<30} {len(y):>15,}")
    print(f"{'Features':<30} {len(detector.feature_names):>15}")
    print(f"{'Accuracy':<30} {metrics['accuracy']:>14.2%}")
    print(f"{'Precision':<30} {metrics['precision']:>14.2%}")
    print(f"{'Recall':<30} {metrics['recall']:>14.2%}")
    print(f"{'F1-Score':<30} {metrics['f1_score']:>14.2%}")
    print(f"{'ROC AUC':<30} {metrics['roc_auc']:>15.4f}")
    print("-" * 47)
    
    if args.output:
        print(f"\n✓ Results saved to: {args.output}")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
