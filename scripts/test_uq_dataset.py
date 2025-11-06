#!/usr/bin/env python3
"""
Test on UQ Dataset - ARP Spoofing Detection

This script tests Random Forest and all hybrid/ensemble models on the UQ dataset.
Tests include:
- Random Forest (supervised)
- Isolation Forest (unsupervised)
- One-Class SVM (unsupervised)
- Local Outlier Factor (unsupervised)
- Hybrid ensembles (RF + each unsupervised model)
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Dict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score)
from sklearn.utils import resample

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detector import ARPSpoofingDetector
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from utils import setup_logging, load_config, print_header, save_metrics

logger = logging.getLogger(__name__)


def balance_dataset(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Balance dataset using undersampling of majority class.
    
    Args:
        X: Feature dataframe
        y: Target labels
        
    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    # Combine X and y for resampling
    df = X.copy()
    df['Label'] = y
    
    # Separate majority and minority classes
    df_majority = df[df['Label'] == 0]
    df_minority = df[df['Label'] == 1]
    
    n_majority = len(df_majority)
    n_minority = len(df_minority)
    
    logger.info(f"Original distribution - Normal: {n_majority}, Attack: {n_minority}")
    
    # Undersample majority class to match minority
    df_majority_sampled = resample(df_majority,
                                   replace=False,
                                   n_samples=n_minority,
                                   random_state=42)
    
    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_sampled, df_minority])
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split back into X and y
    y_balanced = df_balanced['Label'].values
    X_balanced = df_balanced.drop(columns=['Label'])
    
    logger.info(f"Balanced distribution - Normal: {n_minority}, Attack: {n_minority}")
    
    # Save balanced dataset to CSV
    try:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        balanced_dir = project_root / 'data' / 'balanced'
        balanced_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = balanced_dir / 'UQ_MITM_ARP_test_balanced_50_50.csv'
        df_balanced.to_csv(output_path, index=False)
        logger.info(f"✓ Saved balanced dataset to: {output_path}")
        print(f"✓ Saved balanced dataset: {output_path.name}")
    except Exception as e:
        logger.warning(f"Could not save balanced dataset: {e}")
    
    return X_balanced, y_balanced


def print_confusion_matrix(cm: np.ndarray, model_name: str):
    """Print confusion matrix in a formatted way."""
    if cm.shape != (2, 2):
        print(f"⚠ Unexpected confusion matrix shape: {cm.shape}")
        return
    
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX - {model_name}")
    print(f"{'='*60}")
    print(f"                 Predicted Negative | Predicted Positive")
    print(f"                 -------------------|-------------------")
    print(f"Actual Negative |        TN: {tn:4d}    |      FP: {fp:4d}")
    print(f"Actual Positive |        FN: {fn:4d}    |      TP: {tp:4d}")
    print(f"{'='*60}")
    print(f"True Positives  (TP): {tp:4d} - Correctly identified attacks")
    print(f"True Negatives  (TN): {tn:4d} - Correctly identified normal")
    print(f"False Positives (FP): {fp:4d} - Normal flagged as attack")
    print(f"False Negatives (FN): {fn:4d} - Attack missed (CRITICAL!)")
    print(f"{'='*60}\n")


def test_random_forest(detector: ARPSpoofingDetector, X_test: pd.DataFrame, 
                      y_test: np.ndarray, threshold: float = 0.4) -> Dict:
    """Test Random Forest model."""
    print_header("Testing Random Forest", width=60, char='-')
    
    # Get predictions
    y_proba = detector.model.predict_proba(detector.scaler.transform(X_test))[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'model_name': 'Random Forest',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': cm
    }
    
    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print_confusion_matrix(cm, 'Random Forest')
    
    return metrics


def test_unsupervised_model(model, model_name: str, scaler, 
                           X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
    """Test an unsupervised model (Isolation Forest, One-Class SVM, LOF)."""
    print_header(f"Testing {model_name}", width=60, char='-')
    
    # Scale data
    X_scaled = scaler.transform(X_test)
    
    # Get predictions (-1 for anomaly/attack, 1 for normal)
    y_pred_raw = model.predict(X_scaled)
    # Convert to 0/1 (0 = normal, 1 = attack)
    y_pred = (y_pred_raw == -1).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': cm
    }
    
    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print_confusion_matrix(cm, model_name)
    
    return metrics


def test_hybrid_ensemble(rf_model, rf_scaler, unsup_model, unsup_name: str,
                        X_test: pd.DataFrame, y_test: np.ndarray, 
                        threshold: float = 0.4, logic: str = 'OR') -> Dict:
    """Test hybrid ensemble with OR/AND logic."""
    model_name = f"Hybrid ({unsup_name} + RF) - {logic}"
    print_header(f"Testing {model_name}", width=60, char='-')
    
    # Get RF predictions
    X_scaled = rf_scaler.transform(X_test)
    rf_proba = rf_model.predict_proba(X_scaled)[:, 1]
    rf_pred = (rf_proba >= threshold).astype(int)
    
    # Get unsupervised predictions
    unsup_pred_raw = unsup_model.predict(X_scaled)
    unsup_pred = (unsup_pred_raw == -1).astype(int)
    
    # Combine predictions
    if logic == 'OR':
        y_pred = ((rf_pred == 1) | (unsup_pred == 1)).astype(int)
    else:  # AND
        y_pred = ((rf_pred == 1) & (unsup_pred == 1)).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': cm
    }
    
    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print_confusion_matrix(cm, model_name)
    
    return metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Test on UQ dataset with all models (balanced & imbalanced)')
    parser.add_argument('--threshold', type=float, default=0.40,
                       help='Decision threshold for Random Forest (default: 0.40)')
    parser.add_argument('--skip-balanced', action='store_true',
                       help='Skip balanced testing (only test imbalanced)')
    parser.add_argument('--skip-imbalanced', action='store_true',
                       help='Skip imbalanced testing (only test balanced)')
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'config.yaml'
    config = load_config(str(config_path))
    
    # Setup logging
    log_file = str(project_root / config['output']['logs_path'] / 'arp_detection.log')
    setup_logging(log_file=log_file)
    
    print_header("TESTING ON UQ DATASET - ALL MODELS", width=60, char='=')
    
    # Paths
    data_dir = project_root.parent / 'dataset'
    models_dir = project_root / 'models' / 'saved_models'
    output_dir = project_root / 'outputs'
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== STEP 1: LOAD MODELS =====
    print_header("STEP 1: LOADING TRAINED MODELS", width=60, char='-')
    
    # Load Random Forest detector
    detector_path = models_dir / 'arp_spoofing_detector.pkl'
    if not detector_path.exists():
        print(f"✗ Detector not found: {detector_path}")
        return 1
    
    detector = ARPSpoofingDetector.load(detector_path)
    rf_model = detector.model
    scaler = detector.scaler
    print(f"✓ Loaded Random Forest")
    
    # Load unsupervised models
    unsupervised_models = {}
    
    # Isolation Forest
    iso_path = models_dir / 'isolation_forest.pkl'
    if iso_path.exists():
        with open(iso_path, 'rb') as f:
            iso_package = pickle.load(f)
            # Extract model from package if it's a dict, otherwise use as-is
            if isinstance(iso_package, dict):
                unsupervised_models['Isolation Forest'] = iso_package['model']
            else:
                unsupervised_models['Isolation Forest'] = iso_package
        print(f"✓ Loaded Isolation Forest")
    
    # One-Class SVM
    svm_path = models_dir / 'one-class_svm.pkl'
    if svm_path.exists():
        with open(svm_path, 'rb') as f:
            svm_package = pickle.load(f)
            # Extract model from package if it's a dict, otherwise use as-is
            if isinstance(svm_package, dict):
                unsupervised_models['One-Class SVM'] = svm_package['model']
            else:
                unsupervised_models['One-Class SVM'] = svm_package
        print(f"✓ Loaded One-Class SVM")
    
    # Local Outlier Factor
    lof_path = models_dir / 'local_outlier_factor.pkl'
    if lof_path.exists():
        with open(lof_path, 'rb') as f:
            lof_package = pickle.load(f)
            # Extract model from package if it's a dict, otherwise use as-is
            if isinstance(lof_package, dict):
                unsupervised_models['Local Outlier Factor'] = lof_package['model']
            else:
                unsupervised_models['Local Outlier Factor'] = lof_package
        print(f"✓ Loaded Local Outlier Factor")
    
    print(f"\n✓ Loaded {1 + len(unsupervised_models)} models total")
    
    # ===== STEP 2: LOAD UQ DATASET =====
    print_header("STEP 2: LOADING UQ DATASET", width=60, char='-')
    
    loader = DataLoader(data_dir=str(data_dir))
    engineer = FeatureEngineer()
    
    try:
        df_uq = loader.load_all_datasets(
            filenames=['UQ_MITM_ARP_labeled_data.csv'],
            select_best=False,
            balance_classes=False
        )
        print(f"✓ Loaded UQ dataset: {len(df_uq):,} samples")
    except Exception as e:
        print(f"✗ Failed to load UQ dataset: {str(e)}")
        return 1
    
    # ===== STEP 3: PREPARE DATASET =====
    print_header("STEP 3: PREPARING DATASET", width=60, char='-')
    
    # Clean and engineer features
    df_clean = engineer.clean_data(df_uq)
    df_features = engineer.create_derived_features(df_clean)
    
    # Extract labels and features
    # Encode labels: 'arp_spoofing' -> 1, 'normal' -> 0
    label_map = {
        'arp_spoofing': 1, 
        'normal': 0,
        'Attacker': 1, 
        'Normal': 0, 
        'MITM-ArpSpoofing': 1
    }
    y_test = df_features['Label'].map(label_map).values
    
    # Select only the features that the model was trained on
    X_test = df_features[detector.feature_names]
    
    # Count attacks and normal
    n_attacks = np.sum(y_test == 1)
    n_normal = np.sum(y_test == 0)
    
    print(f"✓ Prepared UQ dataset:")
    print(f"  Total samples: {len(X_test):,}")
    print(f"  Attacks: {n_attacks} ({n_attacks/len(y_test)*100:.1f}%)")
    print(f"  Normal: {n_normal} ({n_normal/len(y_test)*100:.1f}%)")
    print(f"  Features: {X_test.shape[1]}")
    
    # Determine which tests to run
    run_imbalanced = not args.skip_imbalanced
    run_balanced = not args.skip_balanced
    
    # ===== STEP 4: TEST ON IMBALANCED DATASET =====
    if run_imbalanced:
        print_header("STEP 4A: TESTING ON IMBALANCED DATASET", width=60, char='=')
        
        all_results_imbalanced = {}
        
        # Test Random Forest
        rf_results = test_random_forest(detector, X_test, y_test, threshold=args.threshold)
        all_results_imbalanced['Random Forest'] = rf_results
        
        # Test each unsupervised model
        for model_name, model in unsupervised_models.items():
            results = test_unsupervised_model(model, model_name, scaler, X_test, y_test)
            all_results_imbalanced[model_name] = results
        
        # Test hybrid ensembles
        print_header("Testing Hybrid Ensembles (Imbalanced)", width=60, char='-')
        for model_name, model in unsupervised_models.items():
            results = test_hybrid_ensemble(
                rf_model, scaler, model, model_name, 
                X_test, y_test, threshold=args.threshold, logic='OR'
            )
            all_results_imbalanced[results['model_name']] = results
    
    # ===== STEP 4B: TEST ON BALANCED DATASET =====
    if run_balanced:
        print_header("STEP 4B: PREPARING BALANCED DATASET", width=60, char='=')
        
        # Balance the dataset
        X_test_balanced, y_test_balanced = balance_dataset(X_test, y_test)
        
        n_attacks_bal = np.sum(y_test_balanced == 1)
        n_normal_bal = np.sum(y_test_balanced == 0)
        
        print(f"✓ Balanced UQ dataset:")
        print(f"  Total samples: {len(X_test_balanced):,}")
        print(f"  Attacks: {n_attacks_bal} ({n_attacks_bal/len(y_test_balanced)*100:.1f}%)")
        print(f"  Normal: {n_normal_bal} ({n_normal_bal/len(y_test_balanced)*100:.1f}%)")
        
        print_header("STEP 4C: TESTING ON BALANCED DATASET", width=60, char='=')
        
        all_results_balanced = {}
        
        # Test Random Forest
        rf_results = test_random_forest(detector, X_test_balanced, y_test_balanced, threshold=args.threshold)
        all_results_balanced['Random Forest'] = rf_results
        
        # Test each unsupervised model
        for model_name, model in unsupervised_models.items():
            results = test_unsupervised_model(model, model_name, scaler, X_test_balanced, y_test_balanced)
            all_results_balanced[model_name] = results
        
        # Test hybrid ensembles
        print_header("Testing Hybrid Ensembles (Balanced)", width=60, char='-')
        for model_name, model in unsupervised_models.items():
            results = test_hybrid_ensemble(
                rf_model, scaler, model, model_name, 
                X_test_balanced, y_test_balanced, threshold=args.threshold, logic='OR'
            )
            all_results_balanced[results['model_name']] = results
    
    # ===== STEP 5: SUMMARY =====
    print_header("STEP 5: RESULTS SUMMARY", width=60, char='=')
    
    # Print imbalanced results
    if run_imbalanced:
        print("\n" + "="*80)
        print("IMBALANCED DATASET RESULTS")
        print("="*80)
        print(f"{'Model':<45} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
        print("="*80)
        
        for model_name, metrics in all_results_imbalanced.items():
            print(f"{model_name:<45} "
                  f"{metrics['accuracy']:<8.2%} "
                  f"{metrics['precision']:<8.2%} "
                  f"{metrics['recall']:<8.2%} "
                  f"{metrics['f1_score']:<8.2%}")
        
        print("="*80)
        
        # Find best models for imbalanced
        print("\nBEST PERFORMERS (IMBALANCED):")
        print("-"*60)
        
        best_acc = max(all_results_imbalanced.keys(), key=lambda m: all_results_imbalanced[m]['accuracy'])
        print(f"  Best Accuracy:  {best_acc} ({all_results_imbalanced[best_acc]['accuracy']:.2%})")
        
        best_prec = max(all_results_imbalanced.keys(), key=lambda m: all_results_imbalanced[m]['precision'])
        print(f"  Best Precision: {best_prec} ({all_results_imbalanced[best_prec]['precision']:.2%})")
        
        best_rec = max(all_results_imbalanced.keys(), key=lambda m: all_results_imbalanced[m]['recall'])
        print(f"  Best Recall:    {best_rec} ({all_results_imbalanced[best_rec]['recall']:.2%})")
        
        best_f1 = max(all_results_imbalanced.keys(), key=lambda m: all_results_imbalanced[m]['f1_score'])
        print(f"  Best F1-Score:  {best_f1} ({all_results_imbalanced[best_f1]['f1_score']:.2%})")
        
        print("-"*60)
    
    # Print balanced results
    if run_balanced:
        print("\n" + "="*80)
        print("BALANCED DATASET RESULTS")
        print("="*80)
        print(f"{'Model':<45} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
        print("="*80)
        
        for model_name, metrics in all_results_balanced.items():
            print(f"{model_name:<45} "
                  f"{metrics['accuracy']:<8.2%} "
                  f"{metrics['precision']:<8.2%} "
                  f"{metrics['recall']:<8.2%} "
                  f"{metrics['f1_score']:<8.2%}")
        
        print("="*80)
        
        # Find best models for balanced
        print("\nBEST PERFORMERS (BALANCED):")
        print("-"*60)
        
        best_acc = max(all_results_balanced.keys(), key=lambda m: all_results_balanced[m]['accuracy'])
        print(f"  Best Accuracy:  {best_acc} ({all_results_balanced[best_acc]['accuracy']:.2%})")
        
        best_prec = max(all_results_balanced.keys(), key=lambda m: all_results_balanced[m]['precision'])
        print(f"  Best Precision: {best_prec} ({all_results_balanced[best_prec]['precision']:.2%})")
        
        best_rec = max(all_results_balanced.keys(), key=lambda m: all_results_balanced[m]['recall'])
        print(f"  Best Recall:    {best_rec} ({all_results_balanced[best_rec]['recall']:.2%})")
        
        best_f1 = max(all_results_balanced.keys(), key=lambda m: all_results_balanced[m]['f1_score'])
        print(f"  Best F1-Score:  {best_f1} ({all_results_balanced[best_f1]['f1_score']:.2%})")
        
        print("-"*60)
    
    # ===== STEP 6: SAVE RESULTS =====
    print_header("STEP 6: SAVING RESULTS", width=60, char='-')
    
    # Save imbalanced results
    if run_imbalanced:
        results_serializable = {}
        for model_name, metrics in all_results_imbalanced.items():
            results_serializable[model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
                if k != 'confusion_matrix'
            }
        
        results_path = reports_dir / 'uq_dataset_results_imbalanced.json'
        save_metrics(results_serializable, str(results_path))
        print(f"✓ Saved imbalanced results: {results_path}")
    
    # Save balanced results
    if run_balanced:
        results_serializable = {}
        for model_name, metrics in all_results_balanced.items():
            results_serializable[model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
                if k != 'confusion_matrix'
            }
        
        results_path = reports_dir / 'uq_dataset_results_balanced.json'
        save_metrics(results_serializable, str(results_path))
        print(f"✓ Saved balanced results: {results_path}")
    
    print_header("TESTING COMPLETE", width=60, char='=')
    
    if run_imbalanced and run_balanced:
        print(f"\n✓ Tested on both balanced and imbalanced UQ dataset")
        print(f"  Imbalanced: {len(X_test):,} packets")
        print(f"  Balanced: {len(X_test_balanced):,} packets")
    elif run_imbalanced:
        print(f"\n✓ Tested {len(all_results_imbalanced)} models on {len(X_test):,} packets (imbalanced)")
    elif run_balanced:
        print(f"\n✓ Tested {len(all_results_balanced)} models on {len(X_test_balanced):,} packets (balanced)")
    
    print(f"  Threshold: {args.threshold}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
