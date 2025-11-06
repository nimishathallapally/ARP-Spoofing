#!/usr/bin/env python3
"""
Test All Models - Including New Unsupervised Models

Tests all supervised and unsupervised models on balanced dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import ARPSpoofingModels
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from utils import load_config, setup_logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # Setup
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    config = load_config(str(config_path))
    setup_logging(config['logging']['log_file'])
    
    print("="*70)
    print(" "*20 + "TESTING ALL MODELS")
    print("="*70)
    
    # Load data
    print("\n1. Loading iot_intrusion dataset...")
    data_dir = Path(__file__).parent.parent.parent / 'dataset'
    loader = DataLoader(data_dir=str(data_dir))
    df = loader.load_dataset(['iot_intrusion_MITM_ARP_labeled_data.csv'])
    print(f"   Loaded {len(df):,} samples")
    
    # Feature engineering
    print("\n2. Feature engineering...")
    engineer = FeatureEngineer()
    df_clean = engineer.clean_data(df)
    df_enhanced = engineer.create_derived_features(df_clean)
    
    # Prepare features
    y = df_enhanced['Label']
    X = df_enhanced.drop(columns=['Label'])
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    y_encoded = engineer.encode_labels(y)
    
    # Balance dataset
    from sklearn.utils import resample
    df_temp = X.copy()
    df_temp['Label'] = y_encoded
    df_majority = df_temp[df_temp['Label'] == 0]
    df_minority = df_temp[df_temp['Label'] == 1]
    
    df_majority_sampled = resample(df_majority, replace=False, 
                                   n_samples=len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority_sampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    y_balanced = df_balanced['Label'].values
    X_balanced = df_balanced.drop(columns=['Label'])
    
    print(f"   Balanced dataset: {len(X_balanced)} samples (50% each class)")
    
    # Train-test split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    print("\n5. Initializing all models...")
    model_trainer = ARPSpoofingModels(random_state=42)
    model_trainer.initialize_models()
    
    # Train supervised models
    print("\n6. Training supervised models...")
    supervised_models = model_trainer.train_supervised_models(X_train_scaled, y_train)
    print(f"   Trained {len(supervised_models)} supervised models")
    
    # Train unsupervised models
    print("\n7. Training unsupervised models...")
    unsupervised_models = model_trainer.train_unsupervised_models(X_train_scaled)
    print(f"   Trained {len(unsupervised_models)} unsupervised models")
    
    # Evaluate all models
    print("\n8. Evaluating all models...")
    print("="*70)
    
    all_results = {}
    
    # Evaluate supervised
    print("\nSUPERVISED MODELS:")
    print("-"*70)
    for name, model in supervised_models.items():
        metrics = model_trainer.evaluate_model(model, X_test_scaled, y_test, name)
        all_results[name] = metrics
    
    # Evaluate unsupervised
    print("\nUNSUPERVISED MODELS:")
    print("-"*70)
    for name, model in unsupervised_models.items():
        metrics = model_trainer.evaluate_model(model, X_test_scaled, y_test, name)
        all_results[name] = metrics
    
    # Create hybrid ensembles
    print("\nHYBRID ENSEMBLES:")
    print("-"*70)
    best_model = supervised_models.get('Random Forest')
    if best_model and unsupervised_models:
        hybrid_results = model_trainer.create_multi_hybrid_ensembles(
            best_model, unsupervised_models, X_test_scaled, y_test
        )
        all_results.update(hybrid_results)
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "SUMMARY")
    print("="*70)
    print(f"{'Model':<40} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*70)
    
    for name, metrics in all_results.items():
        model_name = metrics.get('model_name', name)
        print(f"{model_name:<40} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     "
              f"{metrics['recall']:.4f}     {metrics['f1_score']:.4f}")
    
    # Find best models
    print("\n" + "="*70)
    best_f1 = max(all_results.values(), key=lambda x: x['f1_score'])
    best_recall = max(all_results.values(), key=lambda x: x['recall'])
    best_precision = max(all_results.values(), key=lambda x: x['precision'])
    
    print(f"Best F1-Score:  {best_f1['model_name']} ({best_f1['f1_score']:.4f})")
    print(f"Best Recall:    {best_recall['model_name']} ({best_recall['recall']:.4f})")
    print(f"Best Precision: {best_precision['model_name']} ({best_precision['precision']:.4f})")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
