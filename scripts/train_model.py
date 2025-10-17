#!/usr/bin/env python3
"""
Train ARP Spoofing Detection Model

This script:
1. Loads and combines multiple datasets
2. Performs feature engineering
3. Trains multiple ML models (supervised + unsupervised)
4. Evaluates and selects the best model
5. Saves the trained model for deployment
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import ARPSpoofingModels
from detector import ARPSpoofingDetector
from visualizer import Visualizer
from utils import setup_logging, load_config, save_metrics, print_header

logger = logging.getLogger(__name__)


def main(config_path: str = None):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    config = load_config(str(config_path))
    
    # Setup logging
    setup_logging(
        log_file=config['logging']['log_file'] if config['logging']['log_to_file'] else None,
        level=config['logging']['level']
    )
    
    print_header("ARP SPOOFING DETECTION - MODEL TRAINING", width=70)
    
    # ===== STEP 1: LOAD DATA =====
    print_header("STEP 1: DATA LOADING", width=70, char='-')
    
    loader = DataLoader(data_dir=config['data']['raw_data_path'])
    
    try:
        combined_df = loader.load_all_datasets(
            filenames=config['data']['dataset_files'],
            select_best=config['data']['select_best_datasets'],
            top_n=config['data']['top_n_datasets'],
            balance_classes=config['data']['balance_classes']
        )
        logger.info(f"✓ Loaded combined dataset: {combined_df.shape}")
    except Exception as e:
        logger.error(f"✗ Failed to load datasets: {str(e)}")
        return 1
    
    # ===== STEP 2: FEATURE ENGINEERING =====
    print_header("STEP 2: FEATURE ENGINEERING", width=70, char='-')
    
    engineer = FeatureEngineer()
    
    try:
        X_train, X_test, y_train, y_test, feature_names = engineer.prepare_data(
            combined_df,
            feature_selection_method=config['features']['selection_method'],
            n_features=config['features']['n_features'],
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state']
        )
        logger.info(f"✓ Training set: {X_train.shape}")
        logger.info(f"✓ Test set: {X_test.shape}")
        logger.info(f"✓ Selected features: {len(feature_names)}")
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {str(e)}")
        return 1
    
    # ===== STEP 3: TRAIN MODELS =====
    print_header("STEP 3: MODEL TRAINING", width=70, char='-')
    
    model_trainer = ARPSpoofingModels(random_state=config['training']['random_state'])
    model_trainer.initialize_models()
    
    try:
        # Train supervised models
        supervised_models = model_trainer.train_supervised_models(X_train, y_train)
        logger.info(f"✓ Trained {len(supervised_models)} supervised models")
        
        # Train unsupervised model
        iso_forest = model_trainer.train_unsupervised_model(X_train)
        logger.info(f"✓ Trained unsupervised model (Isolation Forest)")
    except Exception as e:
        logger.error(f"✗ Model training failed: {str(e)}")
        return 1
    
    # ===== STEP 4: EVALUATE MODELS ON TRAIN SET =====
    print_header("STEP 4A: MODEL EVALUATION - TRAINING SET", width=70, char='-')
    
    try:
        logger.info("Evaluating models on training data...")
        train_comparison_df = model_trainer.evaluate_all_models(X_train, y_train, dataset_type='train')
        logger.info(f"✓ Evaluated {len(train_comparison_df)} models on training set")
    except Exception as e:
        logger.error(f"✗ Training set evaluation failed: {str(e)}")
        return 1
    
    # ===== STEP 4B: EVALUATE MODELS ON TEST SET =====
    print_header("STEP 4B: MODEL EVALUATION - TEST SET", width=70, char='-')
    
    try:
        logger.info("Evaluating models on test data...")
        test_comparison_df = model_trainer.evaluate_all_models(X_test, y_test, dataset_type='test')
        logger.info(f"✓ Evaluated {len(test_comparison_df)} models on test set")
    except Exception as e:
        logger.error(f"✗ Test set evaluation failed: {str(e)}")
        return 1
    
    # ===== STEP 4C: CREATE AND EVALUATE HYBRID ENSEMBLE =====
    print_header("STEP 4C: HYBRID ENSEMBLE MODEL", width=70, char='-')
    
    try:
        logger.info("Creating hybrid ensemble (supervised + unsupervised)...")
        hybrid_predictions, hybrid_metrics = model_trainer.create_hybrid_ensemble(
            supervised_models=supervised_models,
            iso_forest=iso_forest,
            X_test=X_test,
            y_test=y_test
        )
        
        # Store hybrid ensemble metrics
        model_trainer.performance_metrics['Hybrid Ensemble'] = hybrid_metrics
        
        # Generate and save confusion matrix for hybrid ensemble
        visualizer = Visualizer(output_dir=config['output']['plots_path'])
        
        logger.info("Generating confusion matrix for Hybrid Ensemble...")
        visualizer.plot_confusion_matrix(
            y_true=y_test,
            y_pred=hybrid_predictions,
            model_name='Hybrid Ensemble',
            title='Hybrid Ensemble (Supervised + Unsupervised)'
        )
        logger.info(f"✓ Hybrid Ensemble confusion matrix saved to {config['output']['plots_path']}")
        
        # Add hybrid ensemble to comparison dataframe
        hybrid_row = pd.DataFrame([{
            'Model': 'Hybrid Ensemble',
            'Accuracy': hybrid_metrics['accuracy'],
            'Precision': hybrid_metrics['precision'],
            'Recall': hybrid_metrics['recall'],
            'F1-Score': hybrid_metrics['f1_score'],
            'ROC AUC': hybrid_metrics.get('roc_auc', 0.0)
        }])
        test_comparison_df = pd.concat([test_comparison_df, hybrid_row], ignore_index=True)
        
        logger.info(f"✓ Hybrid Ensemble created and evaluated")
        logger.info(f"   Accuracy: {hybrid_metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {hybrid_metrics['precision']:.4f}")
        logger.info(f"   Recall: {hybrid_metrics['recall']:.4f}")
        logger.info(f"   F1-Score: {hybrid_metrics['f1_score']:.4f}")
        
    except Exception as e:
        logger.error(f"✗ Hybrid ensemble creation failed: {str(e)}")
        logger.warning("Continuing without hybrid ensemble...")
    
    # ===== STEP 5: SELECT BEST MODEL =====
    print_header("STEP 5: MODEL SELECTION", width=70, char='-')
    
    try:
        best_model_name, best_model = model_trainer.select_best_model(
            test_comparison_df,  # Use test set for model selection
            weights=config['model_selection']['weights']
        )
        logger.info(f"✓ Selected best model: {best_model_name}")
    except Exception as e:
        logger.error(f"✗ Model selection failed: {str(e)}")
        return 1
    
    # ===== STEP 6: CREATE DETECTOR =====
    print_header("STEP 6: DETECTOR CREATION", width=70, char='-')
    
    try:
        detector = ARPSpoofingDetector(
            model=best_model,
            scaler=engineer.scaler,  # Pass the SAME scaler used during training
            feature_names=feature_names,
            model_name=best_model_name,
            alert_thresholds={
                level: tuple(thresholds) 
                for level, thresholds in config['alert_thresholds'].items()
            }
        )
        logger.info(f"✓ Created detector with {best_model_name}")
    except Exception as e:
        logger.error(f"✗ Detector creation failed: {str(e)}")
        return 1
    
    # ===== STEP 7: SAVE MODEL =====
    print_header("STEP 7: SAVING MODEL", width=70, char='-')
    
    try:
        # Create output directory
        models_dir = Path(config['output']['models_path'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detector
        model_path = models_dir / 'arp_spoofing_detector.pkl'
        detector.save(str(model_path))
        logger.info(f"✓ Model saved to: {model_path}")
        
        # Get train and test metrics for best model
        train_metrics = model_trainer.performance_metrics.get(f"{best_model_name}_train", {})
        test_metrics = model_trainer.performance_metrics.get(best_model_name, {})
        
        # Get hybrid ensemble metrics if available
        hybrid_metrics = model_trainer.performance_metrics.get('Hybrid Ensemble', None)
        
        # Prepare comprehensive metrics dictionary
        metrics_dict = {
            'model_name': best_model_name,
            'training_data': {
                'samples': int(len(X_train)),
                'features': len(feature_names)
            },
            'test_data': {
                'samples': int(len(X_test)),
                'features': len(feature_names)
            },
            'feature_names': feature_names,
            
            # Training Set Metrics
            'train_metrics': {
                'accuracy': float(train_metrics.get('accuracy', 0)),
                'precision': float(train_metrics.get('precision', 0)),
                'recall': float(train_metrics.get('recall', 0)),
                'f1_score': float(train_metrics.get('f1_score', 0)),
                'roc_auc': float(train_metrics.get('roc_auc', 0)),
                'confusion_matrix': train_metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() if hasattr(train_metrics.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') else train_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            } if train_metrics else None,
            
            # Test Set Metrics
            'test_metrics': {
                'accuracy': float(test_metrics.get('accuracy', 0)),
                'precision': float(test_metrics.get('precision', 0)),
                'recall': float(test_metrics.get('recall', 0)),
                'f1_score': float(test_metrics.get('f1_score', 0)),
                'roc_auc': float(test_metrics.get('roc_auc', 0)),
                'confusion_matrix': test_metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() if hasattr(test_metrics.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') else test_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            },
            
            # Hybrid Ensemble Metrics
            'hybrid_ensemble_metrics': {
                'accuracy': float(hybrid_metrics.get('accuracy', 0)),
                'precision': float(hybrid_metrics.get('precision', 0)),
                'recall': float(hybrid_metrics.get('recall', 0)),
                'f1_score': float(hybrid_metrics.get('f1_score', 0)),
                'confusion_matrix': hybrid_metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() if hasattr(hybrid_metrics.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') else hybrid_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            } if hybrid_metrics else None,
            
            # All Models Comparison
            'all_models_train': train_comparison_df.to_dict('records'),
            'all_models_test': test_comparison_df.to_dict('records')
        }
        
        metrics_path = Path(config['output']['reports_path']) / 'model_metrics.json'
        save_metrics(metrics_dict, str(metrics_path))
        logger.info(f"✓ Metrics saved to: {metrics_path}")
        
    except Exception as e:
        logger.error(f"✗ Failed to save model: {str(e)}")
        return 1
    
    # ===== SUMMARY =====
    print_header("TRAINING COMPLETE", width=70)
    
    # Get test metrics for summary
    test_metrics = model_trainer.performance_metrics.get(best_model_name, {})
    hybrid_metrics = model_trainer.performance_metrics.get('Hybrid Ensemble', None)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 47)
    print(f"{'Best Model':<30} {best_model_name:>15}")
    print(f"{'Test Accuracy':<30} {test_metrics.get('accuracy', 0):>14.2%}")
    print(f"{'Test Precision':<30} {test_metrics.get('precision', 0):>14.2%}")
    print(f"{'Test Recall':<30} {test_metrics.get('recall', 0):>14.2%}")
    print(f"{'Test F1-Score':<30} {test_metrics.get('f1_score', 0):>14.2%}")
    
    # Show hybrid ensemble performance if available
    if hybrid_metrics:
        print("-" * 47)
        print(f"{'Hybrid Ensemble Model':<30}")
        print(f"{'  Accuracy':<30} {hybrid_metrics.get('accuracy', 0):>14.2%}")
        print(f"{'  Precision':<30} {hybrid_metrics.get('precision', 0):>14.2%}")
        print(f"{'  Recall':<30} {hybrid_metrics.get('recall', 0):>14.2%}")
        print(f"{'  F1-Score':<30} {hybrid_metrics.get('f1_score', 0):>14.2%}")
    
    print("-" * 47)
    print(f"{'Training Samples':<30} {len(X_train):>15,}")
    print(f"{'Test Samples':<30} {len(X_test):>15,}")
    print(f"{'Features':<30} {len(feature_names):>15}")
    print("-" * 47)
    print(f"\n✓ Model ready for deployment: {model_path}")
    print(f"✓ Comprehensive metrics (train + test + hybrid): {metrics_path}")
    if hybrid_metrics:
        print(f"✓ Hybrid Ensemble confusion matrix: outputs/plots/confusion_matrix_hybrid_ensemble.png")
    print("✓ Run 'python scripts/detect_realtime.py' for real-time detection demo\n")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ARP Spoofing Detection Model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = main(config_path=args.config)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)
