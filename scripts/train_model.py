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
        
        # Train all unsupervised models
        unsupervised_models = model_trainer.train_unsupervised_models(X_train)
        logger.info(f"✓ Trained {len(unsupervised_models)} unsupervised models:")
        for model_name in unsupervised_models.keys():
            logger.info(f"   - {model_name}")
        
        # Keep individual references for compatibility
        iso_forest = unsupervised_models.get('Isolation Forest')
        one_class_svm = unsupervised_models.get('One-Class SVM')
        lof = unsupervised_models.get('Local Outlier Factor')
        dbscan = unsupervised_models.get('DBSCAN')
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
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
    print_header("STEP 4C: HYBRID ENSEMBLE MODELS", width=70, char='-')
    
    try:
        logger.info("Creating multiple hybrid ensemble combinations...")
        
        # Get the best supervised model (typically Random Forest)
        rf_model = supervised_models.get('Random Forest')
        if rf_model is None:
            rf_model = list(supervised_models.values())[0]  # Fallback to first model
        
        # Get all hybrid ensemble predictions
        hybrid_ensembles = model_trainer.create_multi_hybrid_ensembles(
            best_model=rf_model,
            unsupervised_models=unsupervised_models,
            X_test=X_test,
            y_test=y_test
        )
        
        logger.info(f"✓ Created {len(hybrid_ensembles)} hybrid ensemble models")
        
        # Visualize and store each hybrid ensemble
        visualizer = Visualizer(output_dir=config['output']['plots_path'])
        
        for ensemble_name, metrics in hybrid_ensembles.items():
            # Store metrics
            model_trainer.performance_metrics[metrics['model_name']] = metrics
            
            # Add to comparison dataframe
            hybrid_row = pd.DataFrame([{
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC AUC': metrics.get('roc_auc', 0.0),
                'TPR': metrics.get('TPR', 0),
                'FPR': metrics.get('FPR', 0),
                'FNR': metrics.get('FNR', 0),
                'TNR': metrics.get('TNR', 0)
            }])
            test_comparison_df = pd.concat([test_comparison_df, hybrid_row], ignore_index=True)
        
        logger.info(f"✓ {len(hybrid_ensembles)} hybrid ensemble models evaluated")
        logger.info("✓ Hybrid Ensemble created and evaluated")
        
        # Get Random Forest and Isolation Forest
        rf_model = supervised_models.get('Random Forest')
        if_model = unsupervised_models.get('Isolation Forest')
        
        if rf_model and if_model:
            # Create weighted ensemble
            weighted_pred, weighted_metrics = model_trainer.create_hybrid_ensemble(
                supervised_models={'Random Forest': rf_model},
                iso_forest=if_model,
                X_test=X_test,
                y_test=y_test,
                rf_weight=0.7,
                iso_weight=0.3
            )
            
            # Store metrics
            weighted_name = 'Weighted Hybrid (RF:0.7, IF:0.3)'
            model_trainer.performance_metrics[weighted_name] = weighted_metrics
            
            # Generate confusion matrix
            logger.info(f"Generating confusion matrix for {weighted_name}...")
            visualizer.plot_confusion_matrix(
                y_true=y_test,
                y_pred=weighted_pred,
                model_name=weighted_name
            )
            
            # Add to comparison dataframe
            weighted_row = pd.DataFrame([{
                'Model': weighted_name,
                'Accuracy': weighted_metrics['accuracy'],
                'Precision': weighted_metrics['precision'],
                'Recall': weighted_metrics['recall'],
                'F1-Score': weighted_metrics['f1_score'],
                'ROC AUC': weighted_metrics.get('roc_auc', 0.0),
                'TPR': weighted_metrics.get('TPR', 0),
                'FPR': weighted_metrics.get('FPR', 0),
                'FNR': weighted_metrics.get('FNR', 0),
                'TNR': weighted_metrics.get('TNR', 0)
            }])
            test_comparison_df = pd.concat([test_comparison_df, weighted_row], ignore_index=True)
            
            logger.info(f"✓ Weighted Hybrid (RF:0.7, IF:0.3) created")
            logger.info(f"   Accuracy: {weighted_metrics['accuracy']:.4f}")
            logger.info(f"   Precision: {weighted_metrics['precision']:.4f}")
            logger.info(f"   Recall: {weighted_metrics['recall']:.4f}")
            logger.info(f"   F1-Score: {weighted_metrics['f1_score']:.4f}")
        else:
            logger.warning("⚠ Could not create weighted hybrid: RF or IF model missing")
        
    except Exception as e:
        logger.warning(f"⚠ Hybrid ensemble creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.info("Continuing without hybrid ensemble models...")
    
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

    # ===== STEP 5B: SAVE ALL MODELS =====
    print_header("STEP 5B: SAVING ALL TRAINED MODELS", width=70, char='-')
    
    try:
        models_dir = Path(config['output']['models_path'])
        # Pass scaler and feature_names to save models in proper format
        model_trainer.save_all_models(
            output_dir=str(models_dir),
            scaler=engineer.scaler,
            feature_names=feature_names
        )
        logger.info(f"✓ All models saved in Flask-ready format to: {models_dir}")
        logger.info(f"  Each model includes: model, scaler, feature_names, metadata")
    except Exception as e:
        logger.error(f"✗ Failed to save all models: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ===== STEP 5C: GENERATE COMPREHENSIVE VISUALIZATIONS =====
    print_header("STEP 5C: COMPREHENSIVE VISUALIZATIONS", width=70, char='-')
    
    try:
        visualizer = Visualizer(output_dir=config['output']['plots_path'])
        
        # Plot all confusion matrices in grid
        logger.info("Generating comprehensive confusion matrix grid...")
        visualizer.plot_all_confusion_matrices(
            performance_metrics=model_trainer.performance_metrics,
            y_test=y_test
        )
        
        # Plot all ROC curves
        logger.info("Generating comprehensive ROC curves plot...")
        visualizer.plot_all_roc_curves(
            performance_metrics=model_trainer.performance_metrics,
            y_test=y_test
        )
        
        logger.info("✓ Comprehensive visualizations generated")
    except Exception as e:
        logger.error(f"✗ Visualization generation failed: {str(e)}")
    
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
    
    # ===== STEP 7: SAVE DETECTOR & METRICS =====
    print_header("STEP 7: SAVING DETECTOR & METRICS", width=70, char='-')
    
    try:
        # Save detector
        model_path = models_dir / 'arp_spoofing_detector.pkl'
        detector.save(str(model_path))
        logger.info(f"✓ Model saved to: {model_path}")
        
        # Get train and test metrics for best model
        train_metrics = model_trainer.performance_metrics.get(f"{best_model_name}_train", {})
        test_metrics = model_trainer.performance_metrics.get(best_model_name, {})
        
        # Collect all hybrid ensemble metrics
        hybrid_ensemble_metrics = {}
        for model_name, metrics in model_trainer.performance_metrics.items():
            # Check if this is a hybrid model (contains "Hybrid" in name)
            if 'Hybrid' in model_name and not model_name.endswith('_train'):
                hybrid_ensemble_metrics[model_name] = {
                    'accuracy': float(metrics.get('accuracy', 0)),
                    'precision': float(metrics.get('precision', 0)),
                    'recall': float(metrics.get('recall', 0)),
                    'f1_score': float(metrics.get('f1_score', 0)),
                    'roc_auc': float(metrics.get('roc_auc', 0)) if metrics.get('roc_auc') else None,
                    'TPR': float(metrics.get('TPR', 0)),
                    'FPR': float(metrics.get('FPR', 0)),
                    'FNR': float(metrics.get('FNR', 0)),
                    'TNR': float(metrics.get('TNR', 0)),
                    'TP': int(metrics.get('TP', 0)),
                    'TN': int(metrics.get('TN', 0)),
                    'FP': int(metrics.get('FP', 0)),
                    'FN': int(metrics.get('FN', 0)),
                    'confusion_matrix': metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() 
                        if hasattr(metrics.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') 
                        else metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                }
        
        logger.info(f"✓ Collected {len(hybrid_ensemble_metrics)} hybrid ensemble models")
        
        # Prepare all models metrics with extended info
        all_models_metrics = {}
        for model_name, metrics in model_trainer.performance_metrics.items():
            if not model_name.endswith('_train'):
                all_models_metrics[model_name] = {
                    'accuracy': float(metrics.get('accuracy', 0)),
                    'precision': float(metrics.get('precision', 0)),
                    'recall': float(metrics.get('recall', 0)),
                    'f1_score': float(metrics.get('f1_score', 0)),
                    'roc_auc': float(metrics.get('roc_auc', 0)) if metrics.get('roc_auc') else None,
                    'TPR': float(metrics.get('TPR', 0)),
                    'FPR': float(metrics.get('FPR', 0)),
                    'FNR': float(metrics.get('FNR', 0)),
                    'TNR': float(metrics.get('TNR', 0)),
                    'TP': int(metrics.get('TP', 0)),
                    'TN': int(metrics.get('TN', 0)),
                    'FP': int(metrics.get('FP', 0)),
                    'FN': int(metrics.get('FN', 0)),
                    'confusion_matrix': metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() 
                        if hasattr(metrics.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') 
                        else metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                }
        
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
            
            # All models metrics with extended info
            'all_models_metrics': all_models_metrics,
            
            # Training Set Metrics
            'train_metrics': {
                'accuracy': float(train_metrics.get('accuracy', 0)),
                'precision': float(train_metrics.get('precision', 0)),
                'recall': float(train_metrics.get('recall', 0)),
                'f1_score': float(train_metrics.get('f1_score', 0)),
                'roc_auc': float(train_metrics.get('roc_auc', 0)),
                'TPR': float(train_metrics.get('TPR', 0)),
                'FPR': float(train_metrics.get('FPR', 0)),
                'FNR': float(train_metrics.get('FNR', 0)),
                'TNR': float(train_metrics.get('TNR', 0)),
                'confusion_matrix': train_metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() if hasattr(train_metrics.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') else train_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            } if train_metrics else None,
            
            # Test Set Metrics
            'test_metrics': {
                'accuracy': float(test_metrics.get('accuracy', 0)),
                'precision': float(test_metrics.get('precision', 0)),
                'recall': float(test_metrics.get('recall', 0)),
                'f1_score': float(test_metrics.get('f1_score', 0)),
                'roc_auc': float(test_metrics.get('roc_auc', 0)),
                'TPR': float(test_metrics.get('TPR', 0)),
                'FPR': float(test_metrics.get('FPR', 0)),
                'FNR': float(test_metrics.get('FNR', 0)),
                'TNR': float(test_metrics.get('TNR', 0)),
                'confusion_matrix': test_metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() if hasattr(test_metrics.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') else test_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            },
            
            # Hybrid Ensemble Metrics - All hybrid models
            'hybrid_ensemble_metrics': hybrid_ensemble_metrics if hybrid_ensemble_metrics else None,
            
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
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 47)
    print(f"{'Best Model':<30} {best_model_name:>15}")
    print(f"{'Test Accuracy':<30} {test_metrics.get('accuracy', 0):>14.2%}")
    print(f"{'Test Precision':<30} {test_metrics.get('precision', 0):>14.2%}")
    print(f"{'Test Recall (TPR)':<30} {test_metrics.get('recall', 0):>14.2%}")
    print(f"{'Test F1-Score':<30} {test_metrics.get('f1_score', 0):>14.2%}")
    print(f"{'Test FPR':<30} {test_metrics.get('FPR', 0):>14.2%}")
    print(f"{'Test TNR (Specificity)':<30} {test_metrics.get('TNR', 0):>14.2%}")
    print(f"{'Test FNR (Miss Rate)':<30} {test_metrics.get('FNR', 0):>14.2%}")
    
    # Show hybrid ensemble performance if available
    if hybrid_ensemble_metrics:
        print("-" * 47)
        print(f"{'Hybrid Ensemble Models':<30} {len(hybrid_ensemble_metrics):>15}")
        # Show best hybrid based on F1-score
        best_hybrid = max(hybrid_ensemble_metrics.items(), key=lambda x: x[1].get('f1_score', 0))
        print(f"{'Best Hybrid Model':<30} {best_hybrid[0][:15]:>15}")
        print(f"{'  F1-Score':<30} {best_hybrid[1].get('f1_score', 0):>14.2%}")
        print(f"{'  Recall (TPR)':<30} {best_hybrid[1].get('recall', 0):>14.2%}")
        print(f"{'  Precision':<30} {best_hybrid[1].get('precision', 0):>14.2%}")
    
    print("-" * 47)
    print(f"{'Training Samples':<30} {len(X_train):>15,}")
    print(f"{'Test Samples':<30} {len(X_test):>15,}")
    print(f"{'Features':<30} {len(feature_names):>15}")
    print(f"{'Models Trained':<30} {len(model_trainer.models):>15}")
    print("-" * 47)
    print(f"\n✓ Best model detector: {model_path}")
    print(f"✓ All models saved to: {models_dir}")
    print(f"✓ Comprehensive metrics with TPR/FPR/TNR/FNR: {metrics_path}")
    print(f"✓ All confusion matrices grid: {config['output']['plots_path']}/all_confusion_matrices.png")
    print(f"✓ All ROC curves: {config['output']['plots_path']}/all_roc_curves.png")
    if hybrid_ensemble_metrics:
        print(f"✓ Hybrid Ensemble models: {len(hybrid_ensemble_metrics)} combinations saved")
    print("\n✓ Run 'python scripts/detect_realtime.py' for real-time detection demo\n")
    
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
