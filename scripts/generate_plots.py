#!/usr/bin/env python3
"""
Generate All Visualizations

This script generates all EDA and model evaluation plots.
"""

import sys
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import ARPSpoofingModels
from detector import ARPSpoofingDetector
from visualizer import Visualizer
from utils import setup_logging, load_config, print_header

logger = logging.getLogger(__name__)


def main(config_path: str = None):
    """
    Generate all visualizations.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    config = load_config(str(config_path))
    
    # Setup logging
    setup_logging(level='INFO')
    
    print_header("GENERATING VISUALIZATIONS FOR ARP SPOOFING DETECTION", width=70)
    
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
        
        # Train unsupervised model
        iso_forest = model_trainer.train_unsupervised_model(X_train)
        
        logger.info(f"✓ Trained {len(supervised_models) + 1} models")
    except Exception as e:
        logger.error(f"✗ Model training failed: {str(e)}")
        return 1
    
    # ===== STEP 4: EVALUATE MODELS =====
    print_header("STEP 4: MODEL EVALUATION", width=70, char='-')
    
    try:
        comparison_df = model_trainer.evaluate_all_models(X_test, y_test)
        logger.info(f"✓ Evaluated {len(comparison_df)} models")
    except Exception as e:
        logger.error(f"✗ Model evaluation failed: {str(e)}")
        return 1
    
    # ===== STEP 5: SELECT BEST MODEL =====
    print_header("STEP 5: BEST MODEL SELECTION", width=70, char='-')
    
    try:
        best_model_name, best_model = model_trainer.select_best_model(
            comparison_df,
            weights=config['model_selection']['weights']
        )
        logger.info(f"✓ Selected: {best_model_name}")
    except Exception as e:
        logger.error(f"✗ Model selection failed: {str(e)}")
        return 1
    
    # ===== STEP 6: REAL-TIME SIMULATION =====
    print_header("STEP 6: REAL-TIME SIMULATION", width=70, char='-')
    
    try:
        detector = ARPSpoofingDetector(
            model=best_model,
            scaler=engineer.scaler,  # Use the same scaler from training
            feature_names=feature_names,
            model_name=best_model_name
        )
        
        # Run simulation
        sim_y_true, sim_y_pred, detection_results = detector.simulate_realtime(
            X_test, y_test, 
            n_packets=100,
            random_state=42
        )
        
        logger.info(f"✓ Completed simulation with 100 packets")
    except Exception as e:
        logger.error(f"✗ Simulation failed: {str(e)}")
        detection_results = None
    
    # ===== STEP 7: GENERATE ALL PLOTS =====
    print_header("STEP 7: GENERATING VISUALIZATIONS", width=70, char='-')
    
    try:
        visualizer = Visualizer(output_dir=config['output']['plots_path'])
        
        visualizer.generate_all_plots(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            models=model_trainer.models,
            comparison_df=comparison_df,
            detection_results=detection_results
        )
        
        logger.info("✓ All visualizations generated successfully!")
    except Exception as e:
        logger.error(f"✗ Visualization generation failed: {str(e)}")
        return 1
    
    # ===== SUMMARY =====
    print_header("VISUALIZATION GENERATION COMPLETE", width=70)
    
    plots_dir = Path(config['output']['plots_path'])
    plot_files = list(plots_dir.glob('*.png'))
    
    print(f"\n✓ Generated {len(plot_files)} visualization(s):")
    for plot_file in sorted(plot_files):
        print(f"  - {plot_file.name}")
    
    print(f"\n✓ All plots saved to: {plots_dir}")
    print(f"✓ Plots are ready for inclusion in PROJECT_DELIVERABLES.md\n")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate All Visualizations')
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
        print("\n\nVisualization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)
