"""
Machine learning models for ARP spoofing detection.

This module implements:
- Supervised learning models (Random Forest, Gradient Boosting, Neural Network, Logistic Regression)
- Unsupervised learning (Isolation Forest for anomaly detection)
- Hybrid ensemble combining supervised and unsupervised approaches
- Model training, evaluation, and selection
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class ARPSpoofingModels:
    """Collection of ML models for ARP spoofing detection."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.performance_metrics = {}
        self.best_model_name = None
        self.best_model = None
        
    def initialize_models(self) -> Dict:
        """
        Initialize all models with default hyperparameters.
        
        Returns:
            Dictionary of model_name -> model object
        """
        logger.info("Initializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                random_state=self.random_state
            ),
            'Isolation Forest': IsolationForest(
                contamination=0.1,
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        logger.info(f"  Initialized {len(self.models)} models")
        return self.models
    
    def train_supervised_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train all supervised learning models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUPERVISED MODELS")
        logger.info("="*60)
        
        supervised_models = {}
        
        for name, model in self.models.items():
            if name == 'Isolation Forest':
                continue  # Skip unsupervised model
            
            logger.info(f"\nTraining {name}...")
            try:
                model.fit(X_train, y_train)
                supervised_models[name] = model
                logger.info(f"  ✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"  ✗ Error training {name}: {str(e)}")
        
        return supervised_models
    
    def train_unsupervised_model(self, X_train: np.ndarray) -> IsolationForest:
        """
        Train Isolation Forest for anomaly detection.
        
        Args:
            X_train: Training features
            
        Returns:
            Trained Isolation Forest model
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING UNSUPERVISED MODEL (Isolation Forest)")
        logger.info("="*60)
        
        iso_forest = self.models.get('Isolation Forest')
        if iso_forest is None:
            iso_forest = IsolationForest(
                contamination=0.1,
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        logger.info("Training Isolation Forest for anomaly detection...")
        iso_forest.fit(X_train)
        logger.info("  ✓ Isolation Forest trained successfully")
        
        return iso_forest
    
    def evaluate_model(self, 
                      model, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      model_name: str) -> Dict:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Make predictions
        if model_name == 'Isolation Forest':
            # Isolation Forest returns -1 for anomalies, 1 for normal
            y_pred_raw = model.predict(X_test)
            y_pred = np.where(y_pred_raw == -1, 1, 0)  # Convert: -1 -> 1 (attack), 1 -> 0 (normal)
            y_proba = model.score_samples(X_test)  # Anomaly scores
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Normalize to [0, 1]
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            except:
                metrics['roc_auc'] = None
        
        # Log results
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics.get('roc_auc'):
            logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray,
                           dataset_type: str = 'test') -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            dataset_type: Type of dataset ('train' or 'test') for metric storage
            
        Returns:
            DataFrame with performance metrics for all models
        """
        logger.info("\n" + "="*60)
        logger.info(f"EVALUATING ALL MODELS - {dataset_type.upper()} SET")
        logger.info("="*60)
        
        results = []
        
        for name, model in self.models.items():
            if not hasattr(model, 'predict'):
                continue
                
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results.append(metrics)
            
            # Store metrics with dataset identifier
            if dataset_type == 'train':
                self.performance_metrics[f"{name}_train"] = metrics
            else:
                self.performance_metrics[name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': m['model_name'],
                'Accuracy': m['accuracy'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1-Score': m['f1_score'],
                'ROC AUC': m.get('roc_auc', None)
            }
            for m in results
        ])
        
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        logger.info("\n" + "="*60)
        logger.info(f"MODEL COMPARISON - {dataset_type.upper()} SET")
        logger.info("="*60)
        logger.info("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
    
    def select_best_model(self, 
                         comparison_df: pd.DataFrame,
                         weights: Dict[str, float] = None) -> Tuple[str, object]:
        """
        Select the best model based on composite scoring.
        
        Args:
            comparison_df: DataFrame with model comparison
            weights: Dictionary of metric weights (default: f1=0.4, recall=0.3, accuracy=0.2, precision=0.1)
            
        Returns:
            Tuple of (best_model_name, best_model_object)
        """
        if weights is None:
            weights = {
                'f1_score': 0.40,
                'recall': 0.30,
                'accuracy': 0.20,
                'precision': 0.10
            }
        
        # Normalize weight keys to match DataFrame columns
        # Convert underscore format to DataFrame column names
        weight_mapping = {
            'f1_score': 'F1-Score',
            'recall': 'Recall',
            'accuracy': 'Accuracy',
            'precision': 'Precision'
        }
        
        normalized_weights = {}
        for key, value in weights.items():
            normalized_key = weight_mapping.get(key.lower(), key)
            normalized_weights[normalized_key] = value
        
        logger.info("\n" + "="*60)
        logger.info("SELECTING BEST MODEL")
        logger.info("="*60)
        
        # Calculate composite score
        comparison_df['Composite Score'] = (
            comparison_df['F1-Score'] * normalized_weights.get('F1-Score', 0.40) +
            comparison_df['Recall'] * normalized_weights.get('Recall', 0.30) +
            comparison_df['Accuracy'] * normalized_weights.get('Accuracy', 0.20) +
            comparison_df['Precision'] * normalized_weights.get('Precision', 0.10)
        )
        
        comparison_df = comparison_df.sort_values('Composite Score', ascending=False)
        
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        self.best_model_name = best_model_name
        self.best_model = best_model
        
        logger.info(f"\n{'Model':<25} {'Composite Score':>15}")
        logger.info("-" * 42)
        for _, row in comparison_df.iterrows():
            logger.info(f"{row['Model']:<25} {row['Composite Score']:>15.4f}")
        
        logger.info("\n" + "="*60)
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info("="*60)
        logger.info(f"  Accuracy:  {comparison_df.iloc[0]['Accuracy']:.4f}")
        logger.info(f"  Precision: {comparison_df.iloc[0]['Precision']:.4f}")
        logger.info(f"  Recall:    {comparison_df.iloc[0]['Recall']:.4f}")
        logger.info(f"  F1-Score:  {comparison_df.iloc[0]['F1-Score']:.4f}")
        logger.info(f"  Composite: {comparison_df.iloc[0]['Composite Score']:.4f}")
        logger.info("="*60)
        
        return best_model_name, best_model
    
    def create_hybrid_ensemble(self, 
                              supervised_models: Dict,
                              iso_forest: IsolationForest,
                              X_test: np.ndarray,
                              y_test: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Create hybrid ensemble combining supervised and unsupervised predictions.
        
        Args:
            supervised_models: Dictionary of trained supervised models
            iso_forest: Trained Isolation Forest
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (ensemble_predictions, metrics)
        """
        logger.info("\n" + "="*60)
        logger.info("CREATING HYBRID ENSEMBLE")
        logger.info("="*60)
        
        # Get supervised predictions
        supervised_preds = []
        for name, model in supervised_models.items():
            if name == 'Isolation Forest':
                continue
            pred = model.predict(X_test)
            supervised_preds.append(pred)
        
        # Average supervised predictions
        supervised_avg = np.mean(supervised_preds, axis=0)
        supervised_final = (supervised_avg >= 0.5).astype(int)
        
        # Get unsupervised predictions
        iso_pred_raw = iso_forest.predict(X_test)
        iso_pred = np.where(iso_pred_raw == -1, 1, 0)
        
        # Combine: If either supervised OR unsupervised detects attack, flag it
        ensemble_pred = np.where((supervised_final == 1) | (iso_pred == 1), 1, 0)
        
        # Evaluate ensemble
        metrics = {
            'model_name': 'Hybrid Ensemble',
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'f1_score': f1_score(y_test, ensemble_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred)
        }
        
        logger.info("Hybrid Ensemble Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info("="*60)
        
        return ensemble_pred, metrics


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    print("See train_model.py script for complete usage example")
