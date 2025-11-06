"""
Machine learning models for ARP spoofing detection.

This module implements:
- Supervised learning models (Random Forest, Gradient Boosting, Neural Network, Logistic Regression)
- Unsupervised learning (Isolation Forest, DBSCAN, One-Class SVM, Local Outlier Factor)
- Hybrid ensemble combining supervised and unsupervised approaches
- Model training, evaluation, and selection
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
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
            # Supervised Models
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state
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
            
            # Unsupervised Anomaly Detection Models
            'Isolation Forest': IsolationForest(
                contamination=0.1,
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'One-Class SVM': OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=0.1  # Expected fraction of outliers
            ),
            'Local Outlier Factor': LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1,
                novelty=True,  # For predict() method
                n_jobs=-1
            ),
            'DBSCAN': DBSCAN(
                eps=0.5,
                min_samples=5,
                n_jobs=-1
            )
        }
        
        logger.info(f"  Initialized {len(self.models)} models")
        logger.info(f"  Supervised: Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, Neural Network")
        logger.info(f"  Unsupervised: Isolation Forest, One-Class SVM, Local Outlier Factor, DBSCAN")
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
        unsupervised_models = ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 'DBSCAN']
        
        for name, model in self.models.items():
            if name in unsupervised_models:
                continue  # Skip unsupervised models
            
            logger.info(f"\nTraining {name}...")
            try:
                model.fit(X_train, y_train)
                supervised_models[name] = model
                logger.info(f"  ✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"  ✗ Error training {name}: {str(e)}")
        
        return supervised_models
    
    def train_unsupervised_models(self, X_train: np.ndarray) -> Dict:
        """
        Train all unsupervised anomaly detection models.
        
        Args:
            X_train: Training features (unlabeled)
            
        Returns:
            Dictionary of trained unsupervised models
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING UNSUPERVISED MODELS (Anomaly Detection)")
        logger.info("="*60)
        
        unsupervised_models = {}
        
        # Train Isolation Forest
        logger.info("\nTraining Isolation Forest...")
        try:
            iso_forest = self.models.get('Isolation Forest')
            if iso_forest is None:
                iso_forest = IsolationForest(
                    contamination=0.1,
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            iso_forest.fit(X_train)
            unsupervised_models['Isolation Forest'] = iso_forest
            logger.info("  ✓ Isolation Forest trained successfully")
        except Exception as e:
            logger.error(f"  ✗ Error training Isolation Forest: {str(e)}")
        
        # Train One-Class SVM
        logger.info("\nTraining One-Class SVM...")
        try:
            ocsvm = self.models.get('One-Class SVM')
            if ocsvm is None:
                ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
            ocsvm.fit(X_train)
            unsupervised_models['One-Class SVM'] = ocsvm
            logger.info("  ✓ One-Class SVM trained successfully")
        except Exception as e:
            logger.error(f"  ✗ Error training One-Class SVM: {str(e)}")
        
        # Train Local Outlier Factor
        logger.info("\nTraining Local Outlier Factor...")
        try:
            lof = self.models.get('Local Outlier Factor')
            if lof is None:
                lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, n_jobs=-1)
            lof.fit(X_train)
            unsupervised_models['Local Outlier Factor'] = lof
            logger.info("  ✓ Local Outlier Factor trained successfully")
        except Exception as e:
            logger.error(f"  ✗ Error training Local Outlier Factor: {str(e)}")
        
        # Train DBSCAN (clustering-based anomaly detection)
        logger.info("\nTraining DBSCAN...")
        try:
            dbscan = self.models.get('DBSCAN')
            if dbscan is None:
                dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            # DBSCAN doesn't have fit, it uses fit_predict
            # We'll store it for later use
            unsupervised_models['DBSCAN'] = dbscan
            logger.info("  ✓ DBSCAN initialized successfully")
            logger.info("  Note: DBSCAN uses fit_predict() for clustering")
        except Exception as e:
            logger.error(f"  ✗ Error initializing DBSCAN: {str(e)}")
        
        logger.info(f"\n  ✓ Trained {len(unsupervised_models)} unsupervised models")
        return unsupervised_models
    
    def calculate_extended_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate extended performance metrics including TPR, FPR, FNR, TNR.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with TPR, FPR, FNR, TNR
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle case where confusion matrix might not be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # If only one class present in predictions
            tn = fp = fn = tp = 0
            if len(cm) == 1:
                if y_pred[0] == 0:  # Only predicted normal
                    tn = cm[0, 0]
                else:  # Only predicted attack
                    tp = cm[0, 0]
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall/Sensitivity)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate (Miss Rate)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        
        return {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TPR': float(tpr),  # Sensitivity / Recall
            'FPR': float(fpr),  # Fall-out
            'FNR': float(fnr),  # Miss Rate
            'TNR': float(tnr)   # Specificity
        }
    
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
        
        # Make predictions based on model type
        unsupervised_models = ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 'DBSCAN']
        
        if model_name in unsupervised_models:
            # Handle unsupervised anomaly detection models
            if model_name == 'DBSCAN':
                # DBSCAN uses fit_predict, not predict
                y_pred_raw = model.fit_predict(X_test)
                # -1 is outlier (attack), others are normal
                y_pred = np.where(y_pred_raw == -1, 1, 0)
                y_proba = None
            elif model_name in ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor']:
                # These return -1 for anomalies, 1 for normal
                y_pred_raw = model.predict(X_test)
                y_pred = np.where(y_pred_raw == -1, 1, 0)  # Convert: -1 -> 1 (attack), 1 -> 0 (normal)
                
                # Get anomaly scores
                if hasattr(model, 'score_samples'):
                    y_proba = -model.score_samples(X_test)  # Anomaly scores (negative)
                    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)  # Normalize
                elif hasattr(model, 'decision_function'):
                    y_proba = -model.decision_function(X_test)  # Decision scores
                    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)  # Normalize
                else:
                    y_proba = None
            else:
                y_pred = model.predict(X_test)
                y_proba = None
        else:
            # Supervised models
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate basic metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Calculate extended metrics (TPR, FPR, FNR, TNR)
        extended = self.calculate_extended_metrics(y_test, y_pred)
        metrics.update(extended)
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                metrics['y_proba'] = y_proba  # Store for ROC curve plotting
            except:
                metrics['roc_auc'] = None
                metrics['y_proba'] = None
        else:
            metrics['roc_auc'] = None
            metrics['y_proba'] = None
        
        # Log results
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f} (TPR)")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  TPR: {metrics['TPR']:.4f}, FPR: {metrics['FPR']:.4f}, FNR: {metrics['FNR']:.4f}, TNR: {metrics['TNR']:.4f}")
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
                'ROC AUC': m.get('roc_auc', None),
                'TPR': m.get('TPR', 0),
                'FPR': m.get('FPR', 0),
                'FNR': m.get('FNR', 0),
                'TNR': m.get('TNR', 0)
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
    
    def save_all_models(self, output_dir: str, scaler=None, feature_names: list = None) -> None:
        """
        Save all trained models to disk in proper format for Flask app.
        Each model is saved as a dictionary containing:
        - model: The trained model
        - scaler: StandardScaler for feature normalization
        - feature_names: List of feature names
        - model_name: Name of the model
        
        Args:
            output_dir: Directory to save models
            scaler: Fitted StandardScaler (optional but recommended)
            feature_names: List of feature names (optional but recommended)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("SAVING ALL MODELS IN FLASK-READY FORMAT")
        logger.info("="*60)
        
        if scaler is None:
            logger.warning("⚠ No scaler provided - models will be saved without scaling")
        if feature_names is None:
            logger.warning("⚠ No feature names provided - using generic names")
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                filename = name.lower().replace(' ', '_') + '.pkl'
                filepath = output_path / filename
                try:
                    # Create detector package with model, scaler, and feature names
                    detector_package = {
                        'model': model,
                        'scaler': scaler,
                        'feature_names': feature_names if feature_names else [f'feature_{i}' for i in range(len(feature_names) if feature_names else 25)],
                        'model_name': name,
                        'dataset': 'Combined',
                        'num_features': len(feature_names) if feature_names else 25
                    }
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(detector_package, f)
                    
                    logger.info(f"  ✓ Saved {name} (with scaler & features) to {filepath}")
                except Exception as e:
                    logger.error(f"  ✗ Error saving {name}: {str(e)}")
        
        logger.info("="*60)
        logger.info(f"✓ All models saved in Flask-ready format")
        logger.info(f"  Each model includes: model, scaler, feature_names, metadata")
        logger.info("="*60)
    
    def create_multi_hybrid_ensembles(self,
                                     best_model,
                                     unsupervised_models: Dict,
                                     X_test: np.ndarray,
                                     y_test: np.ndarray) -> Dict:
        """
        Create multiple hybrid ensemble combinations with different unsupervised models.
        
        Args:
            best_model: Best performing supervised model (typically Random Forest)
            unsupervised_models: Dictionary of trained unsupervised models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of ensemble predictions and metrics
        """
        logger.info("\n" + "="*60)
        logger.info("CREATING MULTI-HYBRID ENSEMBLES")
        logger.info("="*60)
        
        ensemble_results = {}
        
        # Get best model predictions
        y_pred_best = best_model.predict(X_test)
        if hasattr(best_model, 'predict_proba'):
            y_proba_best = best_model.predict_proba(X_test)[:, 1]
        else:
            y_proba_best = y_pred_best.astype(float)
        
        # Test each unsupervised model in hybrid with best supervised
        for unsup_name, unsup_model in unsupervised_models.items():
            logger.info(f"\nCreating Hybrid: Best Model + {unsup_name}")
            
            try:
                # Get unsupervised predictions
                if unsup_name == 'DBSCAN':
                    y_pred_unsup_raw = unsup_model.fit_predict(X_test)
                    y_pred_unsup = np.where(y_pred_unsup_raw == -1, 1, 0)
                else:
                    y_pred_unsup_raw = unsup_model.predict(X_test)
                    y_pred_unsup = np.where(y_pred_unsup_raw == -1, 1, 0)
                
                # Weighted ensemble (70% best model, 30% unsupervised)
                weighted_score = (0.7 * y_proba_best) + (0.3 * y_pred_unsup.astype(float))
                y_pred_ensemble = (weighted_score >= 0.5).astype(int)
                
                # Calculate metrics
                metrics = {
                    'model_name': f'Hybrid (Best + {unsup_name})',
                    'accuracy': accuracy_score(y_test, y_pred_ensemble),
                    'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
                    'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred_ensemble, zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble)
                }
                
                # Calculate ROC AUC
                try:
                    roc_auc = roc_auc_score(y_test, weighted_score)
                    metrics['roc_auc'] = roc_auc
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC for {unsup_name}: {str(e)}")
                    metrics['roc_auc'] = None
                
                # Extended metrics
                extended = self.calculate_extended_metrics(y_test, y_pred_ensemble)
                metrics.update(extended)
                
                ensemble_results[f'Hybrid_{unsup_name}'] = metrics
                
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                          f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"  ✗ Error creating hybrid with {unsup_name}: {str(e)}")
        
        logger.info("="*60)
        return ensemble_results
    
    def create_hybrid_ensemble(self, 
                              supervised_models: Dict,
                              iso_forest: IsolationForest,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              rf_weight: float = 0.7,
                              iso_weight: float = 0.3) -> Tuple[np.ndarray, Dict]:
        """
        Create hybrid ensemble combining supervised and unsupervised predictions with weights.
        
        Args:
            supervised_models: Dictionary of trained supervised models
            iso_forest: Trained Isolation Forest
            X_test: Test features
            y_test: Test labels
            rf_weight: Weight for Random Forest predictions (default: 0.7)
            iso_weight: Weight for Isolation Forest predictions (default: 0.3)
            
        Returns:
            Tuple of (ensemble_predictions, metrics)
        """
        logger.info("\n" + "="*60)
        logger.info("CREATING WEIGHTED HYBRID ENSEMBLE")
        logger.info(f"Random Forest Weight: {rf_weight:.2f}, Isolation Forest Weight: {iso_weight:.2f}")
        logger.info("="*60)
        
        # Get Random Forest predictions (probabilities if available)
        rf_pred = None
        rf_proba = None
        for name, model in supervised_models.items():
            if 'Random Forest' in name:
                rf_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    rf_proba = model.predict_proba(X_test)[:, 1]  # Probability of attack class
                break
        
        if rf_pred is None:
            logger.warning("Random Forest not found, falling back to simple ensemble")
            # Fallback to simple OR logic
            supervised_preds = []
            for name, model in supervised_models.items():
                if name == 'Isolation Forest':
                    continue
                pred = model.predict(X_test)
                supervised_preds.append(pred)
            supervised_avg = np.mean(supervised_preds, axis=0)
            supervised_final = (supervised_avg >= 0.5).astype(int)
            iso_pred_raw = iso_forest.predict(X_test)
            iso_pred = np.where(iso_pred_raw == -1, 1, 0)
            ensemble_pred = np.where((supervised_final == 1) | (iso_pred == 1), 1, 0)
        else:
            # Get Isolation Forest predictions
            iso_pred_raw = iso_forest.predict(X_test)
            iso_pred = np.where(iso_pred_raw == -1, 1, 0).astype(float)
            
            # Weighted voting
            if rf_proba is not None:
                # Use probabilities for RF
                weighted_score = (rf_weight * rf_proba) + (iso_weight * iso_pred)
                ensemble_pred = (weighted_score >= 0.5).astype(int)
                logger.info("Using probability-based weighted ensemble")
            else:
                # Use binary predictions with weights
                weighted_score = (rf_weight * rf_pred.astype(float)) + (iso_weight * iso_pred)
                ensemble_pred = (weighted_score >= 0.5).astype(int)
                logger.info("Using binary-based weighted ensemble")
        
        # Evaluate ensemble
        metrics = {
            'model_name': f'Weighted Hybrid (RF:{rf_weight:.1f}, IF:{iso_weight:.1f})',
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'f1_score': f1_score(y_test, ensemble_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred)
        }
        
        # Calculate ROC AUC if we have probabilities/scores
        try:
            if rf_proba is not None:
                # Use weighted score for ROC AUC
                roc_auc = roc_auc_score(y_test, weighted_score)
                metrics['roc_auc'] = roc_auc
            else:
                metrics['roc_auc'] = None
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            metrics['roc_auc'] = None
        
        # Calculate extended metrics for hybrid ensemble
        extended = self.calculate_extended_metrics(y_test, ensemble_pred)
        metrics.update(extended)
        
        logger.info("Weighted Hybrid Ensemble Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f} (TPR)")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  TPR: {metrics['TPR']:.4f}, FPR: {metrics['FPR']:.4f}, FNR: {metrics['FNR']:.4f}, TNR: {metrics['TNR']:.4f}")
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
