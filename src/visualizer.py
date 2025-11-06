"""
Visualization utilities for ARP spoofing detection.

This module creates all plots for:
- Exploratory Data Analysis (EDA)
- Model Performance Evaluation
- Real-Time Detection Results
- Feature Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Create visualizations for ARP spoofing detection."""
    
    def __init__(self, output_dir: str = "outputs/plots"):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
        logger.info(f"Visualizer initialized. Plots will be saved to: {self.output_dir}")
    
    def plot_class_distribution(self, y: np.ndarray, title: str = "Class Distribution"):
        """
        Plot class distribution as bar chart and pie chart.
        
        Args:
            y: Labels array
            title: Plot title
        """
        logger.info("Creating class distribution plots...")
        
        unique, counts = np.unique(y, return_counts=True)
        labels = ['Normal' if label == 0 else 'Attack' for label in unique]
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Class Distribution - Bar Chart', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (label, count) in enumerate(zip(labels, counts)):
            axes[0].text(i, count + max(counts)*0.01, f'{count:,}\n({count/sum(counts)*100:.1f}%)',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Pie chart
        explode = (0.05, 0.05)
        axes[1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=colors, explode=explode, shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title('Class Distribution - Pie Chart', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'class_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_feature_correlation(self, X: pd.DataFrame, top_n: int = 25):
        """
        Plot correlation heatmap of features.
        
        Args:
            X: Feature DataFrame
            top_n: Number of top features to show
        """
        logger.info("Creating correlation heatmap...")
        
        # Select top features if more than top_n
        if len(X.columns) > top_n:
            # Calculate variance and select top features
            variances = X.var().sort_values(ascending=False)
            top_features = variances.head(top_n).index
            X_subset = X[top_features]
        else:
            X_subset = X
        
        # Calculate correlation
        corr = X_subset.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'correlation_matrix.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_feature_distributions(self, X: pd.DataFrame, y: np.ndarray, 
                                   features: List[str], n_cols: int = 3):
        """
        Plot distributions of selected features by class.
        
        Args:
            X: Feature DataFrame
            y: Labels
            features: List of features to plot
            n_cols: Number of columns in subplot grid
        """
        logger.info(f"Creating feature distribution plots for {len(features)} features...")
        
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature not in X.columns:
                continue
                
            ax = axes[idx]
            
            # Plot distributions for each class
            for label, color, name in [(0, '#2ecc71', 'Normal'), (1, '#e74c3c', 'Attack')]:
                # Use .loc to ensure proper alignment, or convert y to numpy
                if hasattr(y, 'values'):
                    mask = y.values == label
                else:
                    mask = y == label
                data = X[mask][feature]
                ax.hist(data, bins=30, alpha=0.6, label=name, color=color, edgecolor='black')
            
            ax.set_xlabel(feature, fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'Distribution: {feature}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'feature_distributions.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model"):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        logger.info(f"Creating confusion matrix for {model_name}...")
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   square=True, linewidths=2, linecolor='black',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                   ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=15, fontweight='bold', pad=20)
        
        # Add performance metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.2%}\nPrecision: {precision:.2%}\nRecall: {recall:.2%}\nF1-Score: {f1:.2%}'
        ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_roc_curve(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_data: Dict of {model_name: (y_true, y_proba)}
        """
        logger.info(f"Creating ROC curves for {len(models_data)} models...")
        
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
        
        for (model_name, (y_true, y_proba)), color in zip(models_data.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'roc_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: np.ndarray, 
                               top_n: int = 20,
                               model_name: str = "Model"):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            top_n: Number of top features to show
            model_name: Name of the model
        """
        logger.info(f"Creating feature importance plot for {model_name}...")
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=colors, edgecolor='black')
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'feature_importance.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame):
        """
        Plot model comparison metrics.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
        """
        logger.info("Creating model comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for ax, metric in zip(axes.flatten(), metrics):
            data = comparison_df.sort_values(metric, ascending=True)
            
            colors = plt.cm.RdYlGn(data[metric])
            bars = ax.barh(range(len(data)), data[metric], color=colors, edgecolor='black')
            
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data['Model'], fontsize=10)
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
            ax.set_xlim([0, 1.0])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                ax.text(val, i, f' {val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'model_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_detection_timeline(self, detection_results: List[Dict], 
                               max_packets: int = 100):
        """
        Plot real-time detection timeline.
        
        Args:
            detection_results: List of detection result dictionaries
            max_packets: Maximum number of packets to plot
        """
        logger.info("Creating detection timeline plot...")
        
        results = detection_results[:max_packets]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Extract data
        packet_ids = [r['packet_id'] for r in results]
        predictions = [1 if r['prediction'] == 'ATTACK' else 0 for r in results]
        true_labels = [1 if r['true_label'] == 'ATTACK' else 0 for r in results]
        confidences = [r['confidence'] for r in results]
        correct = [r['correct'] for r in results]
        
        # Plot 1: Predictions vs True Labels
        ax1.scatter(packet_ids, predictions, c=['red' if p == 1 else 'green' for p in predictions],
                   s=100, alpha=0.6, label='Predicted', marker='o', edgecolors='black')
        ax1.scatter(packet_ids, true_labels, c=['darkred' if t == 1 else 'darkgreen' for t in true_labels],
                   s=50, alpha=0.8, label='True Label', marker='x')
        
        ax1.set_ylim([-0.2, 1.2])
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Normal', 'Attack'], fontsize=11, fontweight='bold')
        ax1.set_xlabel('Packet Number', fontsize=12, fontweight='bold')
        ax1.set_title('Detection Timeline - Predictions vs True Labels', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Highlight incorrect predictions
        incorrect_ids = [pid for pid, c in zip(packet_ids, correct) if not c]
        if incorrect_ids:
            for pid in incorrect_ids:
                ax1.axvline(x=pid, color='orange', alpha=0.3, linestyle='--', linewidth=2)
        
        # Plot 2: Confidence scores
        colors = ['red' if p == 1 else 'green' for p in predictions]
        ax2.bar(packet_ids, confidences, color=colors, alpha=0.6, edgecolor='black')
        
        # Add alert level thresholds
        ax2.axhline(y=0.3, color='yellow', linestyle='--', linewidth=2, alpha=0.5, label='Medium Threshold')
        ax2.axhline(y=0.6, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='High Threshold')
        ax2.axhline(y=0.85, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Critical Threshold')
        
        ax2.set_ylim([0, 1.0])
        ax2.set_xlabel('Packet Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax2.set_title('Detection Confidence Timeline', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'detection_timeline.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_alert_level_distribution(self, detection_results: List[Dict]):
        """
        Plot alert level distribution.
        
        Args:
            detection_results: List of detection result dictionaries
        """
        logger.info("Creating alert level distribution plot...")
        
        # Count alert levels
        alert_counts = {}
        for result in detection_results:
            level = result['alert_level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        # Ensure all levels are present
        for level in ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']:
            if level not in alert_counts:
                alert_counts[level] = 0
        
        levels = ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']
        counts = [alert_counts[level] for level in levels]
        colors_map = {'SAFE': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e67e22', 'CRITICAL': '#e74c3c'}
        colors = [colors_map[level] for level in levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        bars = ax1.bar(levels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_xlabel('Alert Level', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Number of Packets', fontsize=13, fontweight='bold')
        ax1.set_title('Alert Level Distribution - Bar Chart', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/sum(counts)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Pie chart
        ax2.pie(counts, labels=levels, autopct='%1.1f%%', startangle=90,
               colors=colors, explode=(0.05, 0.05, 0.05, 0.05), shadow=True,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Alert Level Distribution - Pie Chart', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / 'alert_level_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def generate_all_plots(self, 
                          X_train: np.ndarray,
                          X_test: np.ndarray,
                          y_train: np.ndarray,
                          y_test: np.ndarray,
                          feature_names: List[str],
                          models: Dict,
                          comparison_df: pd.DataFrame,
                          detection_results: Optional[List[Dict]] = None):
        """
        Generate all plots at once.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            feature_names: List of feature names
            models: Dictionary of trained models
            comparison_df: Model comparison DataFrame
            detection_results: Real-time detection results
        """
        logger.info("\n" + "="*60)
        logger.info("GENERATING ALL VISUALIZATIONS")
        logger.info("="*60)
        
        # Convert to DataFrame for some plots
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # 1. Class distribution
        self.plot_class_distribution(y_train, "Training Set - Class Distribution")
        
        # 2. Correlation matrix
        self.plot_feature_correlation(X_train_df)
        
        # 3. Feature distributions (top 6 features)
        top_features = feature_names[:6] if len(feature_names) >= 6 else feature_names
        self.plot_feature_distributions(X_train_df, y_train, top_features)
        
        # 4. Model comparison
        self.plot_model_comparison(comparison_df)
        
        # 5. Confusion matrices for each model
        for model_name, model in models.items():
            if model_name == 'Isolation Forest':
                y_pred_raw = model.predict(X_test)
                y_pred = np.where(y_pred_raw == -1, 1, 0)
            else:
                y_pred = model.predict(X_test)
            
            self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        # 6. ROC curves
        roc_data = {}
        for model_name, model in models.items():
            if model_name == 'Isolation Forest':
                y_scores = model.score_samples(X_test)
                y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
            elif hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                continue
            
            roc_data[model_name] = (y_test, y_scores)
        
        if roc_data:
            self.plot_roc_curve(roc_data)
        
        # 7. Feature importance (for Random Forest)
        if 'Random Forest' in models:
            rf_model = models['Random Forest']
            if hasattr(rf_model, 'feature_importances_'):
                self.plot_feature_importance(feature_names, rf_model.feature_importances_, 
                                            model_name='Random Forest')
        
        # 8. Detection timeline and alert levels
        if detection_results:
            self.plot_detection_timeline(detection_results)
            self.plot_alert_level_distribution(detection_results)
        
        logger.info("="*60)
        logger.info("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        logger.info(f"Plots saved to: {self.output_dir}")
        logger.info("="*60 + "\n")
    
    def plot_all_confusion_matrices(self, performance_metrics: Dict, y_test: np.ndarray):
        """
        Plot confusion matrices for all models in a grid layout.
        
        Args:
            performance_metrics: Dictionary of model metrics containing confusion matrices
            y_test: True labels for test set
        """
        logger.info("Creating comprehensive confusion matrix grid...")
        
        # Filter out train metrics and get only test metrics
        test_metrics = {k: v for k, v in performance_metrics.items() if not k.endswith('_train')}
        
        n_models = len(test_metrics)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(sorted(test_metrics.items())):
            cm = metrics.get('confusion_matrix')
            if cm is None:
                continue
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, square=True, linewidths=1, linecolor='gray')
            
            axes[idx].set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.2%}, F1: {metrics["f1_score"]:.2%}',
                              fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xticklabels(['Normal', 'Attack'])
            axes[idx].set_yticklabels(['Normal', 'Attack'])
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = self.output_dir / 'all_confusion_matrices.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")
    
    def plot_all_roc_curves(self, performance_metrics: Dict, y_test: np.ndarray):
        """
        Plot ROC curves for all models in a single figure.
        
        Args:
            performance_metrics: Dictionary of model metrics containing y_proba
            y_test: True test labels
        """
        logger.info("Creating comprehensive ROC curves plot...")
        
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(12, 8))
        
        # Filter out train metrics
        test_metrics = {k: v for k, v in performance_metrics.items() if not k.endswith('_train')}
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(test_metrics)))
        
        for idx, (model_name, metrics) in enumerate(sorted(test_metrics.items())):
            y_proba = metrics.get('y_proba')
            if y_proba is None or len(y_proba) == 0:
                continue
            
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                logger.warning(f"Could not plot ROC for {model_name}: {str(e)}")
                continue
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        
        filepath = self.output_dir / 'all_roc_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Visualizer module loaded. Use generate_all_plots() to create all visualizations.")
