"""
Real-time ARP spoofing detector.

This module provides:
- ARPSpoofingDetector class for production deployment
- Real-time packet analysis
- Confidence scoring and alert level classification
- Model persistence (save/load)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ARPSpoofingDetector:
    """Real-time ARP spoofing detection system."""
    
    def __init__(self, 
                 model,
                 scaler: StandardScaler,
                 feature_names: List[str],
                 model_name: str = "Unknown",
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize detector.
        
        Args:
            model: Trained ML model
            scaler: Fitted StandardScaler from training
            feature_names: List of feature names
            model_name: Name of the model
            alert_thresholds: Custom alert level thresholds
        """
        self.model = model
        self.scaler = scaler  # Use the SAME scaler from training
        self.feature_names = feature_names
        self.model_name = model_name
        
        # Default alert thresholds
        if alert_thresholds is None:
            self.alert_thresholds = {
                'SAFE': (0.0, 0.3),
                'MEDIUM': (0.3, 0.6),
                'HIGH': (0.6, 0.85),
                'CRITICAL': (0.85, 1.0)
            }
        else:
            self.alert_thresholds = alert_thresholds
        
        logger.info(f"Detector initialized with {model_name}")
        logger.info(f"  Features: {len(feature_names)}")
        logger.info(f"  Alert levels: {list(self.alert_thresholds.keys())}")
    
    def _get_alert_level(self, confidence: float) -> str:
        """
        Determine alert level based on confidence.
        
        Args:
            confidence: Prediction confidence (0-1)
            
        Returns:
            Alert level string
        """
        for level, (min_conf, max_conf) in self.alert_thresholds.items():
            if min_conf <= confidence < max_conf:
                return level
        return 'CRITICAL'
    
    def detect(self, packet_features) -> Dict:
        """
        Detect if a packet is an ARP spoofing attack.
        
        Args:
            packet_features: Dictionary of packet features OR numpy array of feature values
            
        Returns:
            Dictionary with detection results
        """
        # Handle numpy array input
        if isinstance(packet_features, np.ndarray):
            # Already scaled features in correct order
            if packet_features.ndim == 1:
                packet_scaled = packet_features.reshape(1, -1)
            else:
                packet_scaled = packet_features
        else:
            # Handle dictionary input
            packet_df = pd.DataFrame([packet_features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in packet_df.columns:
                    logger.warning(f"Missing feature: {feature}, setting to 0")
                    packet_df[feature] = 0
            
            # Select only required features in correct order
            packet_df = packet_df[self.feature_names]
            
            # Scale features
            packet_scaled = self.scaler.transform(packet_df)
        
        # Make prediction
        prediction = self.model.predict(packet_scaled)[0]
        
        # Get confidence (probability of attack class)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(packet_scaled)[0]
            # For binary classification: class 1 is attack, class 0 is normal
            # We want probability of attack for alert level
            if prediction == 1:
                confidence = proba[1]  # Probability of attack
            else:
                confidence = proba[0]  # Probability of normal (low risk)
        else:
            confidence = 1.0 if prediction == 1 else 0.0
        
        # Determine alert level based on attack probability
        attack_prob = proba[1] if hasattr(self.model, 'predict_proba') else (1.0 if prediction == 1 else 0.0)
        alert_level = self._get_alert_level(attack_prob)
        
        # Prepare result
        result = {
            'prediction': 'arp_spoofing' if prediction == 1 else 'normal',
            'label': 'arp_spoofing' if prediction == 1 else 'normal',
            'prediction_label': int(prediction),
            'confidence': float(confidence),
            'probability': float(attack_prob),
            'alert_level': alert_level,
            'model_name': self.model_name,
            'timestamp': pd.Timestamp.now()
        }
        
        return result
    
    def detect_batch(self, packets: List[Dict[str, float]]) -> List[Dict]:
        """
        Detect multiple packets at once.
        
        Args:
            packets: List of packet feature dictionaries
            
        Returns:
            List of detection results
        """
        results = []
        for packet in packets:
            result = self.detect(packet)
            results.append(result)
        return results
    
    def simulate_realtime(self, 
                         X_test: np.ndarray, 
                         y_test: np.ndarray,
                         n_packets: int = 100,
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Simulate real-time detection on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            n_packets: Number of packets to simulate
            random_state: Random seed
            
        Returns:
            Tuple of (true_labels, predictions, detection_results)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMULATING REAL-TIME DETECTION")
        logger.info(f"{'='*60}")
        logger.info(f"Processing {n_packets} packets...")
        
        # Select random packets
        np.random.seed(random_state)
        indices = np.random.choice(len(X_test), size=n_packets, replace=False)
        
        sim_X = X_test[indices]
        # Handle pandas Series indexing
        if hasattr(y_test, 'iloc'):
            sim_y = y_test.iloc[indices].values
        else:
            sim_y = y_test[indices]
        
        # Make predictions
        predictions = self.model.predict(sim_X)
        
        # Get confidence scores
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(sim_X)
            confidences = probas[np.arange(len(predictions)), predictions]
        else:
            confidences = np.ones(len(predictions))
        
        # Create detection results
        detection_results = []
        for i, (pred, conf, true_label) in enumerate(zip(predictions, confidences, sim_y)):
            result = {
                'packet_id': i + 1,
                'prediction': 'ATTACK' if pred == 1 else 'NORMAL',
                'true_label': 'ATTACK' if true_label == 1 else 'NORMAL',
                'correct': pred == true_label,
                'confidence': float(conf),
                'alert_level': self._get_alert_level(conf)
            }
            detection_results.append(result)
        
        # Calculate statistics
        correct = sum(1 for r in detection_results if r['correct'])
        accuracy = correct / n_packets
        
        # Alert level distribution
        alert_counts = {}
        for r in detection_results:
            level = r['alert_level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        logger.info(f"\n{'='*60}")
        logger.info("SIMULATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Packets: {n_packets}")
        logger.info(f"Correct Predictions: {correct} ({accuracy:.2%})")
        logger.info(f"Incorrect Predictions: {n_packets - correct} ({(1-accuracy):.2%})")
        logger.info(f"\nAlert Level Distribution:")
        for level in ['SAFE', 'MEDIUM', 'HIGH', 'CRITICAL']:
            count = alert_counts.get(level, 0)
            logger.info(f"  {level}: {count} packets ({count/n_packets:.1%})")
        logger.info(f"{'='*60}")
        
        return sim_y, predictions, detection_results
    
    def save(self, filepath: str):
        """
        Save detector to file.
        
        Args:
            filepath: Path to save file
        """
        save_obj = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.model_name,
            'alert_thresholds': self.alert_thresholds
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(save_obj, filepath)
        logger.info(f"Detector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ARPSpoofingDetector':
        """
        Load detector from file.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            Loaded ARPSpoofingDetector instance
        """
        save_obj = joblib.load(filepath)
        
        # Create detector instance
        detector = cls(
            model=save_obj['model'],
            X_train_data=np.zeros((1, len(save_obj['feature_names']))),  # Dummy data
            feature_names=save_obj['feature_names'],
            model_name=save_obj['model_name'],
            alert_thresholds=save_obj['alert_thresholds']
        )
        
        # Replace scaler with loaded one
        detector.scaler = save_obj['scaler']
        
        logger.info(f"Detector loaded from {filepath}")
        return detector
    
    def get_feature_importance(self, top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance or None
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"{self.model_name} does not support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_prediction(self, packet_features: Dict[str, float], top_n: int = 5) -> Dict:
        """
        Explain why a prediction was made (basic explanation).
        
        Args:
            packet_features: Packet features
            top_n: Number of top contributing features
            
        Returns:
            Dictionary with explanation
        """
        result = self.detect(packet_features)
        
        # Get feature importance
        importance_df = self.get_feature_importance(top_n=top_n)
        
        explanation = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'alert_level': result['alert_level']
        }
        
        if importance_df is not None:
            # Get values of top important features
            top_features = []
            for _, row in importance_df.iterrows():
                feature_name = row['feature']
                if feature_name in packet_features:
                    top_features.append({
                        'feature': feature_name,
                        'value': packet_features[feature_name],
                        'importance': row['importance']
                    })
            
            explanation['top_contributing_features'] = top_features
        
        return explanation


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("See detect_realtime.py script for usage examples")
