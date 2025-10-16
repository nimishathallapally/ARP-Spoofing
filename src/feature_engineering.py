"""
Feature engineering and preprocessing for ARP spoofing detection.

This module handles:
- Feature selection (SelectKBest, mutual information, random forest importance)
- Feature scaling and normalization
- Feature engineering (derived features)
- Train-test splitting
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering and preprocessing pipeline."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        self.feature_importance = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Remove columns with all missing values
        null_cols = df_clean.columns[df_clean.isnull().all()]
        if len(null_cols) > 0:
            logger.info(f"  Removing {len(null_cols)} columns with all null values")
            df_clean = df_clean.drop(columns=null_cols)
        
        # Fill remaining missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        # Handle infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(0)
        
        logger.info(f"  Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def encode_labels(self, y: pd.Series) -> np.ndarray:
        """
        Encode string labels to numeric.
        
        Args:
            y: Label series
            
        Returns:
            Encoded labels (0 = normal, 1 = attack)
        """
        # Map common label formats
        label_mapping = {
            'normal': 0,
            'benign': 0,
            'legitimate': 0,
            '0': 0,
            'attack': 1,
            'malicious': 1,
            'arp_spoofing': 1,
            '1': 1
        }
        
        if y.dtype == 'object':
            y_lower = y.str.lower().str.strip()
            y_encoded = y_lower.map(label_mapping)
            
            # Check for unmapped values
            if y_encoded.isnull().any():
                logger.warning(f"  Found {y_encoded.isnull().sum()} unmapped labels, using LabelEncoder")
                y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        logger.info(f"  Label distribution: Normal={np.sum(y_encoded == 0)}, Attack={np.sum(y_encoded == 1)}")
        return y_encoded
    
    def select_features_f_test(self, X: pd.DataFrame, y: np.ndarray, k: int = 20) -> List[str]:
        """
        Select features using ANOVA F-test.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using F-test...")
        
        # Remove constant features to avoid warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Filter out constant features
            non_constant_cols = X.columns[X.std() > 0].tolist()
            X_filtered = X[non_constant_cols]
            
            if len(non_constant_cols) < len(X.columns):
                logger.debug(f"  Removed {len(X.columns) - len(non_constant_cols)} constant features")
            
            selector = SelectKBest(score_func=f_classif, k=min(k, len(non_constant_cols)))
            selector.fit(X_filtered, y)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_features = X_filtered.columns[selected_mask].tolist()
            
            # Get scores
            scores = pd.DataFrame({
                'feature': X_filtered.columns,
                'score': selector.scores_
            }).sort_values('score', ascending=False)
        
        logger.info(f"  Top 5 features by F-score:")
        for idx, row in scores.head().iterrows():
            logger.info(f"    {row['feature']}: {row['score']:.2f}")
        
        return selected_features
    
    def select_features_mutual_info(self, X: pd.DataFrame, y: np.ndarray, k: int = 20) -> List[str]:
        """
        Select features using mutual information.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using mutual information...")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        logger.info(f"  Top 5 features by MI score:")
        for idx, row in scores.head().iterrows():
            logger.info(f"    {row['feature']}: {row['score']:.4f}")
        
        return selected_features
    
    def select_features_rf_importance(self, X: pd.DataFrame, y: np.ndarray, k: int = 20) -> List[str]:
        """
        Select features using Random Forest feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using Random Forest importance...")
        
        # Train a quick RF model
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features = importance_df.head(k)['feature'].tolist()
        self.feature_importance = importance_df
        
        logger.info(f"  Top 5 features by RF importance:")
        for idx, row in importance_df.head().iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        
        return selected_features
    
    def select_features_hybrid(self, X: pd.DataFrame, y: np.ndarray, k: int = 25) -> List[str]:
        """
        Select features using hybrid approach (combination of methods).
        
        Args:
            X: Feature DataFrame
            y: Target labels
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"\nSelecting top {k} features using hybrid approach...")
        
        # Get features from each method
        f_test_features = set(self.select_features_f_test(X, y, k=k))
        mi_features = set(self.select_features_mutual_info(X, y, k=k))
        rf_features = set(self.select_features_rf_importance(X, y, k=k))
        
        # Features that appear in at least 2 methods
        feature_votes = {}
        for feature in X.columns:
            votes = 0
            if feature in f_test_features:
                votes += 1
            if feature in mi_features:
                votes += 1
            if feature in rf_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Sort by votes, then by RF importance if available
        if self.feature_importance is not None:
            importance_dict = dict(zip(self.feature_importance['feature'], 
                                     self.feature_importance['importance']))
        else:
            importance_dict = {f: 0 for f in X.columns}
        
        selected_features = sorted(feature_votes.keys(), 
                                  key=lambda f: (feature_votes[f], importance_dict.get(f, 0)), 
                                  reverse=True)[:k]
        
        logger.info(f"\n  Selected {len(selected_features)} features:")
        for feature in selected_features[:10]:
            logger.info(f"    {feature} (votes: {feature_votes[feature]})")
        if len(selected_features) > 10:
            logger.info(f"    ... and {len(selected_features) - 10} more")
        
        self.selected_features = selected_features
        return selected_features
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        logger.info("Creating derived features...")
        df_enhanced = df.copy()
        
        # Packet rate features
        if 'bidirectional_packets' in df.columns and 'bidirectional_duration_ms' in df.columns:
            df_enhanced['packet_rate'] = df['bidirectional_packets'] / (df['bidirectional_duration_ms'] + 1)
        
        # Byte rate features
        if 'bidirectional_bytes' in df.columns and 'bidirectional_duration_ms' in df.columns:
            df_enhanced['byte_rate'] = df['bidirectional_bytes'] / (df['bidirectional_duration_ms'] + 1)
        
        # Average packet size
        if 'bidirectional_bytes' in df.columns and 'bidirectional_packets' in df.columns:
            df_enhanced['avg_packet_size'] = df['bidirectional_bytes'] / (df['bidirectional_packets'] + 1)
        
        # Port features
        if 'src_port' in df.columns:
            df_enhanced['src_port_wellknown'] = (df['src_port'] < 1024).astype(int)
        if 'dst_port' in df.columns:
            df_enhanced['dst_port_wellknown'] = (df['dst_port'] < 1024).astype(int)
        
        new_features = len(df_enhanced.columns) - len(df.columns)
        logger.info(f"  Created {new_features} new features")
        
        return df_enhanced
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_column: str = 'Label',
                    feature_selection_method: str = 'hybrid',
                    n_features: int = 25,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_selection_method: Method for feature selection
            n_features: Number of features to select
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test, feature_names
        """
        logger.info("\n" + "="*60)
        logger.info("DATA PREPARATION PIPELINE")
        logger.info("="*60)
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Create derived features
        df_enhanced = self.create_derived_features(df_clean)
        
        # Step 3: Separate features and target
        if target_column not in df_enhanced.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        y = df_enhanced[target_column]
        X = df_enhanced.drop(columns=[target_column])
        
        # Keep only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        logger.info(f"\nFeatures before selection: {X.shape[1]}")
        
        # Step 4: Encode labels
        y_encoded = self.encode_labels(y)
        
        # Step 5: Feature selection
        if feature_selection_method == 'hybrid':
            selected_features = self.select_features_hybrid(X, y_encoded, k=n_features)
        elif feature_selection_method == 'f_test':
            selected_features = self.select_features_f_test(X, y_encoded, k=n_features)
        elif feature_selection_method == 'mutual_info':
            selected_features = self.select_features_mutual_info(X, y_encoded, k=n_features)
        elif feature_selection_method == 'rf_importance':
            selected_features = self.select_features_rf_importance(X, y_encoded, k=n_features)
        else:
            # Use all features
            selected_features = X.columns.tolist()
        
        X_selected = X[selected_features]
        logger.info(f"Features after selection: {X_selected.shape[1]}")
        
        # Step 6: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        logger.info(f"\nTrain set: {X_train.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        # Step 7: Feature scaling
        logger.info("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("="*60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, selected_features


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(data_dir="../data/raw")
    dataset_files = [
        "CIC_MITM_ArpSpoofing_All_Labelled.csv",
        "All_Labelled.csv",
        "GIT_arpspoofLabelledData.csv"
    ]
    combined_df = loader.load_all_datasets(dataset_files)
    
    # Prepare data
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = engineer.prepare_data(
        combined_df,
        feature_selection_method='hybrid',
        n_features=25
    )
    
    print(f"\n✓ Training data shape: {X_train.shape}")
    print(f"✓ Test data shape: {X_test.shape}")
    print(f"✓ Selected features: {len(features)}")
