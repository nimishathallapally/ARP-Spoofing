"""
Data loading and preprocessing utilities for ARP spoofing detection.

This module handles:
- Loading multiple datasets
- Data validation and quality checks
- Dataset combining and deduplication
- Feature alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess ARP spoofing datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw dataset files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.combined_data = None
        
    def load_dataset(self, filename: str, label_column: str = 'Label') -> pd.DataFrame:
        """
        Load a single CSV dataset.
        
        Args:
            filename: Name of the CSV file
            label_column: Name of the label column
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        logger.info(f"Loading dataset: {filename}")
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            logger.info(f"  Loaded {len(df):,} samples with {len(df.columns)} features")
            
            # Validate label column exists
            if label_column not in df.columns:
                logger.warning(f"  Label column '{label_column}' not found. Available columns: {df.columns.tolist()}")
                # Try to find label column with different cases
                label_candidates = [col for col in df.columns if col.lower() == label_column.lower()]
                if label_candidates:
                    label_column = label_candidates[0]
                    logger.info(f"  Using label column: {label_column}")
                else:
                    raise ValueError(f"No label column found in {filename}")
            
            # Store dataset info
            self.datasets[filename] = {
                'data': df,
                'shape': df.shape,
                'label_column': label_column,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def analyze_dataset_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Analyze dataset quality metrics.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info(f"\nAnalyzing {dataset_name}...")
        
        metrics = {
            'dataset_name': dataset_name,
            'total_samples': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicate_rows': df.duplicated().sum(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Label distribution
        if 'Label' in df.columns:
            label_counts = df['Label'].value_counts()
            metrics['label_distribution'] = label_counts.to_dict()
            metrics['class_balance_ratio'] = label_counts.min() / label_counts.max() if len(label_counts) > 1 else 1.0
        
        # Numeric feature statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metrics['numeric_features'] = len(numeric_cols)
        metrics['categorical_features'] = len(df.columns) - len(numeric_cols)
        
        # Calculate quality score (0-100)
        quality_score = 100
        quality_score -= min(metrics['missing_percentage'], 30)  # Penalize missing values
        quality_score -= min(metrics['duplicate_rows'] / len(df) * 100, 20)  # Penalize duplicates
        if 'class_balance_ratio' in metrics:
            quality_score -= (1 - metrics['class_balance_ratio']) * 20  # Penalize imbalance
        
        metrics['quality_score'] = max(0, quality_score)
        
        logger.info(f"  Total Samples: {metrics['total_samples']:,}")
        logger.info(f"  Features: {metrics['total_features']}")
        logger.info(f"  Missing Values: {metrics['missing_values']:,} ({metrics['missing_percentage']:.2f}%)")
        logger.info(f"  Duplicates: {metrics['duplicate_rows']:,}")
        logger.info(f"  Quality Score: {metrics['quality_score']:.2f}/100")
        
        return metrics
    
    def select_best_datasets(self, 
                            datasets: Dict[str, pd.DataFrame], 
                            top_n: int = 3,
                            min_quality_score: float = 60.0) -> List[str]:
        """
        Select best datasets based on quality metrics.
        
        Args:
            datasets: Dictionary of dataset name -> DataFrame
            top_n: Number of top datasets to select
            min_quality_score: Minimum quality score threshold
            
        Returns:
            List of selected dataset names
        """
        logger.info("\n" + "="*60)
        logger.info("DATASET QUALITY ANALYSIS")
        logger.info("="*60)
        
        quality_metrics = []
        
        for name, df in datasets.items():
            metrics = self.analyze_dataset_quality(df, name)
            quality_metrics.append(metrics)
        
        # Sort by quality score
        quality_metrics.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Select top datasets
        selected = [m['dataset_name'] for m in quality_metrics[:top_n] 
                   if m['quality_score'] >= min_quality_score]
        
        logger.info("\n" + "="*60)
        logger.info(f"SELECTED TOP {len(selected)} DATASETS:")
        for name in selected:
            score = next(m['quality_score'] for m in quality_metrics if m['dataset_name'] == name)
            logger.info(f"  ✓ {name} (Quality: {score:.2f}/100)")
        logger.info("="*60)
        
        return selected
    
    def combine_datasets(self, 
                        datasets: Dict[str, pd.DataFrame],
                        selected_names: Optional[List[str]] = None,
                        balance_classes: bool = True) -> pd.DataFrame:
        """
        Combine multiple datasets into one.
        
        Args:
            datasets: Dictionary of dataset name -> DataFrame
            selected_names: List of dataset names to combine (None = all)
            balance_classes: Whether to balance attack/normal samples
            
        Returns:
            Combined DataFrame
        """
        if selected_names is None:
            selected_names = list(datasets.keys())
        
        logger.info("\n" + "="*60)
        logger.info("COMBINING DATASETS")
        logger.info("="*60)
        
        combined_dfs = []
        
        for name in selected_names:
            if name not in datasets:
                logger.warning(f"Dataset {name} not found, skipping...")
                continue
                
            df = datasets[name].copy()
            logger.info(f"Adding {name}: {len(df):,} samples")
            combined_dfs.append(df)
        
        if not combined_dfs:
            raise ValueError("No datasets to combine")
        
        # Combine all datasets
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        logger.info(f"\nCombined: {len(combined_df):,} samples")
        
        # Remove duplicates
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        after_dedup = len(combined_df)
        logger.info(f"After deduplication: {after_dedup:,} samples ({before_dedup - after_dedup:,} duplicates removed)")
        
        # Balance classes if requested
        if balance_classes and 'Label' in combined_df.columns:
            label_counts = combined_df['Label'].value_counts()
            logger.info(f"\nClass distribution before balancing:")
            for label, count in label_counts.items():
                logger.info(f"  {label}: {count:,} samples")
            
            min_samples = label_counts.min()
            balanced_dfs = []
            
            for label in label_counts.index:
                label_df = combined_df[combined_df['Label'] == label]
                if len(label_df) > min_samples:
                    label_df = label_df.sample(n=min_samples, random_state=42)
                balanced_dfs.append(label_df)
            
            combined_df = pd.concat(balanced_dfs, ignore_index=True)
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            logger.info(f"\nClass distribution after balancing:")
            label_counts = combined_df['Label'].value_counts()
            for label, count in label_counts.items():
                logger.info(f"  {label}: {count:,} samples")
        
        logger.info(f"\nFinal combined dataset: {len(combined_df):,} samples, {len(combined_df.columns)} features")
        logger.info("="*60)
        
        self.combined_data = combined_df
        return combined_df
    
    def load_all_datasets(self, 
                         filenames: List[str],
                         select_best: bool = True,
                         top_n: int = 3,
                         balance_classes: bool = True) -> pd.DataFrame:
        """
        Load all datasets and return combined data.
        
        Args:
            filenames: List of dataset filenames
            select_best: Whether to select only best quality datasets
            top_n: Number of best datasets to select
            balance_classes: Whether to balance classes
            
        Returns:
            Combined DataFrame
        """
        datasets = {}
        
        for filename in filenames:
            try:
                df = self.load_dataset(filename)
                datasets[filename] = df
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                continue
        
        if not datasets:
            raise ValueError("No datasets loaded successfully")
        
        # Select best datasets if requested
        if select_best:
            selected_names = self.select_best_datasets(datasets, top_n=top_n)
        else:
            selected_names = list(datasets.keys())
        
        # Combine datasets
        combined_df = self.combine_datasets(datasets, selected_names, balance_classes)
        
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    loader = DataLoader(data_dir="../data/raw")
    
    dataset_files = [
        "CIC_MITM_ArpSpoofing_All_Labelled.csv",
        "All_Labelled.csv",
        "GIT_arpspoofLabelledData.csv"
    ]
    
    combined_df = loader.load_all_datasets(
        filenames=dataset_files,
        select_best=True,
        top_n=3,
        balance_classes=True
    )
    
    print(f"\n✓ Final dataset shape: {combined_df.shape}")
    print(f"✓ Class distribution:\n{combined_df['Label'].value_counts()}")
