"""
Feature Extraction Module
Implements TF-IDF vectorization for log sequences.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class LogFeatureExtractor:
    """
    Log feature extractor using TF-IDF vectorization.
    
    Attributes:
        max_features (int): Maximum number of features.
        min_df (int): Minimum document frequency.
        max_df (float): Maximum document frequency.
    """
    
    def __init__(self, max_features: int = 200, min_df: int = 1, max_df: float = 1.0):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            analyzer='word',
            token_pattern=r'\S+'
        )
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def fit_transform(self, log_sequences: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform log sequences.
        
        Args:
            log_sequences (List[str]): List of log sequence strings.
            
        Returns:
            np.ndarray: Normalized feature matrix.
        """
        tfidf_matrix = self.vectorizer.fit_transform(log_sequences)
        dense_matrix = tfidf_matrix.toarray()
        
        normalized_matrix = self.scaler.fit_transform(dense_matrix)
        
        self.is_fitted = True
        logger.info(f"Vectorizer fitted with {normalized_matrix.shape[1]} features")
        
        return normalized_matrix
    
    def transform(self, log_sequences: List[str]) -> np.ndarray:
        """
        Transform log sequences using fitted vectorizer.
        
        Args:
            log_sequences (List[str]): List of log sequence strings.
            
        Returns:
            np.ndarray: Normalized feature matrix.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        tfidf_matrix = self.vectorizer.transform(log_sequences)
        dense_matrix = tfidf_matrix.toarray()
        
        normalized_matrix = self.scaler.transform(dense_matrix)
        
        return normalized_matrix
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from vectorizer.
        
        Returns:
            List[str]: List of feature names.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_n_features(self) -> int:
        """
        Get number of features.
        
        Returns:
            int: Number of features.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
        
        return len(self.get_feature_names())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config import Config
    from log_parser import DrainLogParser
    
    print("Extracting HDFS features...")
    parser = DrainLogParser(depth=Config.DRAIN_DEPTH)
    hdfs_ids, hdfs_templates = parser.parse_file(Config.HDFS_LOG_PATH)
    
    extractor = LogFeatureExtractor(
        max_features=Config.TFIDF_MAX_FEATURES,
        min_df=Config.TFIDF_MIN_DF,
        max_df=Config.TFIDF_MAX_DF
    )
    
    hdfs_features = extractor.fit_transform(hdfs_templates)
    print(f"HDFS features shape: {hdfs_features.shape}")
    print(f"Feature range: [{hdfs_features.min():.3f}, {hdfs_features.max():.3f}]")
    
    print("\nExtracting BGL features...")
    parser = DrainLogParser(depth=Config.DRAIN_DEPTH)
    bgl_ids, bgl_templates = parser.parse_file(Config.BGL_LOG_PATH)
    
    extractor_bgl = LogFeatureExtractor(
        max_features=Config.TFIDF_MAX_FEATURES,
        min_df=Config.TFIDF_MIN_DF,
        max_df=Config.TFIDF_MAX_DF
    )
    
    bgl_features = extractor_bgl.fit_transform(bgl_templates)
    print(f"BGL features shape: {bgl_features.shape}")
    print(f"Feature range: [{bgl_features.min():.3f}, {bgl_features.max():.3f}]")