"""
Data Preparation Module
Implements temporal data splitting and sequence creation.
"""

import numpy as np
import logging
from typing import Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Data preparation for training/validation/testing split.
    Implements temporal splitting to prevent data leakage.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def temporal_split(self, data: np.ndarray, 
                      train_ratio: float = 0.60,
                      val_ratio: float = 0.20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally to preserve temporal order.
        Prevents data leakage in time-series anomaly detection.
        
        Args:
            data (np.ndarray): Input data array.
            train_ratio (float): Training set ratio.
            val_ratio (float): Validation set ratio.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, validation, test sets.
        """
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]
        
        logger.info(f"Temporal split:")
        logger.info(f"  Training:   {len(train)} samples ({100*train_ratio:.0f}%)")
        logger.info(f"  Validation: {len(val)} samples ({100*val_ratio:.0f}%)")
        logger.info(f"  Testing:    {len(test)} samples ({100*(1-train_ratio-val_ratio):.0f}%)")
        
        return train, val, test
    
    def create_sequences(self, data: np.ndarray, 
                        sequence_ids: List[int],
                        window_size: int = 10) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create log sequences from individual log vectors.
        Groups consecutive logs into sequences.
        
        Args:
            data (np.ndarray): Feature matrix.
            sequence_ids (List[int]): Sequence identifiers for grouping.
            window_size (int): Sequence window size.
            
        Returns:
            Tuple[List[np.ndarray], List[int]]: List of sequences and their IDs.
        """
        sequences = []
        seq_ids = []
        
        for seq_id in sorted(set(sequence_ids)):
            indices = [i for i, s_id in enumerate(sequence_ids) if s_id == seq_id]
            
            for start in range(0, len(indices), window_size):
                end = min(start + window_size, len(indices))
                window_indices = indices[start:end]
                
                sequence = data[window_indices]
                sequences.append(sequence)
                seq_ids.append(seq_id)
        
        logger.info(f"Created {len(sequences)} sequences from {len(set(sequence_ids))} unique sequence IDs")
        return sequences, seq_ids
    
    def aggregate_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate sequences into single feature vectors.
        Uses mean pooling across sequence dimension.
        
        Args:
            sequences (List[np.ndarray]): List of sequence arrays.
            
        Returns:
            np.ndarray: Aggregated feature matrix.
        """
        aggregated = []
        
        for seq in sequences:
            if seq.ndim == 2:
                agg_vector = np.mean(seq, axis=0)
            else:
                agg_vector = seq
            
            aggregated.append(agg_vector)
        
        aggregated_matrix = np.array(aggregated)
        logger.info(f"Aggregated {len(sequences)} sequences to {aggregated_matrix.shape}")
        
        return aggregated_matrix
    
    def get_split_indices(self, n_samples: int,
                         train_ratio: float = 0.60,
                         val_ratio: float = 0.20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get indices for train/val/test split.
        
        Args:
            n_samples (int): Total number of samples.
            train_ratio (float): Training ratio.
            val_ratio (float): Validation ratio.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Index arrays for each split.
        """
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_samples)
        
        return train_idx, val_idx, test_idx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config import Config
    from log_parser import DrainLogParser
    from feature_extractor import LogFeatureExtractor
    
    print("Preparing HDFS data...")
    parser = DrainLogParser(depth=Config.DRAIN_DEPTH)
    hdfs_ids, hdfs_templates = parser.parse_file(Config.HDFS_LOG_PATH)
    
    extractor = LogFeatureExtractor(max_features=Config.TFIDF_MAX_FEATURES)
    hdfs_features = extractor.fit_transform(hdfs_templates)
    
    prep = DataPreparation(random_state=Config.RANDOM_STATE)
    train_idx, val_idx, test_idx = prep.get_split_indices(
        len(hdfs_features),
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )
    
    hdfs_train = hdfs_features[train_idx]
    hdfs_val = hdfs_features[val_idx]
    hdfs_test = hdfs_features[test_idx]
    
    print(f"HDFS split shapes:")
    print(f"  Training:   {hdfs_train.shape}")
    print(f"  Validation: {hdfs_val.shape}")
    print(f"  Testing:    {hdfs_test.shape}")
    
    print("\nPreparing BGL data...")
    parser = DrainLogParser(depth=Config.DRAIN_DEPTH)
    bgl_ids, bgl_templates = parser.parse_file(Config.BGL_LOG_PATH)
    
    extractor = LogFeatureExtractor(max_features=Config.TFIDF_MAX_FEATURES)
    bgl_features = extractor.fit_transform(bgl_templates)
    
    train_idx, val_idx, test_idx = prep.get_split_indices(
        len(bgl_features),
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )
    
    bgl_train = bgl_features[train_idx]
    bgl_val = bgl_features[val_idx]
    bgl_test = bgl_features[test_idx]
    
    print(f"BGL split shapes:")
    print(f"  Training:   {bgl_train.shape}")
    print(f"  Validation: {bgl_val.shape}")
    print(f"  Testing:    {bgl_test.shape}")