"""
Data Pipeline Module
Unified orchestration of log parsing, feature extraction, and data preparation.
"""

import numpy as np
import logging
from typing import Tuple, Dict
from pathlib import Path

from config import Config
from log_parser import DrainLogParser
from feature_extractor import LogFeatureExtractor
from data_preparation import DataPreparation

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data pipeline for XHAD framework.
    Orchestrates log parsing, feature extraction, and data splitting.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config
        self.parser = None
        self.extractor = None
        self.prep = DataPreparation(random_state=self.config.RANDOM_STATE)
    
    def process_dataset(self, log_file: Path, dataset_name: str) -> Dict:
        """
        Process complete dataset through pipeline.
        
        Args:
            log_file (Path): Path to log file.
            dataset_name (str): Name of dataset (for logging).
            
        Returns:
            Dict: Dictionary containing train/val/test splits and metadata.
        """
        logger.info(f"Processing {dataset_name} dataset...")
        
        self.parser = DrainLogParser(
            depth=self.config.DRAIN_DEPTH,
            similarity_threshold=self.config.DRAIN_SIMILARITY_THRESHOLD
        )
        
        logger.info(f"Parsing {dataset_name} logs...")
        template_ids, templates = self.parser.parse_file(log_file)
        
        self.extractor = LogFeatureExtractor(
            max_features=self.config.TFIDF_MAX_FEATURES,
            min_df=self.config.TFIDF_MIN_DF,
            max_df=self.config.TFIDF_MAX_DF
        )
        
        logger.info(f"Extracting features for {dataset_name}...")
        features = self.extractor.fit_transform(templates)
        
        logger.info(f"Splitting {dataset_name} data...")
        train_idx, val_idx, test_idx = self.prep.get_split_indices(
            len(features),
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VAL_RATIO
        )
        
        result = {
            'name': dataset_name,
            'template_ids': template_ids,
            'templates': templates,
            'features': features,
            'train': {
                'features': features[train_idx],
                'indices': train_idx,
                'size': len(train_idx)
            },
            'val': {
                'features': features[val_idx],
                'indices': val_idx,
                'size': len(val_idx)
            },
            'test': {
                'features': features[test_idx],
                'indices': test_idx,
                'size': len(test_idx)
            },
            'statistics': {
                'n_samples': len(features),
                'n_features': features.shape[1],
                'unique_templates': len(set(template_ids)),
                'parser_stats': self.parser.get_statistics()
            }
        }
        
        logger.info(f"{dataset_name} processing complete")
        return result
    
    def process_all(self) -> Dict[str, Dict]:
        """
        Process all datasets in configuration.
        
        Returns:
            Dict: Dictionary with HDFS and BGL processed data.
        """
        datasets = {}
        
        hdfs_data = self.process_dataset(
            self.config.HDFS_LOG_PATH,
            'HDFS'
        )
        datasets['HDFS'] = hdfs_data
        
        bgl_data = self.process_dataset(
            self.config.BGL_LOG_PATH,
            'BGL'
        )
        datasets['BGL'] = bgl_data
        
        return datasets
    
    def print_summary(self, datasets: Dict[str, Dict]) -> None:
        """
        Print pipeline processing summary.
        
        Args:
            datasets (Dict): Processed datasets dictionary.
        """
        print("\n" + "="*70)
        print("DATA PIPELINE SUMMARY")
        print("="*70)
        
        for dataset_name, data in datasets.items():
            stats = data['statistics']
            print(f"\n{dataset_name}:")
            print(f"  Total samples: {stats['n_samples']}")
            print(f"  Features: {stats['n_features']}")
            print(f"  Unique templates: {stats['unique_templates']}")
            print(f"  Training set: {data['train']['size']} samples")
            print(f"  Validation set: {data['val']['size']} samples")
            print(f"  Test set: {data['test']['size']} samples")
            print(f"  Feature shape: {data['features'].shape}")
            print(f"  Feature range: [{data['features'].min():.3f}, {data['features'].max():.3f}]")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    Config.verify_paths()
    
    pipeline = DataPipeline(config=Config)
    datasets = pipeline.process_all()
    pipeline.print_summary(datasets)