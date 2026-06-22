"""
Isolation Forest Model
"""

import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """
    Isolation Forest based anomaly detector.
    
    Attributes:
        n_estimators (int): Number of isolation trees.
        max_samples (int): Number of samples per tree.
        contamination (str or float): Expected contamination rate.
        random_state (int): Random state for reproducibility.
    """
    
    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 contamination: str = 'auto', random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.is_trained = False
        self.training_time = None
    
    def train(self, X_train: np.ndarray) -> Dict:
        """
        Train Isolation Forest on normal data.
        
        Args:
            X_train (np.ndarray): Training feature matrix (normal logs only).
            
        Returns:
            Dict: Training statistics.
        """
        import time
        
        start_time = time.time()
        self.model.fit(X_train)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        
        logger.info(f"Isolation Forest training complete")
        logger.info(f"  Samples: {X_train.shape[0]}")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Training time: {self.training_time:.2f}s")
        
        return {
            'training_time': self.training_time,
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'n_trees': self.n_estimators
        }
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        Higher score = more anomalous.
        
        Args:
            X (np.ndarray): Feature matrix.
            
        Returns:
            np.ndarray: Anomaly scores in range [0, 1].
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        raw_scores = -self.model.score_samples(X)
        
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score - min_score > 0:
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(raw_scores)
        
        return normalized_scores
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict anomalies with threshold.
        
        Args:
            X (np.ndarray): Feature matrix.
            threshold (float): Decision threshold.
            
        Returns:
            np.ndarray: Binary predictions (1=anomaly, 0=normal).
        """
        scores = self.predict_anomaly_score(X)
        predictions = (scores >= threshold).astype(int)
        return predictions
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray,
                threshold: float = 0.5) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X (np.ndarray): Feature matrix.
            y_true (np.ndarray): Ground truth labels (1=anomaly, 0=normal).
            threshold (float): Decision threshold.
            
        Returns:
            Dict: Evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        scores = self.predict_anomaly_score(X)
        y_pred = self.predict(X, threshold=threshold)
        
        metrics = {
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, scores),
            'threshold': threshold
        }
        
        logger.info(f"Isolation Forest evaluation:")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    from config import Config
    from pipeline import DataPipeline
    
    Config.verify_paths()
    
    pipeline = DataPipeline(config=Config)
    datasets = pipeline.process_all()
    
    hdfs_data = datasets['HDFS']
    X_train = hdfs_data['train']['features']
    X_val = hdfs_data['val']['features']
    
    detector = IsolationForestAnomalyDetector(
        n_estimators=Config.IF_N_ESTIMATORS,
        max_samples=Config.IF_MAX_SAMPLES,
        random_state=Config.RANDOM_STATE
    )
    
    print("Training Isolation Forest...")
    detector.train(X_train)
    
    print("\nEvaluating on validation set...")
    scores = detector.predict_anomaly_score(X_val)
    print(f"Anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Mean anomaly score: {scores.mean():.4f}")
    print(f"Std anomaly score: {scores.std():.4f}")