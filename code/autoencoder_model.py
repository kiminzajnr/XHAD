"""
Autoencoder Model
"""

import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from typing import Tuple, Dict, Optional
import time

logger = logging.getLogger(__name__)


class DenseAutoencoder:
    """
    Deep autoencoder for log anomaly detection.
    
    Architecture: Input -> Encoder -> Bottleneck -> Decoder -> Output
    """
    
    def __init__(self, input_dim: int = 200, bottleneck_dim: int = 32,
                 hidden_dims: list = None, learning_rate: float = 0.001,
                 random_state: int = 42):
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.encoder = None
        self.history = None
        self.is_trained = False
        self.training_time = None
    
    def build_model(self) -> keras.Model:
        """
        Build autoencoder architecture.
        
        Returns:
            keras.Model: Compiled autoencoder model.
        """
        inputs = keras.Input(shape=(self.input_dim,))
        
        x = inputs
        for hidden_dim in self.hidden_dims:
            x = layers.Dense(hidden_dim, activation='relu')(x)
        
        bottleneck = layers.Dense(self.bottleneck_dim, activation='relu', name='bottleneck')(x)
        
        x = bottleneck
        for hidden_dim in reversed(self.hidden_dims):
            x = layers.Dense(hidden_dim, activation='relu')(x)
        
        outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        logger.info(f"Autoencoder model built")
        logger.info(f"  Architecture: {self.input_dim} -> {self.hidden_dims} -> {self.bottleneck_dim} -> {self.hidden_dims} -> {self.input_dim}")
        
        return self.model
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 256,
              validation_split: float = 0.2, early_stop_patience: int = 5) -> Dict:
        """
        Train autoencoder on normal data.
        
        Args:
            X_train (np.ndarray): Training feature matrix.
            X_val (np.ndarray): Validation feature matrix (optional).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            validation_split (float): Validation split ratio.
            early_stop_patience (int): Early stopping patience.
            
        Returns:
            Dict: Training statistics.
        """
        if self.model is None:
            self.build_model()
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stop_patience,
            restore_best_weights=True
        )
        
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        logger.info(f"Autoencoder training complete")
        logger.info(f"  Epochs trained: {len(self.history.history['loss'])}")
        logger.info(f"  Final training loss: {self.history.history['loss'][-1]:.6f}")
        logger.info(f"  Final validation loss: {self.history.history['val_loss'][-1]:.6f}")
        logger.info(f"  Training time: {self.training_time:.2f}s")
        
        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'training_time': self.training_time,
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1]
        }
    
    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for samples.
        
        Args:
            X (np.ndarray): Feature matrix.
            
        Returns:
            np.ndarray: Reconstruction errors.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        reconstructed = self.model.predict(X, verbose=0)
        reconstruction_error = np.mean(np.square(X - reconstructed), axis=1)
        
        return reconstruction_error
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores based on reconstruction error.
        Higher score = more anomalous.
        
        Args:
            X (np.ndarray): Feature matrix.
            
        Returns:
            np.ndarray: Normalized anomaly scores in range [0, 1].
        """
        errors = self.compute_reconstruction_error(X)
        
        min_error = errors.min()
        max_error = errors.max()
        
        if max_error - min_error > 0:
            normalized_scores = (errors - min_error) / (max_error - min_error)
        else:
            normalized_scores = np.zeros_like(errors)
        
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
            y_true (np.ndarray): Ground truth labels.
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
        
        logger.info(f"Autoencoder evaluation:")
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
    
    print("Building and training autoencoder...")
    ae = DenseAutoencoder(
        input_dim=Config.AE_INPUT_DIM,
        bottleneck_dim=Config.AE_BOTTLENECK_DIM,
        hidden_dims=Config.AE_HIDDEN_DIMS,
        learning_rate=Config.AE_LEARNING_RATE,
        random_state=Config.RANDOM_STATE
    )
    
    ae.train(
        X_train,
        epochs=Config.AE_EPOCHS,
        batch_size=Config.AE_BATCH_SIZE,
        validation_split=Config.AE_VALIDATION_SPLIT,
        early_stop_patience=Config.AE_EARLY_STOP_PATIENCE
    )
    
    print("\nEvaluating on validation set...")
    scores = ae.predict_anomaly_score(X_val)
    print(f"Anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Mean anomaly score: {scores.mean():.4f}")
    print(f"Std anomaly score: {scores.std():.4f}")