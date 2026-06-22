"""
XHAD Configuration Module
Production-level configuration management for XHAD framework.
"""

import os
from pathlib import Path

class Config:
    """Central configuration class for XHAD pipeline."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    CODE_DIR = PROJECT_ROOT / 'code'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # Data file paths
    HDFS_LOG_PATH = DATA_DIR / 'HDFS_sample.log'
    BGL_LOG_PATH = DATA_DIR / 'BGL_sample.log'
    
    # Log parsing configuration
    DRAIN_DEPTH = 4
    DRAIN_SIMILARITY_THRESHOLD = 0.5
    
    # Feature extraction
    TFIDF_MAX_FEATURES = 200
    TFIDF_MIN_DF = 1
    TFIDF_MAX_DF = 1.0
    
    # Data partitioning
    TRAIN_RATIO = 0.60
    VAL_RATIO = 0.20
    TEST_RATIO = 0.20
    
    # Random state for reproducibility
    RANDOM_STATE = 42
    
    # Isolation Forest parameters
    IF_N_ESTIMATORS = 100
    IF_MAX_SAMPLES = 256
    IF_CONTAMINATION = 'auto'
    
    # Autoencoder parameters
    AE_INPUT_DIM = 200
    AE_BOTTLENECK_DIM = 32
    AE_HIDDEN_DIMS = [128, 64]
    AE_EPOCHS = 50
    AE_BATCH_SIZE = 256
    AE_LEARNING_RATE = 0.001
    AE_VALIDATION_SPLIT = 0.2
    AE_EARLY_STOP_PATIENCE = 5
    
    # Ensemble parameters
    ENSEMBLE_ALPHA_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # SHAP parameters
    SHAP_N_BACKGROUND_SAMPLES = 100
    SHAP_N_SAMPLES = 2048
    
    # Evaluation parameters
    N_TRIALS = 5
    RANDOM_SEEDS = [42, 123, 456, 789, 999]
    
    # Performance targets
    F1_TARGET = 0.85
    AUC_TARGET = 0.90
    PRECISION_TARGET = 0.80
    RECALL_TARGET = 0.80
    
    @classmethod
    def verify_paths(cls):
        """Verify all required directories and files exist."""
        required_dirs = [cls.DATA_DIR, cls.CODE_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Required directory not found: {directory}")
        
        required_files = [cls.HDFS_LOG_PATH, cls.BGL_LOG_PATH]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        return True


if __name__ == "__main__":
    try:
        Config.verify_paths()
        print("Configuration verification successful.")
        print(f"Project root: {Config.PROJECT_ROOT}")
        print(f"Data directory: {Config.DATA_DIR}")
    except FileNotFoundError as e:
        print(f"Configuration error: {e}")