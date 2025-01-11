import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataProcessor:
    """Data preprocessing pipeline"""
    
    def __init__(self, preprocessing_path: Optional[str] = None):
        """
        Initialize data processor
        
        Args:
            preprocessing_path: Path to save/load preprocessing objects
        """
        self.preprocessing_path = preprocessing_path or config.get('preprocessing_path', 'models/preprocessing')
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.trained = False
        logger.info("Initialized DataProcessor")
    
    def _prepare_preprocessing_path(self) -> None:
        """Create preprocessing directory if it doesn't exist"""
        Path(self.preprocessing_path).mkdir(parents=True, exist_ok=True)
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit preprocessors and transform data
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of transformed features and target
        """
        try:
            logger.info("Starting fit_transform process")
            
            # Handle TotalCharges
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
            
            # Split features and target
            X = df.drop(columns=['customerID', target_col], errors='ignore')
            y = df[target_col] if target_col in df.columns else None
            
            # Process categorical columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                logger.info(f"Fitting LabelEncoder for column: {col}")
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col])
            
            # Process numerical columns
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            logger.info("Fitting StandardScaler for numerical columns")
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            
            self.trained = True
            logger.info("Fit_transform completed successfully")
            
            if y is not None:
                y = LabelEncoder().fit_transform(y)
            return X, y
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessors
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.trained:
            raise ValueError("DataProcessor not fitted. Call fit_transform first.")
            
        try:
            logger.info("Starting transform process")
            
            # Create a copy to avoid modifying original data
            X = df.copy()
            
            # Remove customerID if present
            if 'customerID' in X.columns:
                X = X.drop('customerID', axis=1)
            
            # Handle TotalCharges
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            X['TotalCharges'].fillna(X['MonthlyCharges'], inplace=True)
            
            # Transform categorical columns
            for col, encoder in self.encoders.items():
                X[col] = encoder.transform(X[col])
            
            # Transform numerical columns
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
            
            logger.info("Transform completed successfully")
            return X
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
    
    def save_preprocessors(self) -> None:
        """Save preprocessor objects"""
        try:
            logger.info(f"Saving preprocessors to {self.preprocessing_path}")
            self._prepare_preprocessing_path()
            
            # Save encoders
            joblib.dump(
                self.encoders,
                Path(self.preprocessing_path) / 'encoders.joblib'
            )
            
            # Save scaler
            joblib.dump(
                self.scaler,
                Path(self.preprocessing_path) / 'scaler.joblib'
            )
            
            logger.info("Preprocessors saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise
    
    def load_preprocessors(self) -> None:
        """Load preprocessor objects"""
        try:
            logger.info(f"Loading preprocessors from {self.preprocessing_path}")
            
            # Load encoders
            encoders_path = Path(self.preprocessing_path) / 'encoders.joblib'
            self.encoders = joblib.load(encoders_path)
            
            # Load scaler
            scaler_path = Path(self.preprocessing_path) / 'scaler.joblib'
            self.scaler = joblib.load(scaler_path)
            
            self.trained = True
            logger.info("Preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise