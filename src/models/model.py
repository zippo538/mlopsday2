from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Any, Type
from src.utils.logger import default_logger as logger
from src.utils.config import config

class ModelFactory:
    """Factory class for creating ML models"""
    
    @staticmethod
    def get_model_config() -> Dict[str, Dict[str, Any]]:
        """Get model configurations"""
        return {
            'decision_tree': {
                'class': DecisionTreeClassifier,
                'params': {
                    'max_depth': 5,
                    'min_samples_split': 2,
                    'random_state': 42
                }
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            }
        }
    
    @classmethod
    def create_model(cls, model_type: str) -> Any:
        """
        Create model instance
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        try:
            logger.info(f"Creating model of type: {model_type}")
            
            # Get model configurations
            model_configs = cls.get_model_config()
            
            if model_type not in model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Get model class and parameters
            model_info = model_configs[model_type]
            model_class = model_info['class']
            model_params = model_info['params']
            
            # Override parameters from config if provided
            config_params = config.get(f'model_params.{model_type}', {})
            model_params.update(config_params)
            
            # Create model instance
            model = model_class(**model_params)
            
            logger.info(f"Successfully created {model_type} model")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

class ChurnModel:
    """Base class for churn prediction models"""
    
    def __init__(self, model_type: str):
        """
        Initialize churn model
        
        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        logger.info(f"Initialized ChurnModel with type: {model_type}")
    
    def build(self) -> None:
        """Build model instance"""
        try:
            logger.info(f"Building {self.model_type} model")
            self.model = ModelFactory.create_model(self.model_type)
            logger.info("Model built successfully")
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.get_params()
    
    def save_feature_importance(self, feature_names) -> Dict[str, float]:
        """
        Save feature importance
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        try:
            importance = self.get_feature_importance()
            if importance is not None:
                self.feature_importance = dict(zip(feature_names, importance))
                return self.feature_importance
            return None
        except Exception as e:
            logger.error(f"Error saving feature importance: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        if self.model is None:
            raise ValueError("Model not built yet")
            
        try:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            else:
                logger.warning("Model doesn't support feature importance")
                return None
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise