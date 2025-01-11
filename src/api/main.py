from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
from datetime import datetime
from src.utils.logger import default_logger as logger
from src.data.data_processor import DataProcessor
from src.api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    ModelInfo,
    ModelMetrics
)
import pickle

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

def get_latest_model_path() -> tuple[str, ModelInfo]:
    """
    Get the path to the latest trained model
    
    Returns:
        Tuple containing model path and model info
    """
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name("telco_churn_experiment")
    if not experiment:
        raise ValueError("No experiment found with name 'telco_churn_experiment'")
    
    logger.info(f"Found experiment with ID: {experiment.experiment_id}")
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.recall DESC", "metrics.f1 DESC"]
    )
    
    logger.info(f"Found {len(runs)} runs")
    
    if not runs:
        raise ValueError("No runs found in the experiment")
    
    # Find best run based on average of recall and f1
    best_run = None
    best_score = -1
    
    for run in runs:
        metrics = run.data.metrics
        if 'recall' in metrics and 'f1' in metrics:
            combined_score = (metrics['recall'] + metrics['f1']) / 2
            logger.info(f"Run {run.info.run_id} score: {combined_score}")
            if combined_score > best_score:
                best_score = combined_score
                best_run = run
                
    if not best_run:
        raise ValueError("No valid runs found with required metrics")
    
    # Get model path
    run_id = best_run.info.run_id
    logger.info(f"Best run ID: {run_id}")
    
    # Try to load model by run ID
    try:
        # Load directly from run ID and model name
        model_path = f"runs:/{run_id}/gradient_boosting"
        logger.info(f"Trying to load model from: {model_path}")
        model = mlflow.pyfunc.load_model(model_path)
        logger.info(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Try alternative path with 'model' artifact name
        try:
            model_path = f"runs:/{run_id}/model"
            logger.info(f"Trying alternative path: {model_path}")
            model = mlflow.pyfunc.load_model(model_path)
            logger.info(f"Successfully loaded model from alternative path")
        except Exception as e2:
            logger.error(f"Error loading from alternative path: {str(e2)}")
            # Try local filesystem path
            try:
                local_path = os.path.join("mlruns", experiment.experiment_id, 
                                        run_id, "artifacts/model")
                logger.info(f"Trying local filesystem path: {local_path}")
                if not os.path.exists(local_path):
                    raise ValueError(f"Local path does not exist: {local_path}")
                model = mlflow.pyfunc.load_model(local_path)
                model_path = local_path
                logger.info(f"Successfully loaded model from local path")
            except Exception as e3:
                logger.error(f"Error loading from local path: {str(e3)}")
                raise ValueError(f"Could not load model from any path. Errors:\n"
                               f"Primary: {str(e)}\n"
                               f"Alternative: {str(e2)}\n"
                               f"Local: {str(e3)}")
    
    # Create model info
    metrics = ModelMetrics(
        accuracy=best_run.data.metrics.get('accuracy', 0.0),
        precision=best_run.data.metrics.get('precision', 0.0),
        recall=best_run.data.metrics.get('recall', 0.0),
        f1=best_run.data.metrics.get('f1', 0.0),
        roc_auc=best_run.data.metrics.get('roc_auc', 0.0)
    )
    
    model_info = ModelInfo(run_id=run_id, metrics=metrics)
    return model_path, model_info

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    global model, preprocessor, model_info
    try:
        logger.info("Loading best model from MLflow")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Find and load best model
        # model_path, model_info = get_latest_model_path()
        # logger.info(f"Loading model from path: {model_path}")
        # model = mlflow.pyfunc.load_model(model_path)

        # Load the model using the correct method
        model_path = 'mlruns/1/0b346ede29704e2bbb7a1c7c35b2c019/artifacts/random_forest/model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Initialize and load preprocessor
        preprocessor = DataProcessor()
        preprocessor.load_preprocessors()
        
        logger.info(f"Model and preprocessor loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Telco Customer Churn Prediction API",
        # "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict(request: ChurnPredictionRequest):
    """
    Predict customer churn
    
    Args:
        request: Prediction request containing customer data
        
    Returns:
        Prediction response with churn probability and model info
    """
    try:
        logger.info(f"Received prediction request for customer: {request.customerID}")
        
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Preprocess data
        processed_data = preprocessor.transform(data)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)
        churn_probability = prediction_proba[0][1]
        churn_prediction = bool(churn_probability >= 0.5)
        
        response = ChurnPredictionResponse(
            customerID=request.customerID,
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction,
            # model_info=model_info
        )
        
        logger.info(f"Prediction completed for customer: {request.customerID}")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info
    }