from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
from src.utils.logger import default_logger as logger
from src.data.data_processor import DataProcessor

app = FastAPI(title="Telco Churn Prediction API")

class ChurnPredictionRequest(BaseModel):
    """Churn prediction request model"""
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class ChurnPredictionResponse(BaseModel):
    """Churn prediction response model"""
    customerID: str
    churn_probability: float
    churn_prediction: bool

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, preprocessor
    try:
        logger.info("Loading best model from MLflow")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Load the best model (you should implement logic to select best model)
        model = mlflow.pyfunc.load_model(
            model_uri="models:/telco_churn_random_forest/Production"
        )
        
        # Initialize preprocessor
        preprocessor = DataProcessor()
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Telco Churn Prediction API"}

@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict(request: ChurnPredictionRequest):
    """
    Make churn prediction
    
    Args:
        request: Prediction request
        
    Returns:
        Prediction response
    """
    try:
        logger.info(f"Received prediction request for customer: {request.customerID}")
        
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Preprocess data
        processed_data = preprocessor.preprocess(data, training=False)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)
        churn_probability = prediction_proba[0][1]
        churn_prediction = bool(churn_probability >= 0.5)
        
        response = ChurnPredictionResponse(
            customerID=request.customerID,
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction
        )
        
        logger.info(f"Prediction completed for customer: {request.customerID}")
        return response
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}