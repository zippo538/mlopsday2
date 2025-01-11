# Telco Customer Churn Prediction

This project implements an end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. It includes data processing, model training, MLflow tracking, and model serving via FastAPI.

## Project Structure
```
telco_churn_prediction/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── data_processor.py   # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py           # Model architecture definitions
│   │   └── trainer.py         # Training logic
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py          # Logging configuration
│   │   └── config.py          # Configuration management
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI application
│       └── schemas.py         # Pydantic models
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_model.py
├── notebooks/
│   └── exploration.ipynb
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Features

- Data preprocessing pipeline with scikit-learn
- Multiple tree-based models (Decision Tree, Random Forest, Gradient Boosting)
- MLflow experiment tracking and model registry
- Model serving via FastAPI
- Comprehensive logging system
- Configuration management
- Production-ready code structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telco_churn_prediction.git
cd telco_churn_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Update `config/config.yaml` with your settings:
```yaml
# Data Configuration
data_path: "data/TelcoCustomerChurn.csv"
preprocessing_path: "models/preprocessing"

# MLflow Configuration
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "telco_churn_production"

# Model Parameters
model_params:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

## Usage

### Running the Pipeline

1. Place your data file in the data directory:
```bash
cp path/to/your/TelcoCustomerChurn.csv data/
```

2. Run the training pipeline:
```bash
python src/run_pipeline.py
```

3. View experiments in MLflow:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Starting the API

1. Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

- `POST /predict`: Make churn predictions
- `GET /health`: Health check endpoint
- `GET /model-info`: Get current model information

## Model Training

The pipeline trains three types of models:
- Decision Tree
- Random Forest
- Gradient Boosting

The best model is selected based on a combination of recall and F1 score, and is automatically registered in MLflow for production use.

## Monitoring

- Logs are stored in the `logs/` directory
- MLflow tracking information is stored in `mlflow.db`
- Model artifacts are stored in `models/`

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Models

1. Update `src/models/model.py` with your new model configuration
2. Add model-specific parameters in `config.yaml`
3. Update the training pipeline if necessary

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
