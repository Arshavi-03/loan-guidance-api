from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import boto3
import joblib
import os
import logging
from typing import Dict, List
import pandas as pd
from .loan_guidance import AdvancedLoanGuidanceSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loan Guidance API",
    description="API for loan risk assessment and guidance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoanRequest(BaseModel):
    monthly_income: float
    loan_amount: float
    interest_rate: float
    loan_term_months: int
    credit_score: int
    age: int
    borrower_type: str
    sector_data: Dict
    payment_history: List[Dict] = []
    loan_status: str

# S3 client initialization
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION')
)

BUCKET_NAME = 'virtual-herbal-garden-3d-models'
MODEL_PATH = 'models/advanced_loan_guidance_system.joblib'  # Matches your S3 path
guidance_system = None

def load_model_from_s3():
    """Load the model from S3"""
    global guidance_system
    
    try:
        # Create temp directory
        os.makedirs('/tmp', exist_ok=True)
        
        logger.info("Downloading model from S3...")
        model_path = '/tmp/model.joblib'
        
        # Download the model using the correct path
        s3.download_file(
            BUCKET_NAME,
            MODEL_PATH,
            model_path
        )
        
        logger.info("Loading model...")
        guidance_system = joblib.load(model_path)
        
        # Verify model using test data
        logger.info("Testing model...")
        test_data = {
            'monthly_income': 5000,
            'loan_amount': 50000,
            'interest_rate': 8.5,
            'loan_term_months': 36,
            'credit_score': 720,
            'age': 30,
            'borrower_type': 'business',
            'sector_data': {
                'business': {
                    'years': 6,
                    'type': 'retail'
                }
            },
            'payment_history': [
                {
                    'due_date': '2023-01-01',
                    'payment_date': '2023-01-01',
                    'amount_paid': 1500
                }
            ],
            'loan_status': 'Charged Off'
        }
        
        _ = guidance_system.generate_comprehensive_guidance(test_data)
        logger.info("Model verified successfully")
        
        # Clean up
        os.remove(model_path)
        return guidance_system
        
    except Exception as e:
        logger.error(f"Error loading model from S3: {str(e)}")
        logger.info("Initializing new model...")
        return initialize_new_model()

def initialize_new_model():
    """Initialize a new model if S3 load fails"""
    model = AdvancedLoanGuidanceSystem()
    
    # Create consistent training data
    train_data = pd.DataFrame([
        {
            'monthly_income': 5000,
            'loan_amount': 50000,
            'interest_rate': 8.5,
            'loan_term_months': 36,
            'credit_score': 720,
            'age': 30,
            'borrower_type': 'business',
            'sector_data': {
                'business': {
                    'years': 6,
                    'type': 'retail'
                }
            },
            'payment_history': [
                {
                    'due_date': '2023-01-01',
                    'payment_date': '2023-01-01',
                    'amount_paid': 1500
                }
            ],
            'loan_status': 'Charged Off'
        },
        {
            'monthly_income': 5000,
            'loan_amount': 50000,
            'interest_rate': 8.5,
            'loan_term_months': 36,
            'credit_score': 720,
            'age': 30,
            'borrower_type': 'business',
            'sector_data': {
                'business': {
                    'years': 6,
                    'type': 'retail'
                }
            },
            'payment_history': [
                {
                    'due_date': '2023-01-01',
                    'payment_date': '2023-01-01',
                    'amount_paid': 1500
                }
            ],
            'loan_status': 'Current'
        }
    ])
    
    processed_data = model.preprocess_data(train_data)
    X = model.prepare_features(processed_data)
    y_risk = (processed_data['loan_status'] == 'Charged Off').astype(int)
    y_payment = train_data['loan_amount'] / train_data['loan_term_months']
    
    model.train_models(X, y_risk, y_payment)
    return model

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    try:
        global guidance_system
        guidance_system = load_model_from_s3()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "Loan Guidance API",
        "model_status": "loaded" if guidance_system is not None else "not initialized"
    }

@app.post("/analyze-loan")
async def analyze_loan(request: LoanRequest):
    """Process loan analysis request"""
    if not guidance_system:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        result = guidance_system.generate_comprehensive_guidance(request.dict())
        return result
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": guidance_system is not None,
        "environment": {
            "aws_region": os.environ.get('AWS_REGION'),
            "bucket": BUCKET_NAME,
            "model_path": MODEL_PATH
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)