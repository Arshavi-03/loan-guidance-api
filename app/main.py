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
from loan_guidance import AdvancedLoanGuidanceSystem  # Your original model class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION')
)

BUCKET_NAME = 'virtual-herbal-garden-3d-models'
MODEL_KEY = 'models/advanced_loan_guidance_system.joblib'
guidance_system = None

def load_model_from_s3():
    """Load the model from S3"""
    global guidance_system
    try:
        # Create temp directory
        os.makedirs('/tmp', exist_ok=True)
        local_path = '/tmp/model.joblib'
        
        # Download and load model
        logger.info(f"Downloading model from s3://{BUCKET_NAME}/{MODEL_KEY}")
        s3.download_file(BUCKET_NAME, MODEL_KEY, local_path)
        logger.info("Loading model from local file")
        guidance_system = joblib.load(local_path)
        
        # Clean up
        os.remove(local_path)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    if not load_model_from_s3():
        logger.error("Failed to load model from S3")
        raise Exception("Could not load model")

@app.get("/")
async def root():
    return {
        "status": "active",
        "message": "Loan Guidance System API is running"
    }

@app.post("/analyze-loan")
async def analyze_loan(request: LoanRequest):
    if not guidance_system:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        result = guidance_system.generate_comprehensive_guidance(request.dict())
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": guidance_system is not None,
        "environment": {
            "aws_region": os.environ.get('AWS_REGION'),
            "bucket": BUCKET_NAME,
            "model_key": MODEL_KEY
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)