import os
import time
import boto3
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loan_guidance import AdvancedLoanGuidanceSystem

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

# Global variables
guidance_system = None

def get_model():
    """Get model instance"""
    global guidance_system
    
    if guidance_system is not None:
        return guidance_system
        
    try:
        # Create new model instance
        guidance_system = AdvancedLoanGuidanceSystem()
        
        try:
            print("Attempting to load model from S3...")
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION')
            )
            
            # Download model
            local_path = '/tmp/model.joblib'
            s3_client.download_file(
                'virtual-herbal-garden-3d-models',
                'models/advanced_loan_guidance_system.joblib',
                local_path
            )
            
            # Load model
            guidance_system = joblib.load(local_path)
            print("Successfully loaded model from S3")
            
        except Exception as e:
            print(f"Warning: Could not load model from S3: {e}")
            print("Training new model instance...")
            # Create minimal training data
            train_data = pd.DataFrame([
                {
                    'monthly_income': 5000,
                    'loan_amount': 50000,
                    'interest_rate': 10.5,
                    'loan_term_months': 24,
                    'credit_score': 700,
                    'age': 35,
                    'borrower_type': 'business',
                    'loan_status': 'Charged Off',
                    'sector_data': {'business': {'years': 5, 'type': 'retail'}},
                    'payment_history': []
                },
                {
                    'monthly_income': 6000,
                    'loan_amount': 40000,
                    'interest_rate': 9.5,
                    'loan_term_months': 36,
                    'credit_score': 750,
                    'age': 40,
                    'borrower_type': 'business',
                    'loan_status': 'Current',
                    'sector_data': {'business': {'years': 8, 'type': 'service'}},
                    'payment_history': []
                }
            ])
            
            # Process and train
            processed_data = guidance_system.preprocess_data(train_data)
            X = guidance_system.prepare_features(processed_data)
            y_risk = (processed_data['loan_status'] == 'Charged Off').astype(int)
            y_payment = train_data['loan_amount'] / train_data['loan_term_months']
            
            guidance_system.train_models(X, y_risk, y_payment)
            print("New model instance trained successfully")
            
        return guidance_system
    except Exception as e:
        print(f"Error in get_model: {e}")
        return None

@app.get("/")
async def root():
    return {
        "status": "active",
        "message": "Loan Guidance System API is running"
    }

@app.post("/analyze-loan")
async def analyze_loan(request: LoanRequest):
    try:
        model = get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")
        
        guidance = model.generate_comprehensive_guidance(request.dict())
        return guidance
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    model = get_model()
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "initializing",
        "timestamp": time.time()
    }