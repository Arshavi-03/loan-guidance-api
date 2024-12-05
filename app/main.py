from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import joblib
import boto3
import os
from .loan_guidance import AdvancedLoanGuidanceSystem

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

# Global variables
guidance_system = None
model_loading = False

def get_model():
    """Lazy load the model only when needed"""
    global guidance_system, model_loading
    
    if guidance_system is not None:
        return guidance_system
        
    if model_loading:
        return None
        
    try:
        model_loading = True
        # First create a basic model
        guidance_system = AdvancedLoanGuidanceSystem()
        
        # Try to load from S3 if available
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_DEFAULT_REGION')
            )
            
            s3_client.download_file(
                'virtual-herbal-garden-3d-models',
                'models/advanced_loan_guidance_system.joblib',
                '/tmp/model.joblib'
            )
            guidance_system = joblib.load('/tmp/model.joblib')
        except Exception as e:
            print(f"Warning: Could not load model from S3: {e}")
            print("Using default model initialization")
            
        return guidance_system
    except Exception as e:
        print(f"Error in get_model: {e}")
        return None
    finally:
        model_loading = False

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
            model = AdvancedLoanGuidanceSystem()
        
        guidance = model.generate_comprehensive_guidance(request.dict())
        return guidance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    model = get_model()
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "initializing"
    }