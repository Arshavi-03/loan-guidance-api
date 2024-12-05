from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import joblib
import boto3
import os

# Import from the same directory
from .loan_guidance import AdvancedLoanGuidanceSystem

app = FastAPI(
    title="Loan Guidance System API",
    description="API for advanced loan guidance and risk assessment",
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

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

guidance_system = None

def load_model():
    try:
        # Initialize a new model instance first
        model = AdvancedLoanGuidanceSystem()
        
        try:
            # Try to download from S3
            s3_client.download_file(
                'virtual-herbal-garden-3d-models',
                'models/advanced_loan_guidance_system.joblib',
                '/tmp/model.joblib'
            )
            model = joblib.load('/tmp/model.joblib')
        except Exception as e:
            print(f"Warning: Could not load model from S3: {e}")
            print("Using newly initialized model instead.")
        
        return model
    except Exception as e:
        print(f"Error in load_model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global guidance_system
    guidance_system = load_model()

@app.get("/")
async def root():
    return {"status": "active", "message": "Loan Guidance System API is running"}

@app.post("/analyze-loan")
async def analyze_loan(request: LoanRequest):
    try:
        if guidance_system is None:
            guidance_system = AdvancedLoanGuidanceSystem()
        guidance = guidance_system.generate_comprehensive_guidance(request.dict())
        return guidance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": guidance_system is not None
    }