from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import joblib
import boto3
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from loan_guidance import AdvancedLoanGuidanceSystem

app = FastAPI(
    title="Loan Guidance System API",
    description="API for advanced loan guidance and risk assessment",
    version="1.0.0"
)

# Add CORS middleware
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

# Global variable for guidance system
guidance_system = None

def load_model():
    try:
        # Download model from S3
        s3_client.download_file(
            'virtual-herbal-garden-3d-models',
            'models/advanced_loan_guidance_system.joblib',
            '/tmp/model.joblib'
        )
        return joblib.load('/tmp/model.joblib')
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        # Initialize a new model if loading fails
        return AdvancedLoanGuidanceSystem()

# Load model on startup
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)