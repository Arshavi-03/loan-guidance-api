# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Optional
import uvicorn
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from app.loan_guidance import AdvancedLoanGuidanceSystem  # Import from your module

app = FastAPI(
    title="Advanced Loan Guidance System API",
    description="API for loan risk assessment and guidance using machine learning",
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

# Initialize the model
try:
    MODEL_PATH = Path("app/advanced_loan_guidance_system.joblib")
    if MODEL_PATH.exists():
        guidance_system = joblib.load(MODEL_PATH)
    else:
        print(f"Model file not found at {MODEL_PATH}. Initializing new model.")
        guidance_system = AdvancedLoanGuidanceSystem()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Initializing new model instance.")
    guidance_system = AdvancedLoanGuidanceSystem()

class LoanRequest(BaseModel):
    monthly_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    interest_rate: float = Field(..., gt=0, le=100)
    loan_term_months: int = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    age: int = Field(..., ge=18)
    borrower_type: str
    sector_data: Dict[str, Dict[str, Union[str, int, float]]]
    payment_history: List[Dict[str, Union[str, float]]] = []

    class Config:
        json_schema_extra = {  # Updated from schema_extra to json_schema_extra
            "example": {
                "monthly_income": 5000,
                "loan_amount": 50000,
                "interest_rate": 5.5,
                "loan_term_months": 36,
                "credit_score": 720,
                "age": 30,
                "borrower_type": "business",
                "sector_data": {"business": {"years": 5, "type": "retail"}},
                "payment_history": []
            }
        }

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {
        "status": "online",
        "message": "Advanced Loan Guidance System API",
        "model_status": "loaded" if guidance_system is not None else "not loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_loan(request: LoanRequest):
    """Generate loan guidance and risk assessment"""
    try:
        # Convert request to dictionary
        loan_data = request.dict()
        
        # Convert to DataFrame with single row
        df = pd.DataFrame([loan_data])
        
        # Generate guidance
        guidance = guidance_system.generate_comprehensive_guidance(df)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "request_summary": {
                "loan_amount": request.loan_amount,
                "term_months": request.loan_term_months,
                "monthly_income": request.monthly_income
            },
            "guidance": guidance
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )