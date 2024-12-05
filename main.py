# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# Load the model
MODEL_PATH = Path("app/advanced_loan_guidance_system.joblib")
try:
    guidance_system = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    guidance_system = None

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
        schema_extra = {
            "example": {
                "monthly_income": 5000,
                "loan_amount": 50000,
                "interest_rate": 5.5,
                "loan_term_months": 36,
                "credit_score": 720,
                "age": 30,
                "borrower_type": "business",
                "sector_data": {"business": {"years": 5, "type": "retail"}},
                "payment_history": [
                    {
                        "due_date": "2024-01-15",
                        "payment_date": "2024-01-15",
                        "amount_paid": 1500.00
                    }
                ]
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
    if guidance_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_loan(request: LoanRequest):
    """
    Generate loan guidance and risk assessment based on provided data
    """
    if guidance_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to dictionary
        loan_data = request.dict()
        
        # Convert to DataFrame with single row
        df = pd.DataFrame([loan_data])
        
        # Generate comprehensive guidance using the model
        guidance = guidance_system.generate_comprehensive_guidance(df)
        
        # Add timestamp and request summary
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "request_summary": {
                "loan_amount": request.loan_amount,
                "term_months": request.loan_term_months,
                "monthly_income": request.monthly_income
            },
            "guidance": guidance
        }
        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/batch-predict")
async def batch_predict(requests: List[LoanRequest], background_tasks: BackgroundTasks):
    """
    Process multiple loan requests in batch
    """
    if guidance_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert all requests to DataFrame
        loan_data_list = [request.dict() for request in requests]
        df = pd.DataFrame(loan_data_list)
        
        # Process each request
        results = []
        for _, row in df.iterrows():
            guidance = guidance_system.generate_comprehensive_guidance(pd.DataFrame([row]))
            results.append(guidance)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing batch request: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded model
    """
    if guidance_system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Advanced Loan Guidance System",
        "features_expected": [
            "monthly_income",
            "loan_amount",
            "interest_rate",
            "loan_term_months",
            "credit_score",
            "age",
            "borrower_type",
            "sector_data",
            "payment_history"
        ],
        "risk_levels": ["Low Risk", "Moderate Risk", "High Risk"],
        "supported_borrower_types": ["business", "student", "farmer"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        workers=1
    )