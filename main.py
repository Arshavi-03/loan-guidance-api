from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Literal
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.loan_guidance import AdvancedLoanGuidanceSystem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
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
    loan_status: Literal["Active", "Current", "Late", "Default", "Charged Off"] = "Current"

def prepare_loan_data(loan_request: dict) -> pd.DataFrame:
    """Prepare loan data in the correct format"""
    processed_data = {
        'monthly_income': float(loan_request['monthly_income']),
        'loan_amount': float(loan_request['loan_amount']),
        'interest_rate': float(loan_request['interest_rate']),
        'loan_term_months': int(loan_request['loan_term_months']),
        'credit_score': int(loan_request['credit_score']),
        'age': int(loan_request['age']),
        'borrower_type': str(loan_request['borrower_type']),
        'sector_data': str(loan_request['sector_data']),
        'payment_history': str(loan_request['payment_history']),
        'loan_status': str(loan_request['loan_status'])
    }
    return pd.DataFrame([processed_data])

@app.post("/predict")
async def predict_loan(request: LoanRequest):
    try:
        # Convert request to dictionary and preprocess
        loan_data = prepare_loan_data(request.dict())
        
        # First, preprocess the data using the model's preprocess_data method
        processed_data = guidance_system.preprocess_data(loan_data)
        
        # Generate guidance using the preprocessed data
        guidance = guidance_system.generate_comprehensive_guidance(processed_data)
        
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
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "message": "Loan Guidance API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)