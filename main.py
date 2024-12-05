from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Union
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configure logging
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

    class Config:
        json_schema_extra = {
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

def prepare_data_for_model(loan_data: dict) -> pd.DataFrame:
    """Prepare the data in the correct format for the model"""
    # Extract basic features
    basic_features = {
        'monthly_income': float(loan_data['monthly_income']),
        'loan_amount': float(loan_data['loan_amount']),
        'interest_rate': float(loan_data['interest_rate']),
        'loan_term_months': int(loan_data['loan_term_months']),
        'credit_score': int(loan_data['credit_score']),
        'age': int(loan_data['age']),
        'borrower_type': loan_data['borrower_type'],
        'sector_data': str(loan_data['sector_data']),
        'payment_history': str(loan_data['payment_history'])
    }
    
    # Create DataFrame with the correct shape
    df = pd.DataFrame([basic_features])
    
    return df

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Advanced Loan Guidance System API"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_loan(request: LoanRequest):
    try:
        # Convert request to dictionary and prepare data
        loan_data = request.dict()
        logger.info("Preparing data for model...")
        
        # Prepare data using the new function
        processed_df = prepare_data_for_model(loan_data)
        logger.info(f"Processed DataFrame shape: {processed_df.shape}")
        
        try:
            # Generate guidance with processed data
            guidance = guidance_system.generate_comprehensive_guidance(processed_df)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "request_summary": {
                    "loan_amount": loan_data["loan_amount"],
                    "term_months": loan_data["loan_term_months"]
                },
                "guidance": guidance
            }
        except Exception as e:
            logger.error(f"Error in guidance generation: {str(e)}")
            logger.error(f"DataFrame info: {processed_df.info()}")
            raise HTTPException(
                status_code=400,
                detail=f"Error in model processing: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)