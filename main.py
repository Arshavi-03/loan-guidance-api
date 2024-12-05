from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Union
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the model class
from app.loan_guidance import AdvancedLoanGuidanceSystem

app = FastAPI(
    title="Advanced Loan Guidance System API",
    description="API for loan risk assessment and guidance using machine learning",
    version="1.0.0"
)

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
    loan_status: str = "Active"  # Added to match your model's requirements

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
                "sector_data": {
                    "business": {
                        "years": 5,
                        "type": "retail"
                    }
                },
                "payment_history": [
                    {
                        "due_date": "2024-01-15",
                        "payment_date": "2024-01-15",
                        "amount_paid": 1500.00
                    }
                ],
                "loan_status": "Active"
            }
        }

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Advanced Loan Guidance System API",
        "documentation": "/docs"
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
        # Convert request to dictionary
        loan_data = request.dict()
        logger.info("Processing loan request...")
        
        # Create DataFrame with the loan data
        df = pd.DataFrame([loan_data])
        
        try:
            # Generate guidance
            guidance = guidance_system.generate_comprehensive_guidance(df)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "request_summary": {
                    "loan_amount": loan_data["loan_amount"],
                    "term_months": loan_data["loan_term_months"],
                    "monthly_income": loan_data["monthly_income"]
                },
                "guidance": guidance
            }
            
        except Exception as e:
            logger.error(f"Error in guidance generation: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error in generating guidance: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)