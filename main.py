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

# Initialize model with training
def init_model():
    model = AdvancedLoanGuidanceSystem()
    # Create a small training dataset
    training_data = pd.DataFrame({
        'monthly_income': [5000, 6000],
        'loan_amount': [50000, 60000],
        'interest_rate': [5.5, 6.0],
        'loan_term_months': [36, 48],
        'credit_score': [720, 700],
        'age': [30, 35],
        'borrower_type': ['business', 'student'],
        'sector_data': [
            str({'business': {'years': 5, 'type': 'retail'}}),
            str({'student': {'course_type': 'engineering'}})
        ],
        'payment_history': ['[]', '[]'],
        'loan_status': ['Current', 'Late']
    })
    
    # Process training data
    processed_data = model.preprocess_data(training_data)
    X = model.prepare_features(processed_data)
    y_risk = np.array([0, 1])  # Binary risk labels
    y_payment = training_data['loan_amount'] / training_data['loan_term_months']
    
    # Train the model
    model.train_models(X, y_risk, y_payment)
    return model

# Initialize the model
guidance_system = init_model()

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

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Loan Guidance API"
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
        # Convert request to dictionary and format data
        loan_data = {
            'monthly_income': request.monthly_income,
            'loan_amount': request.loan_amount,
            'interest_rate': request.interest_rate,
            'loan_term_months': request.loan_term_months,
            'credit_score': request.credit_score,
            'age': request.age,
            'borrower_type': request.borrower_type,
            'sector_data': str(request.sector_data),  # Convert to string as expected by model
            'payment_history': str(request.payment_history),  # Convert to string
            'loan_status': request.loan_status
        }
        
        # Create DataFrame
        df = pd.DataFrame([loan_data])
        
        # Generate guidance
        try:
            guidance = guidance_system.generate_comprehensive_guidance(df)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "request_summary": {
                    "loan_amount": request.loan_amount,
                    "term_months": request.loan_term_months,
                    "monthly_income": request.monthly_income,
                    "loan_status": request.loan_status
                },
                "guidance": guidance
            }
            
        except Exception as e:
            logger.error(f"Model prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")
            
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Request processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)