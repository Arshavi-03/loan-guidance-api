from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Literal
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os
from xgboost import XGBClassifier, XGBRegressor

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

# Initialize model
guidance_system = AdvancedLoanGuidanceSystem()

@app.get("/")
async def root():
    return {"status": "online", "message": "Loan Guidance API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_loan(request: LoanRequest):
    try:
        # Create the loan data dictionary
        loan_dict = request.dict()
        loan_dict['sector_data'] = str(loan_dict['sector_data'])
        loan_dict['payment_history'] = str(loan_dict['payment_history'])
        
        # Create DataFrame
        loan_df = pd.DataFrame([loan_dict])
        
        try:
            result = {
                'risk_assessment': {
                    'risk_level': 'Moderate Risk',
                    'risk_score': 0.5,
                    'key_factors': guidance_system.identify_risk_factors(loan_df.iloc[0]),
                    'mitigation_strategies': guidance_system.get_risk_mitigation_strategies('Moderate Risk')
                },
                'payment_plan': guidance_system.generate_payment_plan(loan_df.iloc[0], 
                                                                    loan_df['loan_amount'].iloc[0] / loan_df['loan_term_months'].iloc[0]),
                'recommendations': guidance_system.generate_smart_recommendations(loan_df.iloc[0], 0.5),
                'monitoring_plan': guidance_system.generate_monitoring_plan(0.5)
            }
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "request_summary": {
                    "loan_amount": request.loan_amount,
                    "term_months": request.loan_term_months,
                    "monthly_income": request.monthly_income
                },
                "guidance": result
            }
        except Exception as e:
            logger.error(f"Error in guidance generation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)