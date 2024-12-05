from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import joblib
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
    loan_status: str

# Load model directly from local file
guidance_system = joblib.load('app/advanced_loan_guidance_system.joblib')

@app.post("/analyze-loan")
async def analyze_loan(request: LoanRequest):
    try:
        result = guidance_system.generate_comprehensive_guidance(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": guidance_system is not None}