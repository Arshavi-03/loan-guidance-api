from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class LoanInputData(BaseModel):
    """Input model for loan analysis"""
    income: float = Field(..., description="Annual income in dollars", example=85000)
    loan_amount: float = Field(..., description="Loan amount in dollars", example=325000)
    loan_term: int = Field(..., description="Loan term in years", example=30)
    interest_rate: float = Field(..., description="Annual interest rate (percentage)", example=6.5)
    credit_score: int = Field(..., description="Credit score (300-850)", example=720)
    monthly_debt: float = Field(..., description="Monthly debt payments excluding mortgage", example=1200)
    property_value: Optional[float] = Field(None, description="Property value in dollars (optional)", example=400000)
    extra_payment: Optional[float] = Field(0, description="Extra monthly payment in dollars", example=0)

class RiskFactor(BaseModel):
    """Model for individual risk factor assessment"""
    risk_level: str
    impact: str
    suggestion: str

class DebtToIncome(BaseModel):
    """Model for debt-to-income ratio details"""
    before_loan: float
    after_loan: float
    category: str

class CreditScore(BaseModel):
    """Model for credit score details"""
    value: int
    category: str

class AnalysisResult(BaseModel):
    """Model for core analysis results"""
    monthly_payment: float
    total_interest: float
    total_payments: float
    debt_to_income: DebtToIncome
    loan_to_income: float
    payment_to_income: float
    loan_to_value: Optional[float] = None
    credit_score: CreditScore
    debt_service_coverage_ratio: float

class RiskAssessment(BaseModel):
    """Model for risk assessment"""
    risk_factors: Dict[str, RiskFactor]
    overall_risk: str
    recommendations: List[str]

class PaymentEntry(BaseModel):
    """Model for a payment schedule entry"""
    payment_number: Any  # Could be int or string for summary
    payment_date: Optional[str] = None
    payment_amount: float
    principal_payment: float
    interest_payment: float
    remaining_balance: float
    total_interest_paid: float
    years_to_payoff: Optional[float] = None
    months_to_payoff: Optional[int] = None
    
class LoanResponse(BaseModel):
    """Response model for loan analysis"""
    analysis: AnalysisResult
    risk: RiskAssessment
    schedule_summary: List[PaymentEntry]
    visualization_available: bool = True
    recommendations: str

class VisualizationResponse(BaseModel):
    """Response model for visualization endpoints"""
    image_data: str = Field(..., description="Base64 encoded PNG image data")