# app/models.py
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Union, Optional
from datetime import datetime

class PaymentHistory(BaseModel):
    due_date: str
    payment_date: str
    amount_paid: float

    @validator('due_date', 'payment_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

class LoanRequest(BaseModel):
    monthly_income: float = Field(
        ..., 
        gt=0,
        description="Monthly income of the borrower in currency units"
    )
    loan_amount: float = Field(
        ..., 
        gt=0,
        description="Total loan amount requested"
    )
    interest_rate: float = Field(
        ..., 
        gt=0, 
        le=100,
        description="Annual interest rate as a percentage"
    )
    loan_term_months: int = Field(
        ..., 
        gt=0,
        le=360,  # 30 years maximum
        description="Loan term in months"
    )
    credit_score: int = Field(
        ..., 
        ge=300, 
        le=850,
        description="Credit score (300-850 range)"
    )
    age: int = Field(
        ..., 
        ge=18,
        description="Age of the borrower"
    )
    borrower_type: str = Field(
        ...,
        description="Type of borrower (e.g., 'student', 'business', 'farmer')"
    )
    sector_data: Dict[str, Dict[str, Union[str, int, float]]] = Field(
        ...,
        description="Sector-specific information for risk assessment"
    )
    payment_history: List[Dict[str, Union[str, float]]] = Field(
        default=[],
        description="Historical payment records if any"
    )

    @validator('borrower_type')
    def validate_borrower_type(cls, v):
        allowed_types = ['student', 'business', 'farmer']
        if v.lower() not in allowed_types:
            raise ValueError(f'borrower_type must be one of {allowed_types}')
        return v.lower()

    @validator('sector_data')
    def validate_sector_data(cls, v):
        allowed_sectors = ['student', 'business', 'farming']
        if not any(sector in v for sector in allowed_sectors):
            raise ValueError(f'sector_data must contain one of {allowed_sectors}')
        return v

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
                ]
            }
        }