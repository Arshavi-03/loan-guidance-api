from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import base64
from typing import Optional

from .models import LoanInputData, LoanResponse, VisualizationResponse
from .loan_system import LoanSystem

# Initialize FastAPI app
app = FastAPI(
    title="Loan Guidance System API",
    description="API for analyzing loan scenarios and providing financial guidance",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize loan system
loan_system = LoanSystem()

@app.get("/")
async def root():
    return {"message": "Welcome to the Loan Guidance System API"}

@app.post("/analyze", response_model=LoanResponse)
async def analyze_loan(data: LoanInputData):
    """
    Analyze a loan scenario and provide comprehensive guidance.
    Returns full analysis with recommendations.
    """
    try:
        # Process the loan data
        result = loan_system.analyze_loan(
            income=data.income,
            loan_amount=data.loan_amount,
            loan_term_years=data.loan_term,
            interest_rate=data.interest_rate,
            credit_score=data.credit_score,
            monthly_debt=data.monthly_debt,
            property_value=data.property_value,
            extra_payment=data.extra_payment
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing loan: {str(e)}")

@app.post("/visualization")
async def get_visualization(data: LoanInputData):
    """
    Generate visualization for a loan scenario.
    Returns base64 encoded image data.
    """
    try:
        # Generate visualization
        visualization = loan_system.get_visualization(
            loan_amount=data.loan_amount,
            interest_rate=data.interest_rate,
            loan_term_years=data.loan_term,
            extra_payment=data.extra_payment
        )
        
        return VisualizationResponse(image_data=visualization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.post("/enhanced-visualization")
async def get_enhanced_visualization(data: LoanInputData):
    """
    Generate enhanced visualization for a loan scenario.
    Returns base64 encoded image data with multiple charts.
    """
    try:
        # Generate enhanced visualization
        visualization = loan_system.get_enhanced_visualization(
            loan_amount=data.loan_amount,
            interest_rate=data.interest_rate,
            loan_term_years=data.loan_term,
            extra_payment=data.extra_payment
        )
        
        return VisualizationResponse(image_data=visualization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating enhanced visualization: {str(e)}")

@app.post("/payment-schedule")
async def get_payment_schedule(data: LoanInputData):
    """
    Generate a monthly payment schedule for a loan.
    """
    try:
        # Generate payment schedule
        schedule = loan_system.get_payment_schedule(
            loan_amount=data.loan_amount,
            interest_rate=data.interest_rate,
            loan_term_years=data.loan_term,
            extra_payment=data.extra_payment
        )
        
        return {"schedule": schedule}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating payment schedule: {str(e)}")

@app.post("/recommendations")
async def get_ai_recommendations(data: LoanInputData):
    """
    Get AI-powered personalized recommendations for a loan scenario.
    """
    try:
        # Get AI recommendations
        recommendations = loan_system.get_recommendations(
            income=data.income,
            loan_amount=data.loan_amount,
            loan_term_years=data.loan_term,
            interest_rate=data.interest_rate,
            credit_score=data.credit_score,
            monthly_debt=data.monthly_debt,
            property_value=data.property_value
        )
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}