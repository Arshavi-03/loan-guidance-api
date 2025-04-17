import os
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from .utils import sanitize_html

class LoanSystem:
    """Wrapper for the joblib loan guidance system"""
    
    def __init__(self):
        """Initialize the loan system by loading the joblib model"""
        # Get the absolute path to the joblib file
        model_path = Path(__file__).parent.parent / "data" / "loan_guidance_system.joblib"
        
        # Load the loan guidance system
        self.loan_system = joblib.load(model_path)
        
        # Set up OpenAI API (if provided as environment variable)
        self._setup_openai_api()
    
    def _setup_openai_api(self):
        """Set up OpenAI API for recommendations if key is available"""
        # Check if API key is in environment variables
        if "OPENAI_API_KEY" in os.environ:
            # OpenAI will automatically use the key from environment variables
            self.openai_available = True
        else:
            # Log that OpenAI is not available
            print("Warning: OpenAI API key not found. AI recommendations will use fallback mode.")
            self.openai_available = False
            
    def analyze_loan(self, income, loan_amount, loan_term_years, interest_rate, 
                    credit_score, monthly_debt, property_value=None, extra_payment=0):
        """
        Analyze a loan scenario and return comprehensive results
        """
        # Validate and normalize inputs
        income = float(income)
        loan_amount = float(loan_amount)
        loan_term_years = int(loan_term_years)
        interest_rate = float(interest_rate)
        credit_score = int(credit_score)
        monthly_debt = float(monthly_debt)
        
        if property_value and property_value.strip() if isinstance(property_value, str) else property_value:
            property_value = float(property_value)
        else:
            property_value = None
        
        extra_payment = float(extra_payment) if extra_payment else 0
        
        # Run the loan analysis using the loaded model
        results = self.loan_system.analyze_loan_scenario(
            income=income,
            loan_amount=loan_amount,
            loan_term_years=loan_term_years,
            interest_rate=interest_rate,
            credit_score=credit_score,
            monthly_debt=monthly_debt,
            property_value=property_value,
            extra_payment=extra_payment
        )
        
        # Extract recommendations (processing HTML to plain text if needed)
        ai_recommendations = results["ai_recommendations"]
        if ai_recommendations and "<" in ai_recommendations:
            # If recommendations contain HTML, sanitize it
            recommendations = sanitize_html(ai_recommendations)
        else:
            recommendations = ai_recommendations
        
        # Create response object
        response = {
            "analysis": results["analysis"],
            "risk": results["risk"],
            "schedule_summary": self._get_summary_schedule(results["schedule"]),
            "visualization_available": True,
            "recommendations": recommendations
        }
        
        return response
    
    def get_visualization(self, loan_amount, interest_rate, loan_term_years, extra_payment=0):
        """
        Generate basic visualization for a loan scenario
        """
        # Convert parameters to the right types
        loan_amount = float(loan_amount)
        interest_rate = float(interest_rate)
        loan_term_years = int(loan_term_years)
        extra_payment = float(extra_payment) if extra_payment else 0
        
        # Generate visualization using the model
        visualization = self.loan_system.create_repayment_visualization(
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            loan_term_years=loan_term_years,
            extra_payment=extra_payment
        )
        
        return visualization
    
    def get_enhanced_visualization(self, loan_amount, interest_rate, loan_term_years, extra_payment=0):
        """
        Generate enhanced visualization for a loan scenario
        """
        # Convert parameters to the right types
        loan_amount = float(loan_amount)
        interest_rate = float(interest_rate)
        loan_term_years = int(loan_term_years)
        extra_payment = float(extra_payment) if extra_payment else 0
        
        # Generate enhanced visualization using the model
        visualization = self.loan_system.create_enhanced_repayment_visualization(
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            loan_term_years=loan_term_years,
            extra_payment=extra_payment
        )
        
        return visualization
    
    def get_payment_schedule(self, loan_amount, interest_rate, loan_term_years, extra_payment=0):
        """
        Generate monthly payment schedule for a loan
        """
        # Convert parameters to the right types
        loan_amount = float(loan_amount)
        interest_rate = float(interest_rate)
        loan_term_years = int(loan_term_years)
        extra_payment = float(extra_payment) if extra_payment else 0
        
        # Generate repayment schedule using the model
        schedule = self.loan_system.generate_repayment_schedule(
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            loan_term_years=loan_term_years,
            extra_payment=extra_payment
        )
        
        return schedule
    
    def get_recommendations(self, income, loan_amount, loan_term_years, interest_rate, 
                          credit_score, monthly_debt, property_value=None):
        """
        Get AI-powered recommendations for a loan scenario
        """
        # Run analysis to get needed metrics
        analysis = self.loan_system.analyze_loan_data(
            income=income,
            loan_amount=loan_amount,
            loan_term_years=loan_term_years,
            interest_rate=interest_rate,
            credit_score=credit_score,
            monthly_debt=monthly_debt,
            property_value=property_value
        )
        
        # Get risk assessment
        risk = self.loan_system.analyze_risk_factors(
            credit_score=credit_score,
            debt_to_income=analysis["debt_to_income"]["after_loan"],
            loan_to_value=analysis.get("loan_to_value"),
            property_type="single_family" if property_value else None
        )
        
        # Generate recommendations
        recommendations = self.loan_system.generate_personalized_recommendations(
            income=income,
            loan_amount=loan_amount,
            loan_term_years=loan_term_years,
            interest_rate=interest_rate,
            credit_score=credit_score,
            monthly_debt=monthly_debt,
            property_value=property_value,
            analysis_results=analysis,
            risk_assessment=risk
        )
        
        # Sanitize HTML if needed
        if recommendations and "<" in recommendations:
            recommendations = sanitize_html(recommendations)
            
        return recommendations
    
    def _get_summary_schedule(self, schedule):
        """
        Extract a summary of the payment schedule (first few payments and last payment)
        """
        if not schedule:
            return []
            
        # Keep first 3 payments, last payment, and summary
        summary = []
        for i, payment in enumerate(schedule):
            if i < 3 or payment.get("payment_number") == "summary" or i == len(schedule) - 2:
                summary.append(payment)
                
        return summary