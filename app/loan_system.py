# app/loan_system.py - Fixed version

import os
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

class LoanSystem:
    """Simple loan calculator system that doesn't require the joblib model"""
    
    def __init__(self):
        """Initialize the loan system with simple calculation logic"""
        self.openai_available = "OPENAI_API_KEY" in os.environ
        print(f"Loan System initialized with OpenAI: {self.openai_available}")
    
    def analyze_loan(self, income, loan_amount, loan_term_years, interest_rate, 
                    credit_score, monthly_debt, property_value=None, extra_payment=0):
        """Analyze a loan scenario with simple calculations"""
        # Validate inputs
        income = float(income)
        loan_amount = float(loan_amount)
        loan_term_years = int(loan_term_years)
        interest_rate = float(interest_rate)
        credit_score = int(credit_score)
        monthly_debt = float(monthly_debt)
        
        # Convert property_value to float if provided
        if property_value is not None:
            if isinstance(property_value, str) and property_value.strip():
                property_value = float(property_value)
            elif not isinstance(property_value, str):
                property_value = float(property_value) if property_value else None
            else:
                property_value = None
        
        # Calculate basic loan metrics
        monthly_rate = interest_rate / 100 / 12
        n_payments = loan_term_years * 12
        
        # Calculate monthly payment
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
        
        # Calculate total interest
        total_interest = (monthly_payment * n_payments) - loan_amount
        total_payments = monthly_payment * n_payments
        
        # Calculate debt-to-income ratio
        monthly_income = income / 12
        dti_before_loan = (monthly_debt / monthly_income) * 100
        dti_after_loan = ((monthly_debt + monthly_payment) / monthly_income) * 100
        
        # Determine DTI category
        dti_category = "critical"
        if dti_after_loan < 28:
            dti_category = "excellent"
        elif dti_after_loan < 36:
            dti_category = "good"
        elif dti_after_loan < 43:
            dti_category = "fair"
        elif dti_after_loan < 50:
            dti_category = "poor"
            
        # Calculate other ratios
        loan_to_income = (loan_amount / income) * 100
        payment_to_income = (monthly_payment / monthly_income) * 100
        ltv = (loan_amount / property_value) * 100 if property_value else None
        
        # Determine credit score category
        credit_category = "poor"
        if credit_score >= 740:
            credit_category = "excellent"
        elif credit_score >= 670:
            credit_category = "good"
        elif credit_score >= 580:
            credit_category = "fair"
            
        # Calculate debt service coverage ratio
        dscr = monthly_income / (monthly_debt + monthly_payment)
        
        # Risk assessment
        risk_level = "high" if dti_after_loan > 43 or credit_score < 580 else \
                   "moderate" if dti_after_loan > 36 or credit_score < 670 else \
                   "low_moderate" if dti_after_loan > 28 or credit_score < 740 else "low"
        
        # Generate recommendations
        recommendations = []
        if dti_after_loan > 43:
            recommendations.append("Your debt-to-income ratio is high. Consider reducing other debt or increasing income.")
        if credit_score < 670:
            recommendations.append("Work on improving your credit score to qualify for better interest rates.")
        if ltv and ltv > 80:
            recommendations.append("Consider making a larger down payment to reduce loan-to-value ratio and avoid PMI.")
        if extra_payment == 0:
            recommendations.append("Making extra payments could significantly reduce your total interest paid and loan term.")
        
        if not recommendations:
            recommendations.append("Your financial profile appears strong for this loan.")
        
        # Create risk factors
        risk_factors = {
            "credit_score": {
                "risk_level": "high" if credit_score < 580 else "moderate" if credit_score < 670 else "low",
                "impact": "negative" if credit_score < 670 else "positive",
                "suggestion": "Improve credit score" if credit_score < 670 else "Maintain excellent credit"
            },
            "debt_to_income": {
                "risk_level": "high" if dti_after_loan > 43 else "moderate" if dti_after_loan > 36 else "low",
                "impact": "negative" if dti_after_loan > 36 else "positive",
                "suggestion": "Reduce debt or increase income" if dti_after_loan > 36 else "Maintain healthy DTI ratio"
            }
        }
        
        # Create a simple payment schedule
        schedule = self._generate_simple_schedule(
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            loan_term_years=loan_term_years,
            extra_payment=extra_payment
        )
        
        # Format AI recommendations
        rec_text = f"""
        <h3>Loan Assessment</h3>
        <p>Based on your financial profile, this loan represents a {risk_level.replace('_', ' ')} risk. 
        Your debt-to-income ratio is {dti_after_loan:.1f}%, which is considered {dti_category}.</p>
        
        <h3>Recommendations</h3>
        <ul>
        {"".join(f"<li>{rec}</li>" for rec in recommendations)}
        </ul>
        
        <h3>Long-term Outlook</h3>
        <p>With a monthly payment of ${monthly_payment:.2f}, you'll pay a total of ${total_interest:.2f} in interest 
        over the {loan_term_years} year term. Making extra payments of ${extra_payment:.2f} per month could save you 
        significantly in interest costs.</p>
        """
        
        # Create response object
        response = {
            "analysis": {
                "monthly_payment": round(monthly_payment, 2),
                "total_interest": round(total_interest, 2),
                "total_payments": round(total_payments, 2),
                "debt_to_income": {
                    "before_loan": round(dti_before_loan, 2),
                    "after_loan": round(dti_after_loan, 2),
                    "category": dti_category
                },
                "loan_to_income": round(loan_to_income, 2),
                "payment_to_income": round(payment_to_income, 2),
                "loan_to_value": round(ltv, 2) if ltv else None,
                "credit_score": {
                    "value": credit_score,
                    "category": credit_category
                },
                "debt_service_coverage_ratio": round(dscr, 2)
            },
            "risk": {
                "risk_factors": risk_factors,
                "overall_risk": risk_level,
                "recommendations": recommendations
            },
            "schedule_summary": schedule[:3] + [schedule[-1]] if len(schedule) > 4 else schedule,
            "visualization_available": False,
            "recommendations": rec_text
        }
        
        return response
    
    def get_visualization(self, *args, **kwargs):
        """Return a placeholder for visualization"""
        # Return a simple 1x1 pixel transparent PNG in base64
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HgAGgwJ/lK3Q6wAAAABJRU5ErkJggg=="
    
    def get_enhanced_visualization(self, *args, **kwargs):
        """Return a placeholder for enhanced visualization"""
        return self.get_visualization(*args, **kwargs)
    
    def get_payment_schedule(self, loan_amount, interest_rate, loan_term_years, extra_payment=0):
        """Generate a payment schedule"""
        return self._generate_simple_schedule(
            loan_amount=float(loan_amount),
            interest_rate=float(interest_rate),
            loan_term_years=int(loan_term_years),
            extra_payment=float(extra_payment) if extra_payment else 0
        )
    
    def get_recommendations(self, *args, **kwargs):
        """Get simple recommendations"""
        analysis = self.analyze_loan(*args, **kwargs)
        return analysis["recommendations"]
    
    def _generate_simple_schedule(self, loan_amount, interest_rate, loan_term_years, extra_payment=0):
        """Generate a simple payment schedule"""
        schedule = []
        remaining_balance = loan_amount
        monthly_rate = interest_rate / 100 / 12
        n_payments = loan_term_years * 12
        
        # Calculate monthly payment
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
        
        # Initialize variables
        total_interest = 0
        payment_number = 1
        
        # Generate schedule (first few payments and last payment)
        num_payments_to_generate = min(12, n_payments)
        
        for i in range(1, num_payments_to_generate + 1):
            # Calculate interest for this period
            interest_payment = remaining_balance * monthly_rate
            
            # Calculate principal for this period
            principal_payment = monthly_payment - interest_payment
            
            # Add extra payment if specified
            total_principal_payment = principal_payment
            if extra_payment > 0:
                total_principal_payment += extra_payment
            
            # Update remaining balance
            remaining_balance -= total_principal_payment
            
            # Handle final payment or negative balance
            if remaining_balance < 0:
                total_principal_payment += remaining_balance
                remaining_balance = 0
            
            # Update total interest
            total_interest += interest_payment
            
            # Add payment to schedule
            payment = {
                "payment_number": i,
                "payment_date": f"2024-{((i-1)%12)+1:02d}-01",
                "payment_amount": round(monthly_payment + extra_payment, 2),
                "principal_payment": round(total_principal_payment, 2),
                "interest_payment": round(interest_payment, 2),
                "remaining_balance": round(remaining_balance, 2),
                "total_interest_paid": round(total_interest, 2)
            }
            
            schedule.append(payment)
            
            # Break if balance is zero
            if remaining_balance <= 0:
                break
        
        # Add last payment if we have a long loan
        if n_payments > 12 and remaining_balance > 0:
            # Calculate how many more payments based on remaining balance
            payments_left = math.ceil(math.log(monthly_payment / (monthly_payment - remaining_balance * monthly_rate)) / math.log(1 + monthly_rate))
            final_payment_number = payments_left + num_payments_to_generate
            
            # Calculate interest for this period
            interest_payment = remaining_balance * monthly_rate
            
            # Calculate principal for this period
            principal_payment = monthly_payment - interest_payment
            
            # Final payment may be different
            total_principal_payment = remaining_balance
            payment_amount = total_principal_payment + interest_payment
            
            # Update total interest (approximate for final payment)
            total_interest_estimate = total_interest + (payments_left - 1) * (remaining_balance * monthly_rate / 2)
            
            # Add final payment to schedule
            payment = {
                "payment_number": final_payment_number,
                "payment_date": f"Year {final_payment_number//12 + 1}, Month {final_payment_number%12 or 12}",
                "payment_amount": round(payment_amount, 2),
                "principal_payment": round(total_principal_payment, 2),
                "interest_payment": round(interest_payment, 2),
                "remaining_balance": 0,
                "total_interest_paid": round(total_interest_estimate, 2)
            }
            
            schedule.append(payment)
        
        # Add summary - FIX: Convert float to int for months_to_payoff
        estimated_months = n_payments - (extra_payment * n_payments / (loan_amount / 3)) if extra_payment > 0 else n_payments
        
        summary = {
            "payment_number": "summary",
            "payment_date": None,
            "payment_amount": round(monthly_payment * n_payments, 2),
            "principal_payment": round(loan_amount, 2),
            "interest_payment": round(monthly_payment * n_payments - loan_amount, 2),
            "remaining_balance": 0,
            "total_interest_paid": round(monthly_payment * n_payments - loan_amount, 2),
            "years_to_payoff": loan_term_years - (extra_payment * loan_term_years / (loan_amount / 3)) if extra_payment > 0 else loan_term_years,
            "months_to_payoff": int(estimated_months)  # Convert to int to fix the validation error
        }
        
        schedule.append(summary)
        return schedule