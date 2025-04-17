import re
from typing import Dict, List, Any, Optional
import html

def sanitize_html(html_content: str) -> str:
    """
    Convert HTML content to plain text by removing HTML tags.
    Preserves basic formatting like paragraphs and lists.
    """
    if not html_content:
        return ""
    
    # First, decode any HTML entities
    decoded = html.unescape(html_content)
    
    # Replace specific tags with plaintext equivalents
    # Convert headers to text with asterisks
    decoded = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'** \1 **\n', decoded)
    
    # Convert paragraphs to text with newlines
    decoded = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', decoded)
    
    # Convert list items to text with dashes
    decoded = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', decoded)
    
    # Convert break tags to newlines
    decoded = re.sub(r'<br[^>]*>', '\n', decoded)
    
    # Remove remaining HTML tags
    decoded = re.sub(r'<[^>]+>', '', decoded)
    
    # Clean up excessive whitespace
    decoded = re.sub(r'\n{3,}', '\n\n', decoded)
    decoded = decoded.strip()
    
    return decoded

def format_currency(value: float) -> str:
    """Format a numeric value as currency (USD)"""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format a numeric value as a percentage"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

def validate_loan_params(
    income: float, 
    loan_amount: float, 
    loan_term_years: int,
    interest_rate: float,
    credit_score: int,
    monthly_debt: float,
    property_value: Optional[float] = None,
    extra_payment: float = 0
) -> Dict[str, str]:
    """
    Validate loan parameters and return error messages for any invalid values.
    Returns an empty dict if all values are valid.
    """
    errors = {}
    
    # Validate each parameter
    if income <= 0:
        errors["income"] = "Income must be greater than zero"
    
    if loan_amount <= 0:
        errors["loan_amount"] = "Loan amount must be greater than zero"
    
    if loan_term_years <= 0 or loan_term_years > 50:
        errors["loan_term_years"] = "Loan term must be between 1 and 50 years"
    
    if interest_rate < 0 or interest_rate > 30:
        errors["interest_rate"] = "Interest rate must be between 0 and 30 percent"
    
    if credit_score < 300 or credit_score > 850:
        errors["credit_score"] = "Credit score must be between 300 and 850"
    
    if monthly_debt < 0:
        errors["monthly_debt"] = "Monthly debt cannot be negative"
    
    if property_value is not None and property_value <= 0:
        errors["property_value"] = "Property value must be greater than zero"
    
    if extra_payment < 0:
        errors["extra_payment"] = "Extra payment cannot be negative"
    
    return errors