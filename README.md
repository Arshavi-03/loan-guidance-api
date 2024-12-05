# Advanced Loan Guidance System API

This API provides comprehensive loan guidance and risk assessment services for rural financial inclusion.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## API Endpoints

- GET `/`: Root endpoint - API status
- GET `/health`: System health check
- POST `/predict`: Generate loan guidance and risk assessment

## Documentation

API documentation available at `/docs` when running locally.

## Testing the API

Example curl command:
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
    "monthly_income": 5000,
    "loan_amount": 50000,
    "interest_rate": 5.5,
    "loan_term_months": 36,
    "credit_score": 720,
    "age": 30,
    "borrower_type": "business",
    "sector_data": {"business": {"years": 5, "type": "retail"}},
    "payment_history": []
}'
```

## Deployment

This application is configured for deployment on Render.com.