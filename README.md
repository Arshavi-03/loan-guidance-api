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

3. Configure environment variables:
Create a `.env` file with:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
```

4. Run the application:
```bash
uvicorn main:app --reload
```

## API Endpoints

- GET `/`: Health check
- POST `/analyze-loan`: Analyze loan application
- GET `/health`: System health status

## Documentation

API documentation available at `/docs` when running locally.

## Deployment

This application is configured for deployment on Render.com with AWS S3 for model storage.