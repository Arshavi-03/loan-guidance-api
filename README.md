# Loan Guidance System API

A FastAPI application for loan analysis and financial guidance, using a pre-trained joblib model.

## Features

- **Loan Analysis**: Comprehensive loan scenario analysis
- **Visualization**: Generate payment and amortization charts
- **AI Recommendations**: Get personalized financial guidance
- **Payment Schedules**: Calculate detailed payment schedules

## Setup & Deployment

### Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your `loan_guidance_system.joblib` file in the `data/` directory
4. Set environment variables (create a `.env` file in the root directory):
   ```
   OPENAI_API_KEY=your_openai_api_key_here  # Optional for enhanced recommendations
   ```
5. Run the development server:
   ```
   uvicorn app.main:app --reload
   ```
6. Access the API documentation at `http://localhost:8000/docs`

### Deploy to Render

1. Create a new Web Service on Render
2. Connect your repository
3. Set these configuration options:
   - **Environment**: Docker
   - **Build Command**: Automatic (uses Dockerfile)
   - **Start Command**: Automatic (uses Dockerfile)
   - **Environment Variables**:
     - `OPENAI_API_KEY`: Your OpenAI API key (optional)

4. Click "Create Web Service"

## API Endpoints

- **GET /**: Welcome message
- **POST /analyze**: Full loan analysis
- **POST /visualization**: Basic loan visualization
- **POST /enhanced-visualization**: Enhanced visualization with multiple charts
- **POST /payment-schedule**: Monthly payment schedule
- **POST /recommendations**: AI-powered loan recommendations
- **GET /health**: Health check endpoint

## Example Usage

```python
import requests
import json

# API endpoint
url = "https://your-render-app.onrender.com/analyze"

# Example loan data
data = {
    "income": 85000,
    "loan_amount": 325000,
    "loan_term": 30,
    "interest_rate": 6.5,
    "credit_score": 720,
    "monthly_debt": 1200,
    "property_value": 400000,
    "extra_payment": 100
}

# Send request
response = requests.post(url, json=data)

# Process response
if response.status_code == 200:
    result = response.json()
    print(f"Monthly Payment: ${result['analysis']['monthly_payment']}")
    print(f"Risk Level: {result['risk']['overall_risk']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Next.js Integration

To integrate with a Next.js frontend, see the [Next.js Integration Guide](NEXTJS_INTEGRATION.md).