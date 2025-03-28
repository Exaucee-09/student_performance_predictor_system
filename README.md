# Student Performance Predictor System

This project predicts student academic performance based on various factors like study hours, attendance, sleep patterns, and more. It uses machine learning to provide predictions and recommendations for improvement.

## Project Structure

```
STUDENT_PERFORMANCE_PREDICTOR_SYSTEM/
├── back/
│   ├── endpoint/
│   │   └── student_performance.py  # FastAPI endpoints
│   ├── models/
│   │   ├── student.py              # Student data model
│   │   └── predictions.py          # ML prediction model
│   └── services/
│       ├── model_service.py        # Model training service
│       └── student_data.csv        # Training data
├── front/
│   └── index.html                  # Frontend interface
└── main.py                         # FastAPI app initialization
```

## Features

- Predicts student performance based on multiple factors
- Provides confidence levels for predictions
- Shows feature importance visualization
- Offers personalized improvement suggestions
- Simple and intuitive UI

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- scikit-learn
- pandas
- numpy

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib
```

3. Run the server:

```bash
uvicorn back.main:app --reload
```

4. Open the frontend by opening `front/index.html` in a browser

## Model Details

The prediction model uses a Random Forest Regressor to predict student performance based on:

- Hours studied per week
- Previous academic scores
- Sleep hours per day
- Attendance percentage
- Participation in extracurricular activities
- Number of practice tests taken

## API Endpoints

- `POST /api/student/predict`: Predicts student performance
- `GET /api/student/factors`: Returns feature importance data

## Example Usage

```python
# Example API call
import requests
import json

data = {
    "hours_studied": 8.5,
    "previous_scores": 78.5,
    "sleep_hours": 7.0,
    "attendance_percentage": 92.0,
    "extracurricular_activities": True,
    "practice_tests_taken": 6
}

response = requests.post(
    "http://localhost:8000/api/student/predict", 
    data=json.dumps(data),
    headers={"Content-Type": "application/json"}
)

result = response.json()
print(f"Predicted score: {result['predicted_score']}")
```