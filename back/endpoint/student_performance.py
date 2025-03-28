from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..models.predictions import StudentPredictionModel
from typing import List, Optional

router = APIRouter(
    prefix="/api/student",
    tags=["student performance"]
)

class StudentFeatures(BaseModel):
    hours_studied: float
    previous_scores: float
    sleep_hours: float
    attendance_percentage: float
    extracurricular_activities: bool
    practice_tests_taken: int
class StudentPredictionResponse(BaseModel):
    predicted_score: float
    confidence: float

prediction_model = StudentPredictionModel()

@router.post("/predict", response_model=StudentPredictionResponse)
async def predict_performance(student: StudentFeatures):
    try:
        prediction = prediction_model.predict(student)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/factors", response_model=List[dict])
async def get_important_factors():
    """Return the top factors affecting student performance"""
    try:
        return prediction_model.get_feature_importance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")