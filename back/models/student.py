from pydantic import BaseModel, Field
from typing import Optional

class Student(BaseModel):
    """Student model with performance-related attributes"""
    id: Optional[int] = None
    name: str
    hours_studied: float = Field(..., ge=0)
    previous_scores: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=24)
    attendance_percentage: float = Field(..., ge=0, le=100)
    extracurricular_activities: bool = False
    practice_tests_taken: int = Field(..., ge=0)
    predicted_score: Optional[float] = None