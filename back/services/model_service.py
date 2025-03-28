import os
import pandas as pd
from ..models.predictions import StudentPredictionModel  # Assuming correct import path

class StudentPredictionService:
    def __init__(self, csv_file: str):
        # Get the absolute path to the CSV file
        base_dir = os.path.dirname(__file__)
        self.csv_file = os.path.join(base_dir, csv_file)

        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found at {self.csv_file}")
        
        self.df = pd.read_csv(self.csv_file)

        # Initialize the model
        self.model = StudentPredictionModel()  # The model is initialized without needing the dataframe

    def get_prediction(self, hours_studied: float, previous_scores: float, sleep_hours: float, attendance_percentage: float, extracurricular_activities: bool, practice_tests_taken: int) -> Dict[str, Any]:
        # Use the StudentPredictionModel to make predictions
        student_data = {
            "hours_studied": hours_studied,
            "previous_scores": previous_scores,
            "sleep_hours": sleep_hours,
            "attendance_percentage": attendance_percentage,
            "extracurricular_activities": extracurricular_activities,
            "practice_tests_taken": practice_tests_taken
        }

        # Convert student data to a Pydantic model (optional, depends on your use case)
        from ..models.predictions import StudentData
        student_data_model = StudentData(**student_data)

        # Get the prediction using the model's predict method
        return self.model.predict(student_data_model)
