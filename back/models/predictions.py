import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Union
from pydantic import BaseModel
from ..utils.preprocessing import preprocess_data

class StudentPredictionModel:
    def __init__(self, model_path: str = "models/student_performance_model.joblib"):
        self.model_path = model_path
        self.scaler = self._load_scaler()
        self.feature_names = [
            'hours_studied', 'previous_scores', 'sleep_hours', 
            'attendance_percentage', 'extracurricular_activities', 
            'practice_tests_taken'
        ]
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model or train a new one if it doesn't exist"""
        try:
            return joblib.load(self.model_path)
        except:
            print("Training a new model...")
            return self._train_model()
    
    def _load_scaler(self):
        """Load the feature scaler or create a new one"""
        try:
            return joblib.load("models/student_scaler.joblib")
        except:
            return StandardScaler()
            
    def _train_model(self):
        """Train a predictive model for student performance"""
        # Load dataset (this would be your actual dataset)
        try:
            df = pd.read_csv("../services/student_data.csv")
        except:
            # If dataset doesn't exist, create a synthetic one for demonstration
            df = self._create_synthetic_dataset()
            os.makedirs("services", exist_ok=True)
            df.to_csv("../services/student_data.csv", index=False)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['final_score']
        
        # Convert boolean to int
        X['extracurricular_activities'] = X['extracurricular_activities'].astype(int)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and scaler
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(self.scaler, "models/student_scaler.joblib")
        
        # Print model performance
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Model R² score: {r2:.4f}")
        print(f"Model RMSE: {rmse:.4f}")
        
        return model
        
    def _create_synthetic_dataset(self, n_samples=500):
        """Create a synthetic dataset for demonstration purposes"""
        np.random.seed(42)
        
        # Generate synthetic data with realistic correlations
        hours_studied = np.random.normal(7, 2, n_samples)
        previous_scores = np.random.normal(75, 15, n_samples)
        sleep_hours = np.random.normal(7, 1.5, n_samples)
        attendance = np.random.normal(85, 10, n_samples)
        extracurricular = np.random.choice([True, False], size=n_samples)
        practice_tests = np.random.poisson(5, n_samples)
        
        # Create the target variable with some noise
        final_score = (
            0.3 * hours_studied +
            0.25 * previous_scores / 10 +
            0.15 * sleep_hours +
            0.2 * attendance / 10 +
            0.05 * extracurricular.astype(int) * 10 +
            0.05 * practice_tests +
            np.random.normal(0, 5, n_samples)  # Adding noise
        )
        
        # Ensure scores are in a reasonable range
        final_score = np.clip(final_score, 0, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'hours_studied': hours_studied,
            'previous_scores': previous_scores,
            'sleep_hours': sleep_hours,
            'attendance_percentage': attendance,
            'extracurricular_activities': extracurricular,
            'practice_tests_taken': practice_tests,
            'final_score': final_score
        })
        
        return df
    
    def predict(self, student_data):
        """Make a prediction based on student features"""
        # Convert to numpy array
        features = np.array([[
            student_data.hours_studied,
            student_data.previous_scores,
            student_data.sleep_hours,
            student_data.attendance_percentage,
            int(student_data.extracurricular_activities),
            student_data.practice_tests_taken
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get prediction interval (crude approximation)
        confidence = 0.85  # Arbitrary confidence value
        
        return {
            "predicted_score": round(float(prediction), 2),
            "confidence": confidence
        }
    
    def get_feature_importance(self):
        """Return the importance of each feature in the model"""
        if not hasattr(self.model, 'feature_importances_'):
            return [{"feature": name, "importance": 0} for name in self.feature_names]
            
        importances = self.model.feature_importances_
        features = [
            {"feature": name, "importance": float(importance)}
            for name, importance in zip(self.feature_names, importances)
        ]
        
        # Sort by importance
        return sorted(features, key=lambda x: x["importance"], reverse=True)