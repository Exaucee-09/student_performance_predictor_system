import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df: pd.DataFrame, label_encoder: LabelEncoder, scaler: StandardScaler):
    """ Preprocessing for feature extraction and scaling """

    # Encoding categorical data (e.g., extracurricular activities)
    df['extracurricular_activities'] = label_encoder.fit_transform(df['extracurricular_activities'])
    
    # Selecting features and target variable
    X = df[['hours_studied', 'previous_scores', 'sleep_hours', 
            'attendance_percentage', 'extracurricular_activities', 
            'practice_tests_taken']]  # Features
    
    y = df['final_score']  # Target variable
    
    # Scaling the features
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
