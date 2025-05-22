
"""Feature Engineering Module

This module provides functions for generating polynomial and interaction features
used in machine learning pipelines for student performance prediction.
"""

def create_input_engineered_features(X):
    """Creates polynomial and interaction features from the input DataFrame X."""
    X_eng = X.copy()

    # Polynomial features
    if 'Age' in X_eng.columns:
        X_eng['Age_Squared'] = X_eng['Age'] ** 2
    if 'StudyTimeWeekly' in X_eng.columns:
        X_eng['StudyTimeWeekly_Squared'] = X_eng['StudyTimeWeekly'] ** 2

    # Interaction features
    if 'Age' in X_eng.columns and 'StudyTimeWeekly' in X_eng.columns:
        X_eng['Age_StudyTime_Interaction'] = X_eng['Age'] * X_eng['StudyTimeWeekly']

    return X_eng


def create_input_engineered_features_df(X):
    """Creates extended engineered features (polynomials and interactions) from the input DataFrame X."""
    X_eng = X.copy()

    # Add polynomial features
    for col in ['Age', 'StudyTimeWeekly', 'Absences']:
        if col in X_eng.columns:
            X_eng[f'{col}_Squared'] = X_eng[col] ** 2

    # Add interaction features
    if 'StudyTimeWeekly' in X_eng.columns and 'Absences' in X_eng.columns:
        X_eng['StudyTime_Absences_Interaction'] = X_eng['StudyTimeWeekly'] * X_eng['Absences']
    if 'Age' in X_eng.columns and 'StudyTimeWeekly' in X_eng.columns:
        X_eng['Age_StudyTime_Interaction'] = X_eng['Age'] * X_eng['StudyTimeWeekly']

    return X_eng
