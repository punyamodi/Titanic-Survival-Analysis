import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from src.preprocessor import clean_data

def train_and_evaluate():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    y = train_df['Survived']
    train_count = len(train_df)
    
    combined = pd.concat([train_df.drop(columns=['Survived']), test_df], axis=0)
    combined_cleaned = clean_data(combined)
    
    X = combined_cleaned.iloc[:train_count, :]
    X_test_final = combined_cleaned.iloc[train_count:, :]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"--- Training on {len(X_train)} samples, Validating on {len(X_val)} samples ---")
    
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Running GridSearchCV for RandomForest...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best RF Params: {grid_search.best_params_}")
    
    rf_preds = best_rf.predict(X_val)
    print(f"RF Validation Accuracy: {accuracy_score(y_val, rf_preds):.4f}")
    print(classification_report(y_val, rf_preds))
    
    try:
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        print("Training XGBoost...")
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_val)
        print(f"XGBoost Validation Accuracy: {accuracy_score(y_val, xgb_preds):.4f}")
    except Exception as e:
        print(f"XGBoost training failed or not installed: {e}")
        xgb_model = None

    final_model = best_rf
    
    if not os.path.exists('models'):
        os.makedirs('models')
    model_file = 'models/titanic_best_rf_model.joblib'
    joblib.dump(final_model, model_file)
    print(f"--- Best model saved to {model_file} ---")
    
    final_preds = final_model.predict(X_test_final)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_preds
    })
    
    submission_path = 'data/submission_improved.csv'
    submission.to_csv(submission_path, index=False)
    print(f"--- Improved submission saved to {submission_path} ---")

if __name__ == "__main__":
    train_and_evaluate()
