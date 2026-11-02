"""
Model Training and Benchmarking Pipeline for Airfare Price Estimation.
Compares Linear Regression, Ridge, Decision Tree, Random Forest, and HistGradientBoosting.
Generates evaluation metrics (MAE, RMSE, MAPE, R2) and extracts feature importances.
"""

import os
import sys
import json
import time
import tempfile

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl")

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Add current dir to path to import data_preprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_preprocessing import prepare_train_test_data, get_preprocessor, CAT_COLS, NUM_COLS

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid zero division
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_benchmark(csv_path: str, models_dir: str):
    """
    Executes the training and benchmarking routine across candidate regression algorithms.
    """
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 65)
    print("[*] INT395 CA1: AIRFARE COMPONENT PRICE ESTIMATION BENCHMARK")
    print("=" * 65)
    print(f"Loading data from: {csv_path}")

    X_train, X_test, y_train, y_test, full_df = prepare_train_test_data(csv_path)
    print(f"Dataset split: Train={X_train.shape[0]:,} samples, Test={X_test.shape[0]:,} samples")

    preprocessor = get_preprocessor()

    # Define candidate models
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=16, random_state=42, n_jobs=4),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=42)
    }

    results = {}
    fitted_pipelines = {}

    print("\nTraining and evaluating models:")
    print(f"{'Model':<30} | {'MAE (INR)':<10} | {'RMSE (INR)':<10} | {'MAPE (%)':<9} | {'R²':<7} | {'Time (s)':<8}")
    print("-" * 88)

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time

        preds = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mape = float(calculate_mape(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
            "R2": round(r2, 4),
            "Training_Time_s": round(fit_time, 2)
        }
        fitted_pipelines[name] = pipeline

        print(f"{name:<30} | {mae:<10.2f} | {rmse:<10.2f} | {mape:<9.2f} | {r2:<7.4f} | {fit_time:<8.2f}")

    # Determine best model by R2 and MAE
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    print("\n" + "=" * 65)
    print(f"[BEST] Best Model: {best_model_name} (R^2 = {results[best_model_name]['R2']}, MAE = INR {results[best_model_name]['MAE']})")
    print("=" * 65)

    # Save metrics JSON
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # Save best model pipeline
    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"Best pipeline saved to: {best_model_path}")

    # Feature Importance Analysis
    print("\nExtracting feature importances...")
    # Use permutation importance or tree-based feature importances on a test sample
    # We sample 5,000 instances from X_test for fast, robust permutation importance
    sub_X_test = X_test.sample(n=min(5000, len(X_test)), random_state=42)
    sub_y_test = y_test.loc[sub_X_test.index]

    perm_importance = permutation_importance(
        best_pipeline, sub_X_test, sub_y_test, n_repeats=5, random_state=42, n_jobs=4
    )

    feature_names = sub_X_test.columns.tolist()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std
    }).sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)

    # Normalize to percentage
    total_imp = importance_df['Importance_Mean'].sum()
    importance_df['Importance_Percent'] = (importance_df['Importance_Mean'] / total_imp * 100).round(2)

    feat_csv_path = os.path.join(models_dir, "feature_importance.csv")
    importance_df.to_csv(feat_csv_path, index=False)

    feat_json_path = os.path.join(models_dir, "feature_importance.json")
    importance_df.to_json(feat_json_path, orient='records', indent=4)

    print(f"Feature importance saved to: {feat_csv_path}")
    print("\nTop Feature Drivers:")
    for idx, row in importance_df.iterrows():
        print(f"  {idx+1}. {row['Feature']:<20} : {row['Importance_Percent']}% impact")

    return results, best_model_name

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "Clean_Dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    train_and_benchmark(csv_file, models_dir)
