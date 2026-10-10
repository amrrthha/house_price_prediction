from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "train.csv"
OUT = BASE / "artifacts"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["area", "bedrooms", "bathrooms", "location", "price"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"CSV must contain these columns: {required}")

    df = df[required].copy()

    for col in ["area", "bedrooms", "bathrooms", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["location"] = df["location"].astype("string").str.strip()
    df = df.dropna(subset=required)
    df = df[df["price"] > 0]

    if len(df) < 10:
        raise ValueError("Please provide at least 10 valid data rows.")

    features = ["area", "bedrooms", "bathrooms", "location"]
    numeric = ["area", "bedrooms", "bathrooms"]
    categorical = ["location"]

    X = df[features]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    algorithms = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=1,
        ),
    }

    models = {}
    metrics = {}

    for name, algorithm in algorithms.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", algorithm),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        models[name] = pipe
        metrics[name] = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": float(r2_score(y_test, pred)),
        }

        print(f"\n{name}")
        print(metrics[name])

    best_name = min(metrics, key=lambda name: metrics[name]["RMSE"])
    best_model = models[best_name]
    predictions = best_model.predict(X_test)

    bundle = {
        "model": best_model,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "categories": {
            "location": sorted(df["location"].astype(str).unique().tolist())
        },
        "defaults": {
            "area": float(df["area"].median()),
            "bedrooms": float(df["bedrooms"].median()),
            "bathrooms": float(df["bathrooms"].median()),
            "location": str(df["location"].mode().iloc[0]),
        },
        "target": "price",
        "best_model_name": best_name,
        "metrics": metrics,
        "y_test": y_test.to_numpy(),
        "predictions": predictions,
        "rows": len(df),
    }

    joblib.dump(bundle, OUT / "house_price_model.joblib")

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel training completed!")
    print("Best model:", best_name)
    print("Dataset rows:", len(df))
    print("Saved to:", OUT / "house_price_model.joblib")

if __name__ == "__main__":
    main()