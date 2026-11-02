"""Predict the resale price for one vehicle using the saved model.

Example:
    python src/predict.py --name city --year 2015 --present-price 10 --kms 40000 \
        --fuel Petrol --seller Dealer --transmission Manual --owner 0
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import ALL_FEATURES, engineer

ROOT = Path(__file__).resolve().parents[1]


def predict(row: dict) -> float:
    model = joblib.load(ROOT / "models" / "car_price_model.joblib")
    df = engineer(pd.DataFrame([row]))
    return max(0.0, float(model.predict(df[ALL_FEATURES])[0]))  # price can't be negative


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="model name as in the dataset, e.g. 'city'")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--present-price", type=float, required=True, help="new price, in lakhs")
    p.add_argument("--kms", type=int, required=True)
    p.add_argument("--fuel", choices=["Petrol", "Diesel", "CNG"], required=True)
    p.add_argument("--seller", choices=["Dealer", "Individual"], required=True)
    p.add_argument("--transmission", choices=["Manual", "Automatic"], required=True)
    p.add_argument("--owner", type=int, default=0)
    a = p.parse_args()
    price = predict({
        "Car_Name": a.name, "Year": a.year, "Present_Price": a.present_price,
        "Driven_kms": a.kms, "Fuel_Type": a.fuel, "Selling_type": a.seller,
        "Transmission": a.transmission, "Owner": a.owner,
    })
    print(f"Estimated selling price: {price:.2f} lakhs (INR {price * 1e5:,.0f})")


if __name__ == "__main__":
    main()
