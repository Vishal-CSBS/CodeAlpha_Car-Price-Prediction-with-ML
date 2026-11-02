"""Data cleaning, feature engineering and the sklearn preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The dataset snapshot is from ~2018, so vehicle age is measured against that year.
REFERENCE_YEAR = 2018

TARGET = "Selling_Price"  # in lakhs INR

# Car names in the file are models, not makers, so map model -> manufacturer.
CAR_MODEL_TO_BRAND = {
    **{m: "Maruti" for m in ["ritz", "sx4", "ciaz", "wagon r", "swift", "vitara brezza",
                             "s cross", "alto 800", "alto k10", "ertiga", "dzire",
                             "ignis", "800", "baleno", "omni"]},
    **{m: "Toyota" for m in ["fortuner", "innova", "corolla altis", "etios cross",
                             "etios g", "etios liva", "etios gd", "corolla", "camry",
                             "land cruiser"]},
    **{m: "Hyundai" for m in ["i20", "grand i10", "i10", "eon", "xcent", "elantra",
                              "creta", "verna"]},
    **{m: "Honda" for m in ["city", "brio", "amaze", "jazz"]},
}

# Two-wheeler names start with the maker (order matters: longest prefix first).
BIKE_PREFIXES = [
    ("royal enfield", "Royal Enfield"), ("hero honda", "Hero"), ("hero", "Hero"),
    ("bajaj", "Bajaj"), ("ktm", "KTM"), ("um ", "UM"), ("hyosung", "Hyosung"),
    ("mahindra", "Mahindra"), ("honda", "Honda"), ("activa", "Honda"),
    ("yamaha", "Yamaha"), ("tvs", "TVS"), ("suzuki", "Suzuki"),
]


def _brand_and_type(name: str):
    n = " ".join(str(name).lower().split())
    if n in CAR_MODEL_TO_BRAND:
        return CAR_MODEL_TO_BRAND[n], "Car"
    for prefix, brand in BIKE_PREFIXES:
        if n.startswith(prefix):
            return brand, "Bike"
    return "Other", "Car"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, drop exact duplicates, validate ranges."""
    df = df.copy()
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    assert (df[TARGET] > 0).all() and (df["Present_Price"] > 0).all()
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering (safe to apply before the train/test split)."""
    df = df.copy()
    brand_type = df["Car_Name"].apply(_brand_and_type)
    df["Brand"] = [b for b, _ in brand_type]
    df["Vehicle_Type"] = [t for _, t in brand_type]
    df["Car_Age"] = REFERENCE_YEAR - df["Year"]
    df["Log_Kms"] = np.log1p(df["Driven_kms"])
    df["Kms_Per_Year"] = df["Driven_kms"] / df["Car_Age"].clip(lower=1)
    df["Log_Present_Price"] = np.log1p(df["Present_Price"])
    return df


class BrandGoodwill(BaseEstimator, TransformerMixin):
    """Brand 'goodwill' = how well a brand retains value, learnt from training data only.

    For each brand we compute the mean of log(resale price) - log(new price), shrunk toward
    the global mean so brands with few rows don't get extreme values. Fitting happens inside
    the pipeline, so cross-validation never leaks test-fold targets.
    X must contain 'Brand' and 'Present_Price'; y is the (log-transformed) selling price.
    """

    def __init__(self, smoothing: float = 5.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        retention = np.asarray(y) - np.log1p(X["Present_Price"].to_numpy())
        tmp = pd.DataFrame({"Brand": X["Brand"].to_numpy(), "r": retention})
        self.global_mean_ = tmp["r"].mean()
        g = tmp.groupby("Brand")["r"].agg(["sum", "count"])
        self.map_ = ((g["sum"] + self.smoothing * self.global_mean_) /
                     (g["count"] + self.smoothing)).to_dict()
        return self

    def transform(self, X):
        vals = X["Brand"].map(self.map_).fillna(self.global_mean_).to_numpy()
        return vals.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["Brand_Goodwill"])


NUMERIC = ["Car_Age", "Log_Kms", "Kms_Per_Year", "Log_Present_Price", "Owner"]
CATEGORICAL = ["Fuel_Type", "Selling_type", "Transmission", "Vehicle_Type"]
ALL_FEATURES = NUMERIC + CATEGORICAL + ["Brand", "Present_Price"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("goodwill", BrandGoodwill(), ["Brand", "Present_Price"]),
    ])
