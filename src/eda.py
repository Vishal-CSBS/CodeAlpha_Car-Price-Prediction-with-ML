"""Exploratory data analysis: saves plots to outputs/figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from features import TARGET, clean, engineer

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = engineer(clean(pd.read_csv(ROOT / "data" / "car_data.csv")))
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], bins=40, ax=ax[0]).set(title="Selling price (lakhs)")
    sns.histplot(np.log1p(df[TARGET]), bins=40, ax=ax[1]).set(title="log(1 + Selling price)")
    fig.tight_layout(); fig.savefig(FIG / "01_target_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="Present_Price", y=TARGET, hue="Vehicle_Type", ax=ax)
    ax.set_title("New price vs resale price")
    fig.tight_layout(); fig.savefig(FIG / "02_present_vs_selling.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(data=df, x="Fuel_Type", y=TARGET, ax=ax[0])
    sns.boxplot(data=df, x="Transmission", y=TARGET, ax=ax[1])
    sns.boxplot(data=df, x="Selling_type", y=TARGET, ax=ax[2])
    for a in ax:
        a.set_yscale("log")
    fig.tight_layout(); fig.savefig(FIG / "03_categorical_vs_price.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    num = df[[TARGET, "Present_Price", "Car_Age", "Driven_kms", "Kms_Per_Year", "Owner"]]
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(FIG / "04_correlation.png", dpi=150); plt.close(fig)

    print(f"Saved EDA figures to {FIG}")


if __name__ == "__main__":
    main()
