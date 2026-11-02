"""Train, tune and evaluate regression models; save the best one."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from features import ALL_FEATURES, TARGET, build_preprocessor, clean, engineer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"


def make_model(estimator):
    """Preprocessing + estimator, predicting log(1 + price) and converting back."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    df = engineer(clean(pd.read_csv(ROOT / "data" / "car_data.csv")))
    X, y = df[ALL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df["Vehicle_Type"])
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    cv = KFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "Ridge": (Ridge(), {"regressor__model__alpha": [0.1, 1, 10]}),
        "RandomForest": (
            RandomForestRegressor(random_state=SEED),
            {"regressor__model__n_estimators": [200, 400],
             "regressor__model__max_depth": [None, 8, 12],
             "regressor__model__min_samples_leaf": [1, 2]}),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"regressor__model__n_estimators": [200, 400],
             "regressor__model__learning_rate": [0.03, 0.1],
             "regressor__model__max_depth": [2, 3]}),
    }

    results, fitted = {}, {}
    for name, (est, grid) in candidates.items():
        gs = GridSearchCV(make_model(est), grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        cv_r2 = cross_val_score(best, X_train, y_train, cv=cv, scoring="r2")
        pred = best.predict(X_test)
        results[name] = {
            "best_params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
            "cv_RMSE": float(-gs.best_score_),
            "cv_R2_mean": float(cv_r2.mean()), "cv_R2_std": float(cv_r2.std()),
            "test": metrics(y_test, pred),
        }
        fitted[name] = (best, pred)
        print(f"{name:17s} CV RMSE={results[name]['cv_RMSE']:.3f}  CV R2={cv_r2.mean():.3f}±{cv_r2.std():.3f}  "
              f"test R2={results[name]['test']['R2']:.3f}  MAE={results[name]['test']['MAE']:.3f}  "
              f"RMSE={results[name]['test']['RMSE']:.3f}")

    # Select on cross-validated score (not on the test set) to avoid optimistic bias.
    best_name = min(results, key=lambda k: results[k]["cv_RMSE"])
    best_model, best_pred = fitted[best_name]
    print(f"\nBest model by CV RMSE: {best_name}")

    # Naive baseline: assume resale price = new price (shows the value the model adds).
    baseline = metrics(y_test, X_test["Present_Price"])
    results["_baseline_resale_equals_new_price"] = {"test": baseline}

    # Refit the winner on all data for the saved/deployed model.
    final = clone(best_model).fit(X, y)
    joblib.dump(final, ROOT / "models" / "car_price_model.joblib")

    (ROOT / "outputs" / "metrics.json").write_text(
        json.dumps({"best_model": best_name, "results": results}, indent=2))

    # --- Plots ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y_test, best_pred, alpha=0.7)
    lim = [0, max(y_test.max(), best_pred.max()) * 1.05]
    ax[0].plot(lim, lim, "r--"); ax[0].set(xlabel="Actual (lakhs)", ylabel="Predicted (lakhs)",
                                          title=f"{best_name}: actual vs predicted")
    resid = y_test - best_pred
    ax[1].scatter(best_pred, resid, alpha=0.7); ax[1].axhline(0, color="r", ls="--")
    ax[1].set(xlabel="Predicted (lakhs)", ylabel="Residual", title="Residuals")
    fig.tight_layout(); fig.savefig(FIG / "05_actual_vs_predicted.png", dpi=150); plt.close(fig)

    names = list(results)[:-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [results[n]["test"]["R2"] for n in names])
    ax.set(ylabel="Test R²", title="Model comparison", ylim=(0.8, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "06_model_comparison.png", dpi=150); plt.close(fig)

    pi = permutation_importance(best_model, X_test, y_test, n_repeats=20, random_state=SEED,
                                scoring="r2")
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax); ax.set(title="Permutation importance (test R² drop)")
    fig.tight_layout(); fig.savefig(FIG / "07_feature_importance.png", dpi=150); plt.close(fig)

    print("Baseline (resale = new price):", {k: round(v, 3) for k, v in baseline.items()})
    print("Saved model, metrics and figures.")


if __name__ == "__main__":
    main()
