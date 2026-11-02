# Car Price Prediction with Machine Learning

Predict the resale price of a used car or motorbike from its features, using Python, Pandas,
Scikit-learn and Matplotlib. The project covers the full workflow: data cleaning, feature
engineering, model training and tuning, evaluation, and a prediction script.

## Dataset

`data/car_data.csv` — 301 used vehicles (cars and motorbikes). Prices are in **lakhs INR**.

| Column | Meaning |
|---|---|
| Car_Name | Model name (cars) or maker + model (bikes) |
| Year | Year of manufacture |
| Selling_Price | **Target** – resale price |
| Present_Price | Current ex-showroom price of the new vehicle |
| Driven_kms | Kilometres driven |
| Fuel_Type | Petrol / Diesel / CNG |
| Selling_type | Dealer / Individual |
| Transmission | Manual / Automatic |
| Owner | Number of previous owners |

The dataset has no horsepower or brand-reputation column, so those are not used directly.
"Brand goodwill" is engineered instead (see below).

## Workflow

1. **Cleaning** (`src/features.py::clean`): trim whitespace, drop 2 exact duplicates, validate values. No missing values.
2. **Feature engineering** (`src/features.py::engineer`):
   - `Brand` and `Vehicle_Type` (Car/Bike) derived from the name
   - `Car_Age` (relative to the 2018 dataset snapshot), `Log_Kms`, `Kms_Per_Year`, `Log_Present_Price`
   - `Brand_Goodwill`: smoothed average of log(resale) − log(new price) per brand, learned **inside** the
     pipeline so cross-validation never leaks test-fold targets
3. **Modelling** (`src/train.py`): target is log-transformed (prices are heavily right-skewed).
   Ridge, Random Forest and Gradient Boosting are tuned with 5-fold CV grid search on an 80/20 split.
   The best model is chosen by **cross-validated RMSE** on the training set, not by the test set.
4. **Evaluation**: R², MAE and RMSE on the held-out test set, plus a naive baseline
   ("resale price = new price"), residual plots and permutation feature importance.

## Results (60-row held-out test set)

| Model | CV RMSE | Test R² | Test MAE (lakhs) | Test RMSE (lakhs) |
|---|---|---|---|---|
| **Ridge (selected)** | **1.30** | 0.946 | 0.65 | 1.12 |
| Random Forest | 1.56 | 0.967 | 0.53 | 0.89 |
| Gradient Boosting | 1.53 | 0.965 | 0.54 | 0.91 |
| Baseline (resale = new price) | – | 0.189 | 2.64 | 4.37 |

Notes and caveats:
- Ridge was selected by cross-validation. The tree models scored slightly better on this small test
  set but were less stable across CV folds (R² std ≈ 0.06 vs 0.01), so the difference may be noise.
  With only 301 rows, treat all numbers as approximate.
- `Present_Price` and `Car_Age` dominate predictions. Brand goodwill adds little on top of
  `Present_Price`, which already reflects the brand's price tier.
- The model tends to under-predict the few very expensive vehicles (e.g. Fortuner, Land Cruiser),
  which are rare in the data.
- Data is from around 2018, so predictions do not reflect today's used-car market.

Figures are in `outputs/figures/`, metrics in `outputs/metrics.json`.

## Run it

```bash
pip install -r requirements.txt
python src/eda.py        # exploratory plots
python src/train.py      # train, evaluate, save models/car_price_model.joblib
python src/predict.py --name city --year 2015 --present-price 10 --kms 40000 \
    --fuel Petrol --seller Dealer --transmission Manual --owner 0
```

## Project structure

```
data/car_data.csv
src/features.py   cleaning, feature engineering, preprocessing pipeline
src/eda.py        exploratory plots
src/train.py      training, tuning, evaluation, plots
src/predict.py    CLI for single predictions
models/           saved model
outputs/          metrics.json and figures
```

## Real-world applications

Used-vehicle marketplaces, dealers and insurers use price models like this to set listing prices,
screen trade-in offers, detect mispriced listings and estimate asset value for loans.
