
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
DATA_FILE = Path("data/vehicle_registrations.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# --------------------------------------------------
# 2. LOAD AND CLEAN DATA
# --------------------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Download the CSV and save it in the data folder."
    )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

required = [
    "year", "month", "vehicle_class", "fuel", "value"
]
missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df.dropna(
    subset=["year", "month", "vehicle_class", "fuel", "value"]
).copy()

df["year"] = df["year"].astype(int)
df["value"] = df["value"].clip(lower=0)

# Remove exact duplicate rows.
df = df.drop_duplicates()

# Convert month names/numbers into month numbers.
month_map = {
    name.lower(): number
    for number, name in enumerate(
        [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ],
        start=1
    )
}

df["month_num"] = (
    df["month"].astype(str).str.strip().str.lower()
    .map(month_map)
)

numeric_month = pd.to_numeric(df["month"], errors="coerce")
df["month_num"] = df["month_num"].fillna(numeric_month)

df = df[df["month_num"].between(1, 12)].copy()
df["month_num"] = df["month_num"].astype(int)

# The primary dataset is intended for all-India analysis.
# If it contains state-level rows, select national rows only
# to avoid mixing national totals with state-level totals.
if "state" in df.columns:
    all_india = df[
        df["state"].astype(str).str.strip().str.lower()
        .eq("all india")
    ]
    if not all_india.empty:
        df = all_india.copy()

# Remove negative and missing counts.
df = df[df["value"] >= 0].copy()

if df.empty:
    raise ValueError("No usable rows remain after cleaning.")

print("\nDATASET OVERVIEW")
print("-" * 45)
print("Rows:", len(df))
print("Years:", df["year"].min(), "to", df["year"].max())
print("Vehicle classes:", df["vehicle_class"].nunique())
print("Fuel categories:", df["fuel"].nunique())
print(df.head())

# --------------------------------------------------
# 3. MONTHLY AND YEARLY AGGREGATION
# --------------------------------------------------
# The same month may have many vehicle classes and fuels.
# Sum these categories to obtain monthly totals.
monthly = (
    df.groupby(["year", "month_num"], as_index=False)["value"]
    .sum()
    .sort_values(["year", "month_num"])
)

monthly["date"] = pd.to_datetime(
    dict(
        year=monthly["year"],
        month=monthly["month_num"],
        day=1
    )
)

# Only use full calendar years for yearly trend comparisons.
months_per_year = monthly.groupby("year")["month_num"].nunique()
full_years = months_per_year[months_per_year == 12].index

yearly = (
    monthly[monthly["year"].isin(full_years)]
    .groupby("year", as_index=False)["value"]
    .sum()
    .rename(columns={"value": "registrations"})
    .sort_values("year")
)

if len(yearly) < 4:
    raise ValueError(
        "At least four complete years are recommended for "
        "the trend and forecasting sections."
    )

yearly["growth_pct"] = (
    yearly["registrations"].pct_change() * 100
)

yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)

print("\nYEARLY SUMMARY")
print(yearly.to_string(index=False))

# --------------------------------------------------
# 4. VISUALIZATION 1: YEARLY TREND
# --------------------------------------------------
plt.figure()
sns.lineplot(
    data=yearly,
    x="year",
    y="registrations",
    marker="o"
)
plt.title("India: Annual Vehicle Registration Trend")
plt.xlabel("Year")
plt.ylabel("Registrations")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yearly_trend.png", dpi=160)
plt.close()

# --------------------------------------------------
# 5. VISUALIZATION 2: VEHICLE CLASS
# --------------------------------------------------
class_totals = (
    df.groupby("vehicle_class")["value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
class_totals.plot(kind="barh", color="steelblue")
plt.title("Top 10 Vehicle Classes by Recorded Registrations")
plt.xlabel("Registrations in dataset")
plt.ylabel("Vehicle class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vehicle_class_analysis.png", dpi=160)
plt.close()

# --------------------------------------------------
# 6. VISUALIZATION 3: FUEL TYPE
# --------------------------------------------------
fuel_totals = (
    df.groupby("fuel")["value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
fuel_totals.plot(kind="barh", color="darkorange")
plt.title("Top 10 Fuel Categories by Recorded Registrations")
plt.xlabel("Registrations in dataset")
plt.ylabel("Fuel type")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fuel_type_analysis.png", dpi=160)
plt.close()

# --------------------------------------------------
# 7. VISUALIZATION 4: ELECTRIC VEHICLE TREND
# --------------------------------------------------
# This selects fuel labels containing EV/electric terms.
# Inspect the actual fuel labels in your dataset to ensure
# the selected labels match the intended EV definition.
ev_mask = df["fuel"].astype(str).str.contains(
    r"electric|pure ev|electric\(bov\)|plug.?in hybrid",
    case=False,
    regex=True,
    na=False
)

ev_monthly = (
    df[ev_mask]
    .groupby(["year", "month_num"], as_index=False)["value"]
    .sum()
)

if not ev_monthly.empty:
    ev_yearly = (
        ev_monthly[ev_monthly["year"].isin(full_years)]
        .groupby("year", as_index=False)["value"]
        .sum()
    )

    plt.figure()
    sns.lineplot(
        data=ev_yearly,
        x="year",
        y="value",
        marker="o",
        color="green"
    )
    plt.title("Recorded Electric Vehicle Registrations")
    plt.xlabel("Year")
    plt.ylabel("Registrations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ev_trend.png", dpi=160)
    plt.close()
else:
    print(
        "No EV fuel labels matched the filter. "
        "Inspect df['fuel'].unique() and adjust the EV pattern."
    )

# --------------------------------------------------
# 8. MACHINE LEARNING: TIME-BASED HOLDOUT
# --------------------------------------------------
# This forecasts aggregate annual registrations.
# Hold out the latest complete year for testing.
series = yearly.set_index("year")["registrations"].sort_index()

test_year = int(series.index[-1])
train_series = series[series.index < test_year]
test_actual = float(series.loc[test_year])

# Build lag-based training data using only past values.
def make_lag_data(values, lags=3):
    rows = []
    for i in range(lags, len(values)):
        rows.append({
            "lag_1": values[i - 1],
            "lag_2": values[i - 2],
            "lag_3": values[i - 3],
            "year": years_list[i],
            "target": values[i]
        })
    return pd.DataFrame(rows)

years_list = list(series.index)
values_list = list(series.values)

lag_df = make_lag_data(values_list, lags=3)
lag_train = lag_df[lag_df["year"] < test_year]

if len(lag_train) < 3:
    raise ValueError(
        "Not enough annual observations to train the lag model."
    )

features = ["lag_1", "lag_2", "lag_3", "year"]

# Ridge regression is used as a transparent baseline.
baseline = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
baseline.fit(lag_train[features], lag_train["target"])

# The test prediction uses only registrations known before test_year.
test_features = pd.DataFrame([{
    "lag_1": series.loc[test_year - 1],
    "lag_2": series.loc[test_year - 2],
    "lag_3": series.loc[test_year - 3],
    "year": test_year
}])

baseline_prediction = max(
    0.0, float(baseline.predict(test_features)[0])
)

# Random Forest model for comparison.
forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)
forest.fit(lag_train[features], lag_train["target"])

forest_prediction = max(
    0.0, float(forest.predict(test_features)[0])
)

# Compare each model on the same held-out year.
results = pd.DataFrame([
    {
        "year": test_year,
        "model": "Ridge Regression",
        "actual": test_actual,
        "predicted": baseline_prediction,
        "absolute_error": abs(test_actual - baseline_prediction)
    },
    {
        "year": test_year,
        "model": "Random Forest",
        "actual": test_actual,
        "predicted": forest_prediction,
        "absolute_error": abs(test_actual - forest_prediction)
    }
])

results.to_csv(OUTPUT_DIR / "forecast_results.csv", index=False)

for name, prediction in [
    ("Ridge Regression", baseline_prediction),
    ("Random Forest", forest_prediction)
]:
    mae = mean_absolute_error([test_actual], [prediction])
    rmse = np.sqrt(mean_squared_error([test_actual], [prediction]))
    print(
        f"\n{name} — held-out year {test_year}: "
        f"actual={test_actual:,.0f}, "
        f"predicted={prediction:,.0f}, "
        f"MAE={mae:,.0f}, RMSE={rmse:,.0f}"
    )

# Plot actual and predicted values for the test year.
plt.figure(figsize=(8, 5))
plot_data = results.melt(
    id_vars=["year", "model"],
    value_vars=["actual", "predicted"],
    var_name="kind",
    value_name="registrations"
)

sns.barplot(
    data=plot_data,
    x="model",
    y="registrations",
    hue="kind"
)
plt.title(f"Actual vs Predicted Registrations ({test_year})")
plt.xlabel("Model")
plt.ylabel("Registrations")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "forecast.png", dpi=160)
plt.close()

# --------------------------------------------------
# 9. FUTURE FORECAST: THREE YEARS
# --------------------------------------------------
# Refit each model on all available complete years,
# then recursively forecast using its own prior predictions.
all_lag_df = make_lag_data(values_list, lags=3)

final_ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
final_ridge.fit(all_lag_df[features], all_lag_df["target"])

final_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)
final_forest.fit(all_lag_df[features], all_lag_df["target"])

future_rows = []
history = list(values_list)

for future_year in range(int(series.index[-1]) + 1,
                         int(series.index[-1]) + 4):
    x = pd.DataFrame([{
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "year": future_year
    }])

    ridge_forecast = max(
        0.0, float(final_ridge.predict(x)[0])
    )
    forest_forecast = max(
        0.0, float(final_forest.predict(x)[0])
    )

    # Use the average as a simple ensemble forecast.
    ensemble = (ridge_forecast + forest_forecast) / 2

    future_rows.append({
        "year": future_year,
        "ridge_forecast": round(ridge_forecast),
        "random_forest_forecast": round(forest_forecast),
        "ensemble_forecast": round(ensemble)
    })

    history.append(ensemble)

future_df = pd.DataFrame(future_rows)
future_df.to_csv(OUTPUT_DIR / "future_forecast.csv", index=False)

plt.figure()
plt.plot(
    series.index, series.values,
    marker="o", label="Historical"
)
plt.plot(
    future_df["year"], future_df["ensemble_forecast"],
    marker="o", linestyle="--", label="Forecast"
)
plt.title("Annual Vehicle Registrations: Historical and Forecast")
plt.xlabel("Year")
plt.ylabel("Registrations")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "future_trend.png", dpi=160)
plt.close()

print("\nFUTURE FORECAST")
print(future_df.to_string(index=False))

print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
