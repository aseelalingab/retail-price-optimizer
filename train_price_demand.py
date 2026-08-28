"""Train and evaluate demand models on the retail pricing dataset."""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from pathlib import Path


np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("/Users/aseelali/Desktop/csv/retail_pricing_daily.csv")

keep = [
    "date", "store_id", "store_region", "sku_id", "category", "our_price",
    "competitor_price", "unit_cost", "promotion_flag", "is_holiday",
    "available_stock", "units_sold", "stockout_flag",
]
df = df.drop_duplicates()[keep].dropna().copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["store_id", "sku_id", "date"]).copy()

group_keys = ["store_id", "sku_id"]
df["lag_our_price"] = df.groupby(group_keys)["our_price"].shift(1)
df["lag_qty_1"] = df.groupby(group_keys)["units_sold"].shift(1)
df["lag_qty_2"] = df.groupby(group_keys)["units_sold"].shift(2)
df["rolling_qty_mean_7"] = (
    df.groupby(group_keys)["units_sold"]
    .transform(lambda values: values.shift(1).rolling(7, min_periods=1).mean())
)
df["price_vs_competitor"] = df["our_price"] / df["competitor_price"]
df["price_change"] = df["our_price"] - df["lag_our_price"]
df["unit_margin"] = df["our_price"] - df["unit_cost"]
df = df.dropna(
    subset=["lag_our_price", "lag_qty_1", "lag_qty_2", "rolling_qty_mean_7"]
).copy()
df = df[df["stockout_flag"] == 0].copy()

all_dates = np.sort(df["date"].unique())
validation_cutoff = pd.Timestamp(all_dates[int(len(all_dates) * 0.70)])
test_cutoff = pd.Timestamp(all_dates[int(len(all_dates) * 0.85)])
train_df = df[df["date"] < validation_cutoff].copy()
validation_df = df[(df["date"] >= validation_cutoff) & (df["date"] < test_cutoff)].copy()
test_df = df[df["date"] >= test_cutoff].copy()

drop_columns = ["units_sold", "stockout_flag"]
X_train = train_df.drop(columns=drop_columns).copy()
y_train = train_df["units_sold"].astype("float32").copy()
X_validation = validation_df.drop(columns=drop_columns).copy()
y_validation = validation_df["units_sold"].astype("float32").copy()
X_test = test_df.drop(columns=drop_columns).copy()
y_test = test_df["units_sold"].astype("float32").copy()

for data in [X_train, X_validation, X_test]:
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month_sin"] = np.sin(2 * np.pi * data["date"].dt.month / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["date"].dt.month / 12)
    data.drop(columns="date", inplace=True)

categorical_columns = ["store_id", "store_region", "sku_id", "category"]
X_train = pd.get_dummies(X_train, columns=categorical_columns, dtype=int)
X_validation = pd.get_dummies(X_validation, columns=categorical_columns, dtype=int)
X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=int)
X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

numeric_columns = [
    "our_price", "competitor_price", "unit_cost", "promotion_flag", "is_holiday",
    "available_stock", "lag_our_price", "lag_qty_1", "lag_qty_2",
    "rolling_qty_mean_7", "price_vs_competitor", "price_change", "unit_margin",
    "day_of_week", "month_sin", "month_cos",
]
scaler = MinMaxScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_validation[numeric_columns] = scaler.transform(X_validation[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

average_baseline_mae = mean_absolute_error(y_validation, np.full(len(y_validation), y_train.mean()))
previous_day_baseline_mae = mean_absolute_error(y_validation, validation_df["lag_qty_1"])

model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mae", metrics=["mae"])
early_stopping = keras.callbacks.EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train, epochs=100, validation_data=(X_validation, y_validation),
    callbacks=[early_stopping], verbose=0, batch_size=1024,
)
ann_validation_predictions = np.maximum(model.predict(X_validation, verbose=0).flatten(), 0)
ann_validation_mae = mean_absolute_error(y_validation, ann_validation_predictions)
print(f"ANN validation MAE: {ann_validation_mae:.3f}", flush=True)

rf_model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_validation_mae = mean_absolute_error(y_validation, rf_model.predict(X_validation))
print(f"Random Forest validation MAE: {rf_validation_mae:.3f}", flush=True)

gb_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_validation_mae = mean_absolute_error(y_validation, gb_model.predict(X_validation))
print(f"Gradient Boosting validation MAE: {gb_validation_mae:.3f}", flush=True)

validation_scores = {
    "Average-sales baseline": average_baseline_mae,
    "Previous-day-sales baseline": previous_day_baseline_mae,
    "ANN": ann_validation_mae,
    "Random Forest": rf_validation_mae,
    "Gradient Boosting": gb_validation_mae,
}
best_method = min(validation_scores, key=validation_scores.get)

if best_method == "Average-sales baseline":
    final_test_predictions = np.full(len(y_test), y_train.mean())
elif best_method == "Previous-day-sales baseline":
    final_test_predictions = test_df["lag_qty_1"].to_numpy()
elif best_method == "ANN":
    final_test_predictions = np.maximum(model.predict(X_test, verbose=0).flatten(), 0)
elif best_method == "Random Forest":
    final_test_predictions = rf_model.predict(X_test)
else:
    final_test_predictions = gb_model.predict(X_test)

final_test_mae = mean_absolute_error(y_test, final_test_predictions)

print("Validation MAE results:")
for name, score in validation_scores.items():
    print(f"{name}: {score:.3f}")
print(f"Best validation method: {best_method}")
print(f"Final test MAE: {final_test_mae:.3f}")


# STEP 18: Constrained price optimization using the selected ANN.
# This is a what-if simulation for one historical SKU/store/day context.
# It changes only our_price and recomputes price-derived features.
def prepare_price_candidate(base_context, candidate_price):
    """Convert one raw decision context into ANN-ready features."""
    candidate = base_context.drop(columns=["units_sold", "stockout_flag"]).copy()
    candidate["our_price"] = candidate_price
    candidate["price_vs_competitor"] = (
        candidate["our_price"] / candidate["competitor_price"]
    )
    candidate["price_change"] = (
        candidate["our_price"] - candidate["lag_our_price"]
    )
    candidate["unit_margin"] = (
        candidate["our_price"] - candidate["unit_cost"]
    )
    candidate["day_of_week"] = candidate["date"].dt.dayofweek
    candidate["month_sin"] = np.sin(2 * np.pi * candidate["date"].dt.month / 12)
    candidate["month_cos"] = np.cos(2 * np.pi * candidate["date"].dt.month / 12)
    candidate = candidate.drop(columns="date")
    candidate = pd.get_dummies(
        candidate,
        columns=categorical_columns,
        dtype=int,
    )
    candidate = candidate.reindex(columns=X_train.columns, fill_value=0)
    candidate[numeric_columns] = scaler.transform(candidate[numeric_columns])
    return candidate


# Pick one context to demonstrate the optimizer.
# Change these values to the SKU and store you want to study.
selected_store = "KUL_01"
selected_sku = "SKU_001"

context = test_df[
    (test_df["store_id"] == selected_store)
    & (test_df["sku_id"] == selected_sku)
].head(1).copy()

if context.empty:
    raise ValueError("The selected store/SKU has no usable test-period context.")

current_price = float(context["our_price"].iloc[0])
unit_cost = float(context["unit_cost"].iloc[0])
competitor_price = float(context["competitor_price"].iloc[0])
available_stock = float(context["available_stock"].iloc[0])

# Guardrails: never price below a 10% margin, change price by at most 10%,
# and do not exceed 10% above the observed competitor price.
minimum_price = max(unit_cost * 1.10, current_price * 0.90)
maximum_price = min(current_price * 1.10, competitor_price * 1.10)

if minimum_price > maximum_price:
    raise ValueError("No candidate price satisfies the guardrails for this context.")

candidate_prices = np.round(
    np.arange(minimum_price, maximum_price + 0.01, 0.50),
    2,
)

recommendations = []

for price in candidate_prices:
    candidate_features = prepare_price_candidate(context, price)
    predicted_demand = max(float(model.predict(candidate_features, verbose=0)[0][0]), 0)
    expected_units_sold = min(predicted_demand, available_stock)
    expected_profit = (price - unit_cost) * expected_units_sold

    recommendations.append({
        "candidate_price": price,
        "predicted_demand": predicted_demand,
        "expected_units_sold": expected_units_sold,
        "expected_profit": expected_profit,
    })

recommendation_table = pd.DataFrame(recommendations)
recommendation_table = recommendation_table.sort_values(
    "expected_profit",
    ascending=False,
)

print("\nPRICE OPTIMIZATION DEMO")
print("Store:", selected_store, "| SKU:", selected_sku)
print(recommendation_table.round(2).to_string(index=False))
print("\nRecommended price:", recommendation_table.iloc[0]["candidate_price"])


# STEP 19: Save the trained ANN and the exact preprocessing information.
# The Streamlit app loads these files instead of training a new model for
# every user interaction.
models_directory = Path("models")
contexts_directory = Path("data/processed")
models_directory.mkdir(parents=True, exist_ok=True)
contexts_directory.mkdir(parents=True, exist_ok=True)

model.save(models_directory / "ann_demand.keras")
joblib.dump(
    {
        "scaler": scaler,
        "input_columns": X_train.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    },
    models_directory / "ann_preprocessing.joblib",
)

# Only holdout-period rows are saved for the dashboard demo. They represent
# unseen future contexts and include the history needed to calculate features.
test_df.to_csv(contexts_directory / "optimization_contexts.csv", index=False)

print("\nSaved Streamlit artifacts:")
print("- models/ann_demand.keras")
print("- models/ann_preprocessing.joblib")
print("- data/processed/optimization_contexts.csv")
