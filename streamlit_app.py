"""Interactive dashboard for the retail price-optimization project."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# Resolve all project files relative to this app. This works on your Mac and
# after deployment from a GitHub repository.
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ann_demand.keras"
PREPROCESSING_PATH = ROOT / "models" / "ann_preprocessing.joblib"
CONTEXT_PATH = ROOT / "data" / "processed" / "optimization_contexts.csv"
FUTURE_CONTEXT_PATH = ROOT / "data" / "processed" / "future_optimization_contexts.csv"

st.set_page_config(page_title="Retail Price Optimizer", page_icon="💸", layout="wide")


@st.cache_resource
def load_model_and_preprocessing() -> tuple[tf.keras.Model, dict]:
    """Load the ANN and the preprocessing objects saved during training."""
    return tf.keras.models.load_model(MODEL_PATH), joblib.load(PREPROCESSING_PATH)


@st.cache_data
def load_contexts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical backtest rows and the latest SKU-store contexts."""
    historical = pd.read_csv(CONTEXT_PATH, parse_dates=["date"])
    future = pd.read_csv(FUTURE_CONTEXT_PATH, parse_dates=["date"])
    return (
        historical.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True),
        future.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True),
    )


def candidate_to_features(
    context: pd.DataFrame,
    candidate_price: float,
    metadata: dict,
) -> pd.DataFrame:
    """Create ANN-ready features for one candidate price in one context."""
    candidate = context.copy()
    candidate["our_price"] = candidate_price
    candidate["price_vs_competitor"] = (
        candidate["our_price"] / candidate["competitor_price"]
    )
    candidate["price_change"] = (
        candidate["our_price"] - candidate["lag_our_price"]
    )
    candidate["unit_margin"] = candidate["our_price"] - candidate["unit_cost"]
    candidate["day_of_week"] = candidate["date"].dt.dayofweek
    candidate["month_sin"] = np.sin(2 * np.pi * candidate["date"].dt.month / 12)
    candidate["month_cos"] = np.cos(2 * np.pi * candidate["date"].dt.month / 12)
    candidate = candidate.drop(columns=["date", "units_sold", "stockout_flag"])
    candidate = pd.get_dummies(
        candidate,
        columns=metadata["categorical_columns"],
        dtype=int,
    )
    candidate = candidate.reindex(columns=metadata["input_columns"], fill_value=0)
    numeric_columns = metadata["numeric_columns"]
    candidate[numeric_columns] = metadata["scaler"].transform(
        candidate[numeric_columns]
    )
    return candidate


def optimize_price(
    context: pd.DataFrame,
    model: tf.keras.Model,
    metadata: dict,
    max_price_move_percent: float,
    maximum_competitor_premium_percent: float,
    minimum_margin_percent: float,
    price_step: float,
) -> pd.DataFrame:
    """Run constrained grid-search profit optimization for one context."""
    cost = float(context["unit_cost"].iloc[0])
    current_price = float(context["our_price"].iloc[0])
    competitor_price = float(context["competitor_price"].iloc[0])
    stock = float(context["available_stock"].iloc[0])

    minimum_price = max(
        cost * (1 + minimum_margin_percent / 100),
        current_price * (1 - max_price_move_percent / 100),
    )
    maximum_price = min(
        current_price * (1 + max_price_move_percent / 100),
        competitor_price * (1 + maximum_competitor_premium_percent / 100),
    )
    if minimum_price > maximum_price:
        raise ValueError("No candidate price satisfies the current guardrails.")

    candidate_prices = np.round(
        np.arange(minimum_price, maximum_price + price_step / 2, price_step), 2
    )
    outcomes: list[dict[str, float]] = []

    for price in candidate_prices:
        features = candidate_to_features(context, float(price), metadata)
        predicted_demand = max(float(model.predict(features, verbose=0)[0][0]), 0.0)
        expected_units_sold = min(predicted_demand, stock)
        expected_profit = (float(price) - cost) * expected_units_sold
        outcomes.append(
            {
                "candidate_price": float(price),
                "predicted_demand": predicted_demand,
                "expected_units_sold": expected_units_sold,
                "expected_profit": expected_profit,
            }
        )

    return pd.DataFrame(outcomes).sort_values("candidate_price").reset_index(drop=True)


st.title("Retail Price Optimizer")
st.caption("ANN demand forecast + constrained grid-search profit optimization")

if not (
    MODEL_PATH.exists()
    and PREPROCESSING_PATH.exists()
    and CONTEXT_PATH.exists()
    and FUTURE_CONTEXT_PATH.exists()
):
    st.warning("Train the project first to create the dashboard artifacts.")
    st.code("python3 scripts/train_price_demand.py", language="bash")
    st.stop()

model, metadata = load_model_and_preprocessing()
historical_contexts, future_contexts = load_contexts()

with st.sidebar:
    mode = st.segmented_control(
        "Recommendation mode",
        ["Historical backtest", "Future recommendation"],
        default="Historical backtest",
        required=True,
        key="recommendation_mode",
        width="stretch",
    )

    st.header("Select a pricing context")
    if mode == "Historical backtest":
        selected_store = st.selectbox(
            "Store",
            sorted(historical_contexts["store_id"].unique()),
            key="historical_store",
        )
        store_contexts = historical_contexts[
            historical_contexts["store_id"] == selected_store
        ]
        selected_sku = st.selectbox(
            "SKU",
            sorted(store_contexts["sku_id"].unique()),
            key="historical_sku",
        )
        sku_contexts = store_contexts[store_contexts["sku_id"] == selected_sku]
        selected_date = st.selectbox(
            "Historical decision date",
            sorted(sku_contexts["date"].dt.date.unique()),
            key="historical_date",
        )
        context = sku_contexts[
            sku_contexts["date"].dt.date == selected_date
        ].iloc[[0]].copy()
        display_date = selected_date
    else:
        selected_store = st.selectbox(
            "Store",
            sorted(future_contexts["store_id"].unique()),
            key="future_store",
        )
        store_contexts = future_contexts[future_contexts["store_id"] == selected_store]
        selected_sku = st.selectbox(
            "SKU",
            sorted(store_contexts["sku_id"].unique()),
            key="future_sku",
        )
        context = store_contexts[store_contexts["sku_id"] == selected_sku].iloc[[0]].copy()
        latest_date = context["date"].iloc[0].date()
        selected_date = st.date_input(
            "Future decision date",
            value=latest_date + timedelta(days=1),
            min_value=latest_date + timedelta(days=1),
            key="future_date",
        )
        st.caption(f"Latest available history: {latest_date}")
        context["date"] = pd.Timestamp(selected_date)
        context["our_price"] = st.number_input(
            "Current price",
            min_value=0.01,
            value=float(context["our_price"].iloc[0]),
            step=0.01,
            format="%.2f",
            key="future_current_price",
        )
        context["competitor_price"] = st.number_input(
            "Expected competitor price",
            min_value=0.01,
            value=float(context["competitor_price"].iloc[0]),
            step=0.01,
            format="%.2f",
            key="future_competitor_price",
        )
        context["available_stock"] = st.number_input(
            "Available stock",
            min_value=0,
            value=int(context["available_stock"].iloc[0]),
            step=1,
            key="future_stock",
        )
        context["promotion_flag"] = int(st.checkbox("Planned promotion", key="future_promotion"))
        context["is_holiday"] = int(st.checkbox("Holiday period", key="future_holiday"))
        display_date = selected_date

    st.header("Guardrails")
    max_price_move = st.slider("Maximum price change (%)", 1, 30, 10)
    competitor_premium = st.slider("Maximum competitor premium (%)", 0, 30, 10)
    minimum_margin = st.slider("Minimum margin above unit cost (%)", 0, 50, 10)
    price_step = st.select_slider("Candidate price step", options=[0.10, 0.25, 0.50, 1.00], value=0.50)

st.subheader(f"{mode}: {selected_store} - {selected_sku} - {display_date}")
overview = st.columns(4)
overview[0].metric("Current price", f"{context['our_price'].iloc[0]:.2f}")
overview[1].metric("Competitor price", f"{context['competitor_price'].iloc[0]:.2f}")
overview[2].metric("Unit cost", f"{context['unit_cost'].iloc[0]:.2f}")
overview[3].metric("Available stock", f"{context['available_stock'].iloc[0]:.0f}")

try:
    results = optimize_price(
        context,
        model,
        metadata,
        max_price_move,
        competitor_premium,
        minimum_margin,
        price_step,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

best = results.loc[results["expected_profit"].idxmax()]

# Baseline: estimate profit at the current price using the same demand model.
# This makes the uplift a fair what-if comparison with the recommendation.
current_price = float(context["our_price"].iloc[0])
current_features = candidate_to_features(context, current_price, metadata)
current_predicted_demand = max(
    float(model.predict(current_features, verbose=0)[0][0]),
    0.0,
)
current_expected_units = min(
    current_predicted_demand,
    float(context["available_stock"].iloc[0]),
)
current_expected_profit = (
    current_price - float(context["unit_cost"].iloc[0])
) * current_expected_units
expected_profit_uplift = best["expected_profit"] - current_expected_profit
uplift_percent = (
    expected_profit_uplift / current_expected_profit * 100
    if current_expected_profit > 0
    else 0.0
)

recommended = st.columns(4)
recommended[0].metric(
    "Recommended price",
    f"{best['candidate_price']:.2f}",
    delta=f"{best['candidate_price'] - context['our_price'].iloc[0]:+.2f} vs current",
)
recommended[1].metric("Predicted demand", f"{best['predicted_demand']:.1f} units")
recommended[2].metric("Expected profit", f"{best['expected_profit']:.2f}")
recommended[3].metric(
    "Expected profit uplift",
    f"{expected_profit_uplift:+.2f}",
    delta=f"{uplift_percent:+.1f}% vs current price",
)

st.subheader("Candidate-price comparison")
st.line_chart(results.set_index("candidate_price")["expected_profit"])
st.dataframe(
    results,
    column_config={
        "candidate_price": st.column_config.NumberColumn("Candidate price", format="%.2f"),
        "predicted_demand": st.column_config.NumberColumn("Predicted demand", format="%.2f"),
        "expected_units_sold": st.column_config.NumberColumn("Expected units sold", format="%.2f"),
        "expected_profit": st.column_config.NumberColumn("Expected profit", format="%.2f"),
    },
    hide_index=True,
)

with st.expander("Selected context and model note"):
    st.write(context.drop(columns=["units_sold", "stockout_flag"]).T)
    st.caption(
        "This is an offline recommendation. The ANN estimates demand under each "
        "candidate price; a real deployment should validate pricing changes with "
        "controlled experiments."
    )
