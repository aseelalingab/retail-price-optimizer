# Retail Price Optimizer

An end-to-end machine-learning project that forecasts retail demand and recommends a profit-maximizing product price under practical business constraints.

The project combines an Artificial Neural Network (ANN) demand model with constrained grid-search optimization. It is presented through an interactive Streamlit dashboard where a user can select a store and SKU, set pricing guardrails, and receive a recommended price with its expected demand, profit, and profit uplift.

## Project objective

Retailers need to balance demand and margin: a lower price may sell more units, while a higher price may earn more profit per unit. The objective of this project is to recommend the price that maximizes expected profit for a given store, product, and decision date.

For every candidate price, the system estimates demand and calculates:

```text
expected profit = (candidate price - unit cost) × expected units sold
```

The recommended price is the candidate with the highest expected profit while satisfying the chosen business constraints.

## Solution overview

```mermaid
flowchart LR
    A[Historical retail data] --> B[Feature engineering]
    B --> C[Chronological train / validation / test split]
    C --> D[ANN demand model]
    D --> E[Candidate prices within guardrails]
    E --> F[Predicted demand for every candidate]
    F --> G[Expected-profit calculation]
    G --> H[Recommended price and uplift]

    classDef data fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef model fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef optimization fill:#FFF3E0,stroke:#EF6C00,color:#E65100
    classDef outcome fill:#E0F2F1,stroke:#00897B,color:#004D40
    class A,B,C data
    class D model
    class E,F,G optimization
    class H outcome
```

## Key features

- **Demand forecasting:** predicts units sold from price, competitor price, product, store, calendar, promotion, inventory, and historical-demand features.
- **ANN model selection:** compares the ANN with average-sales, previous-day-sales, Random Forest, and Gradient Boosting baselines.
- **Time-aware evaluation:** uses chronological train, validation, and test splits to avoid using future information when evaluating the model.
- **Constrained price optimization:** evaluates a transparent grid of allowable prices rather than relying on a fixed rule.
- **Business guardrails:** supports a maximum price movement, a minimum margin above unit cost, and a competitor-price premium limit.
- **Inventory awareness:** caps expected units sold at available stock.
- **Historical and future modes:** backtests a historical decision context or creates a one-day-ahead recommendation using the latest available SKU-store history and user-provided assumptions.

## Dataset and features

The project uses daily retail observations at the store-SKU level. The dataset contains pricing, demand, competitor, promotion, calendar, and inventory information required to model a realistic pricing decision.

Each row includes the following information:

| Feature group | Examples |
| --- | --- |
| Product and store | `sku_id`, `category`, `store_id`, `store_region` |
| Price and cost | `our_price`, `competitor_price`, `unit_cost` |
| Commercial context | `promotion_flag`, `is_holiday`, `available_stock` |
| Historical behaviour | previous price, 1-day and 2-day sales lags, 7-day rolling average sales |
| Target | `units_sold` |

Price-derived features are recalculated for every candidate price: price versus competitor, price change, and unit margin. This is important because the ANN must see the correct feature values for each price scenario.

## Model evaluation

The ANN was selected because it achieved the lowest validation MAE among the evaluated methods.

| Method | Validation MAE |
| --- | ---: |
| Average-sales baseline | 21.835 |
| Previous-day-sales baseline | 17.834 |
| Random Forest | 11.583 |
| Gradient Boosting | 11.646 |
| **ANN** | **10.740** |

Final holdout test MAE: **10.642 units**.

Average actual sales in the test set were **49.043 units**. Compared with this average, the model's MAE of 10.642 units corresponds to a simple accuracy proxy of **78.3%**. In other words, the model's average prediction error was about 21.7% of typical daily sales.

The 78.3% figure is a useful way to summarise model performance, but it is not a guarantee of future pricing results. The ANN also reduced validation error by approximately 51% compared with the average-sales baseline and 40% compared with the previous-day-sales baseline.

## Optimization method

This project uses **constrained grid-search profit optimization**.

For one SKU-store context, the optimizer creates a set of candidate prices using a selected price step. For each price, it:

1. updates the price-related features;
2. passes the scenario to the trained ANN to predict demand;
3. limits expected sales to available inventory;
4. calculates expected profit; and
5. selects the price with the highest expected profit.

The candidate price range is constrained by:

```text
minimum price = max(unit cost × (1 + minimum margin), current price × (1 - maximum change))
maximum price = min(current price × (1 + maximum change), competitor price × (1 + competitor premium))
```

Grid search is a good fit for a single-product pricing decision because it is easy to explain, auditable, and evaluates every permitted price in the chosen range. More advanced methods such as genetic algorithms or particle swarm optimization would be more suitable for a joint optimization problem involving many products and shared constraints.

## Future recommendation mode

Future dates do not already exist in the dataset. To generate a one-day-ahead recommendation, the dashboard:

1. retrieves the latest available history for the selected store and SKU;
2. lets the user enter known or expected future conditions, including date, current price, competitor price, stock, promotion, and holiday status;
3. derives the future calendar features from the chosen date;
4. evaluates allowable candidate prices with the saved ANN; and
5. recommends the price with the highest expected profit.

This is a scenario-based forecast, not a guarantee. For decisions further into the future, lagged-demand inputs would need to be forecast recursively, so uncertainty increases with the forecast horizon.

## Dashboard

The Streamlit application provides:

- historical backtest and future recommendation modes;
- store, SKU, and date selection;
- editable future business assumptions;
- adjustable optimization guardrails;
- recommended price, predicted demand, expected profit, and expected profit uplift; and
- a chart and table comparing all candidate prices.

### Historical backtest

![Historical backtest dashboard](assets/screenshots/historical-backtest.png)

### Future recommendation

![Future recommendation dashboard](assets/screenshots/future-recommendation.png)

## Project structure

```text
.
├── streamlit_app.py                       # Interactive dashboard
├── scripts/
│   └── train_price_demand.py              # Feature engineering, training, evaluation, artifact creation
├── models/
│   ├── ann_demand.keras                   # Saved ANN model
│   └── ann_preprocessing.joblib           # Scaler and feature metadata
├── data/processed/
│   ├── optimization_contexts.csv          # Holdout contexts for historical backtesting
│   └── future_optimization_contexts.csv   # Latest context per store and SKU
├── assets/screenshots/                     # Dashboard screenshots for this README
├── requirements.txt
└── README.md
```

## Skills demonstrated

Python · Pandas · Scikit-learn · TensorFlow/Keras · Artificial Neural Networks · Feature Engineering · Time-Series Validation · Regression Evaluation · Profit Optimization · Streamlit · Model Deployment
 Scikit-learn · TensorFlow/Keras · Artificial Neural Networks · Feature Engineering · Time-Series Validation · Regression Evaluation · Profit Optimization · Streamlit · Model Deployment
