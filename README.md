# 🚚 Supply Chain Lead Time Prediction

Predicting shipment delivery time for a logistics company using machine learning — comparing Linear Regression vs. Decision Tree models across 5,000 global shipments.

---

## Business Problem

**TransRoute Logistics** was losing customer trust due to inaccurate delivery estimates. Manual averages weren't accounting for real-world variables like weather disruptions, route risk, carrier reliability, and transport mode.

**Goal:** Build an ML model that predicts lead time (days) before a shipment is dispatched — replacing guesswork with data.

---

## Dataset

| Field | Detail |
|---|---|
| Source | `global_supply_chain_risk_2026.csv` |
| Records | 5,000 shipments |
| Target | `Lead_Time_Days` (range: 1–237 days) |
| No nulls | ✅ Clean dataset, no missing values |

**Features used:**
- `Origin_Port`, `Destination_Port` — route endpoints
- `Transport_Mode` — Air, Rail, Road, Sea
- `Product_Category` — Automotive, Electronics, Perishables, Pharmaceuticals, Textiles
- `Distance_km`, `Weight_MT` — shipment size/distance
- `Fuel_Price_Index` — operational cost factor
- `Geopolitical_Risk_Score` — route-level risk (0–10)
- `Weather_Condition` — Clear, Fog, Rain, Storm, Hurricane
- `Carrier_Reliability_Score` — carrier performance score (0–1)
- `Month` — seasonality

---

## ML Pipeline

```
Raw CSV
  ↓
Feature Engineering (Date → Month)
  ↓
Feature Selection (drop Shipment_ID, Date)
  ↓
One-Hot Encoding (32 features after encoding)
  ↓
Train-Test Split (80/20, random_state=42)
  ↓
Model 1: Linear Regression
Model 2: Decision Tree (max_depth=5)
  ↓
Evaluation: MAE, RMSE, R²
```

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 12.06 | 18.65 | 0.614 |
| **Decision Tree** | **5.11** | **7.52** | **0.937** |

**Winner: Decision Tree** — 93.7% of variance explained, 58% lower MAE than Linear Regression.

> Linear Regression produced 231 negative predictions (physically impossible). After clipping to a minimum of 1 day, the issue was resolved — but the Decision Tree had zero negative predictions naturally.

---

## Key Findings — Feature Importance (Decision Tree)

| Feature | Importance |
|---|---|
| Weather: Hurricane | 31.0% |
| Transport Mode: Sea | 28.2% |
| Distance (km) | 26.9% |
| Transport Mode: Road | 5.0% |
| Transport Mode: Rail | 4.4% |

**Top driver:** Hurricane conditions are the single biggest predictor of delivery delays — followed closely by sea transport and route distance.

---

## Business Recommendations

1. **Flag hurricane-prone routes proactively** — highest impact on lead time
2. **Sea shipments need wider delivery windows** — 28% of predictive power
3. **Prioritize carrier reliability on long-haul routes** — distance × reliability interact strongly
4. **Deploy the Decision Tree model** for dispatch planning — R² of 0.937 is production-ready

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation & EDA |
| Scikit-learn | ML models, train-test split, metrics |
| Seaborn | Visualization (lead time distribution) |
| NumPy | Numerical operations |
| Google Colab | Development environment |

---

## Files

```
supply-chain-lead-time-prediction/
├── supply_chain_lead_time_prediction.ipynb   # Full ML notebook
├── global_supply_chain_risk_2026.csv         # Dataset (5,000 shipments)
└── README.md
```

---

## Author

**Manoj Pedarla** — Data Analyst, Dallas TX  
[LinkedIn](https://www.linkedin.com/in/manojpedarla) · [Portfolio](https://data-analyst-portfolio-two-rho.vercel.app/)
