# Prior Authorization Demand Forecasting
### RNN/LSTM Model — Medi-Cal Managed Care

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare%20ML-green)

---

## Business Problem

The Utilization Management (UM) team at a Medi-Cal managed care organization was staffing reactively — no forecasting capability meant authorization backlogs built unpredictably, turnaround SLAs were missed, and DHCS audit findings were increasing.

UM leadership needed 4 weeks of forward visibility into authorization volumes to plan staffing proactively.

## Solution

End-to-end time-series forecasting pipeline using a 2-layer stacked LSTM neural network trained on weekly prior authorization volume, broken down by service category and urgency level.

## Results

| Metric | Value |
|--------|-------|
| Forecast error reduction vs. baseline | **35%** |
| Manual process replaced | Decade-old spreadsheet |
| Validation method | Walk-forward time-series CV |
| Production deployment | Weekly batch forecast |

## Technical Approach

```
Raw auth data (EREX/TruCare)
        ↓
Feature engineering (lag features, rolling stats, seasonal lags)
        ↓
Time-series cross-validation (walk-forward, no data leakage)
        ↓
2-layer LSTM (Adam optimizer, dropout regularization)
        ↓
SHAP explainability reports
        ↓
Model card + deployment
```

**Why LSTM over ARIMA?**  
Authorization volume has long-range temporal dependencies — quarterly seasonal patterns, lag effects from approval backlogs, and interactions between service categories. LSTM captures these better than classical time-series methods.

**Why time-series cross-validation?**  
Standard k-fold would leak future data into training sets. Walk-forward validation mirrors exactly how the model runs in production: always trained on past, predicting future.

## Files

```
prior_auth_forecasting/
├── prior_auth_forecasting.ipynb   ← Full training pipeline
├── sample_data.csv                ← Synthetic data (Disney characters — no PHI)
├── charts/
│   ├── volume_analysis.png        ← EDA charts
│   ├── forecast_vs_actual.png     ← Model performance
│   └── shap_summary.png           ← SHAP feature importance
└── models/
    ├── prior_auth_lstm_v1.h5      ← Saved model
    └── model_card_v1.json         ← Model documentation
```

## How to Run

```bash
# Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib shap

# Launch notebook
jupyter notebook prior_auth_forecasting.ipynb
```

## Data Note

All member and provider names in the sample dataset are fictional (Disney characters).  
No PHI or real patient data is used anywhere in this project.

## Author

**Sorav Bhatia** — Applied AI Scientist | Healthcare ML  
[Portfolio](https://huggingface.co/spaces/sb384/sorav-bhatia-portfolio)
