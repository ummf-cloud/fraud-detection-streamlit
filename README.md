# IEEE-CIS Fraud Detection — Streamlit App

**Author:** Khang Le | Milestone 4

## What this app does
Real-time fraud scoring using a tuned LightGBM model trained on the
IEEE-CIS Fraud Detection dataset (Vesta Corporation / Kaggle).

- Enter values for the top transaction features using the sidebar sliders
- Click **Score Transaction** to get an instant fraud probability
- View a **SHAP waterfall chart** explaining exactly which features drove the prediction

## Model Performance (Test Set)
| Metric | Score |
|--------|-------|
| AUC-ROC | See app |
| Avg Precision | See app |
| F1-Score | See app |

## Files in this repo
| File | Purpose |
|------|---------|
| `fraud_app.py` | Streamlit application |
| `final_model_lgbm.txt` | Trained LightGBM model |
| `imputer.pkl` | Median imputer |
| `scaler.pkl` | StandardScaler |
| `shap_explainer.pkl` | SHAP TreeExplainer |
| `model_config.json` | Threshold, feature list, metrics |
| `X_train.csv` | Training feature means (for default inputs) |
| `requirements.txt` | Python dependencies |

## Run locally
```bash
pip install -r requirements.txt
streamlit run fraud_app.py
```
