# Reliance Volatility Forecaster 📈

A FastAPI backend and interactive dashboard that forecasts the 30-day volatility of Reliance Industries (`RELIANCE.NS`). It uses a GARCH(1,1) model trained on historical Yahoo Finance data, with the ability to dynamically retrain on live market data.


## Features ⚡

- **Live Market Data:** Fetches real-time prices and returns using `yfinance`.
- **Dynamic Retraining:** A built-in `/api/retrain` endpoint re-fits the GARCH model on the latest 5 years of live data and hot-swaps the model in memory.
- **REST API:** Clean endpoints for health checks, historical volatility, and forecasts.
- **Interactive UI:** A vanilla JS/HTML dashboard to visualize volatility bands and trends.

## Tech Stack 🛠️

- **Backend:** FastAPI, Python 3.9+
- **Data & ML:** Pandas, NumPy, `arch` (for GARCH), `yfinance`
- **Frontend:** HTML, JavaScript, Chart.js

## Project Layout

```text
reliance_forecasting/
├── app/
│   ├── main.py                 # FastAPI server & endpoints
│   └── garch_model.pkl         # Saved model weights
├── static/                     # Dashboard frontend (index.html)
├── stocks_forecasting.ipynb    # Original EDA & model training notebook
└── requirements.txt            
```

## Setup & Running 🚀

1. Clone the repo and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the Uvicorn server:
```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
3. Open `http://localhost:8000` to view the dashboard, or `http://localhost:8000/docs` for the API Swagger docs.

## API Endpoints 🔌

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend dashboard |
| `GET` | `/api/health` | Server and model health check |
| `GET` | `/api/stock-price` | Current live closing price |
| `GET` | `/api/historical-volatility` | 21-day & 60-day rolling historical volatility |
| `GET` | `/api/forecast` | 30-day GARCH variance forecast using saved model |
| `GET` | `/api/forecast-live` | 30-day forecast fit dynamically on latest live data |
| `POST` | `/api/retrain` | Fetch live data, re-fit GARCH, update saved model |
| `GET` | `/api/summary` | Market trends and model diagnostics |

---

## Disclaimer ⚠️
**This project is for educational and demonstrative purposes only.** The volatility forecasting models, stock prices, and generated data do not constitute financial advice, investment advice, trading advice, or any other sort of advice. Do not make financial decisions based on the outputs of this application. Use entirely at your own risk.
