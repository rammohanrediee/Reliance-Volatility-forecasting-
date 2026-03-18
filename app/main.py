from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Reliance Stock Volatility Forecasting")

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load the pre-trained model
try:
    model_data = joblib.load('garch_model.pkl')
    garch_model = model_data['model']
    historical_returns = model_data['returns']
    print("GARCH model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    garch_model = None


@app.get("/")
async def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"), media_type="text/html")


@app.get("/api/health")
def health_check():
    """Returns server health and model status."""
    return {
        "status": "healthy",
        "model_loaded": garch_model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/forecast")
def get_forecast(days: int = 30):
    """Produce an N-day GARCH volatility forecast with 95% confidence intervals."""
    if garch_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get forecast
        forecast = garch_model.forecast(horizon=days)
        forecasted_variance = forecast.variance
        forecasted_vol = np.sqrt(forecasted_variance.values[-1, :])
        
        # Create forecast dataframe
        forecast_dates = pd.date_range(
            start=datetime.now() + timedelta(days=1), 
            periods=days, 
            freq='D'
        )
        
        forecast_data = {
            'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'Forecasted_Volatility': forecasted_vol.tolist(),
            'Upper_CI_95': (1.96 * forecasted_vol).tolist(),
            'Lower_CI_95': (-1.96 * forecasted_vol).tolist()
        }
        
        return {
            "forecast": forecast_data,
            "avg_volatility": float(np.mean(forecasted_vol)),
            "max_volatility": float(np.max(forecasted_vol)),
            "min_volatility": float(np.min(forecasted_vol))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical-volatility")
def get_historical_volatility(days: int = 90):
    """Return rolling 21-day and 60-day historical volatility for the last N trading days."""
    try:
        # Fetch latest data
        ticker = yf.Ticker("RELIANCE.NS")
        data = ticker.history(period="5y")
        
        # Calculate returns
        log_close = np.log(data['Close'])
        returns = log_close.diff().dropna() * 100
        
        # Calculate rolling volatility
        rolling_vol_21 = returns.rolling(window=21).std()
        rolling_vol_60 = returns.rolling(window=60).std()
        
        # Get last N days
        recent_dates = rolling_vol_21.tail(days).index
        recent_vol_21 = rolling_vol_21.tail(days).values
        recent_vol_60 = rolling_vol_60.tail(days).values
        
        return {
            "dates": [d.strftime('%Y-%m-%d') for d in recent_dates],
            "volatility_21d": recent_vol_21.tolist(),
            "volatility_60d": recent_vol_60.tolist(),
            "current_vol": float(rolling_vol_21.iloc[-1]),
            "avg_vol": float(rolling_vol_21.mean()),
            "max_vol": float(rolling_vol_21.max()),
            "min_vol": float(rolling_vol_21.min())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock-price")
def get_stock_price():
    """Fetch the latest closing price from Yahoo Finance."""
    try:
        ticker = yf.Ticker("RELIANCE.NS")
        data = ticker.history(period="1d")
        
        if len(data) > 0:
            current_price = float(data['Close'].iloc[-1])
            return {
                "symbol": "RELIANCE.NS",
                "current_price": current_price,
                "date": data.index[-1].strftime('%Y-%m-%d'),
                "currency": "INR"
            }
        else:
            raise HTTPException(status_code=500, detail="No data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summary")
def get_summary():
    """Build an executive summary combining historical stats with GARCH forecasts."""
    try:
        # Get historical volatility
        ticker = yf.Ticker("RELIANCE.NS")
        data = ticker.history(period="5y")
        log_close = np.log(data['Close'])
        returns = log_close.diff().dropna() * 100
        rolling_vol = returns.rolling(window=21).std()
        
        # Get forecast
        forecast = garch_model.forecast(horizon=30)
        forecasted_variance = forecast.variance
        forecasted_vol = np.sqrt(forecasted_variance.values[-1, :])
        
        return {
            "current_volatility": float(rolling_vol.iloc[-1]),
            "avg_volatility": float(rolling_vol.mean()),
            "forecast_avg_volatility": float(np.mean(forecasted_vol)),
            "volatility_trend": "increasing" if rolling_vol.iloc[-1] > rolling_vol.mean() else "stable",
            "key_findings": [
                "Returns are stationary & unpredictable",
                "Volatility is HIGHLY PREDICTABLE",
                "GARCH(1,1) captures fat tails perfectly",
                "Volatility follows distinct regimes"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
def retrain_model():
    """Fetch latest market data, re-fit GARCH(1,1), and update the saved model."""
    global garch_model, historical_returns

    try:
        ticker = yf.Ticker("RELIANCE.NS")
        data = ticker.history(period="5y")

        if data.empty or len(data) < 100:
            raise HTTPException(status_code=503, detail="Not enough live data to retrain")

        log_close = np.log(data['Close'])
        returns = log_close.diff().dropna() * 100

        model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
        fitted = model.fit(disp='off')

        model_path = os.path.join(os.path.dirname(__file__), 'garch_model.pkl')
        joblib.dump({'model': fitted, 'returns': returns}, model_path)

        garch_model = fitted
        historical_returns = returns

        return {
            "status": "retrained",
            "data_points": len(returns),
            "date_range": {
                "start": data.index[0].strftime('%Y-%m-%d'),
                "end": data.index[-1].strftime('%Y-%m-%d')
            },
            "model_params": {
                "alpha": float(fitted.params.get("alpha[1]", 0)),
                "beta": float(fitted.params.get("beta[1]", 0)),
                "persistence": float(fitted.params.get("alpha[1]", 0) + fitted.params.get("beta[1]", 0))
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")


@app.get("/api/forecast-live")
def get_live_forecast(days: int = 30):
    """Fetch fresh data, fit GARCH on the fly, and return an up-to-date forecast."""
    try:
        ticker = yf.Ticker("RELIANCE.NS")
        data = ticker.history(period="5y")

        if data.empty:
            raise HTTPException(status_code=503, detail="Could not fetch live data")

        log_close = np.log(data['Close'])
        returns = log_close.diff().dropna() * 100

        model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
        fitted = model.fit(disp='off')

        forecast = fitted.forecast(horizon=days)
        forecasted_vol = np.sqrt(forecast.variance.values[-1, :])

        forecast_dates = pd.date_range(
            start=datetime.now() + timedelta(days=1),
            periods=days,
            freq='D'
        )

        return {
            "forecast": {
                "Date": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "Forecasted_Volatility": forecasted_vol.tolist(),
                "Upper_CI_95": (1.96 * forecasted_vol).tolist(),
                "Lower_CI_95": (-1.96 * forecasted_vol).tolist()
            },
            "avg_volatility": float(np.mean(forecasted_vol)),
            "max_volatility": float(np.max(forecasted_vol)),
            "min_volatility": float(np.min(forecasted_vol)),
            "data_points_used": len(returns),
            "last_trading_date": data.index[-1].strftime('%Y-%m-%d')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
