"""
FastAPI backend for the Hotel Booking Cancellation Predictor.

Endpoints:
  GET  /           — serves the frontend SPA
  GET  /static/    — CSS, JS
  GET  /figures/   — report charts
  POST /predict    — returns cancellation prediction
  GET  /health     — health check
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.model_loader import get_model

# ─── Config ────────────────────────────────────────────────────────────────
THRESHOLD = 0.37
FRONTEND_DIR = Path(__file__).parent / "frontend"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"

# ─── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hotel Booking Cancellation Predictor",
    description="Predicts whether a hotel booking will be cancelled.",
    version="1.0.0",
)

app.mount("/static",  StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)),  name="figures")


# ─── Request schema ────────────────────────────────────────────────────────
class BookingFeatures(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    deposit_type: str
    agent: str = "Unknown"
    company: str = "Unknown"
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int


# ─── Startup ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load or train the model eagerly so the first request is fast."""
    get_model()


# ─── Routes ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/predict")
async def predict(features: BookingFeatures):
    try:
        model = get_model()

        df = pd.DataFrame([features.dict()])
        probability = float(model.predict_proba(df)[0][1])
        prediction  = int(probability >= THRESHOLD)

        if probability < 0.30:
            risk_level = "Low"
        elif probability < 0.55:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return JSONResponse({
            "prediction":  prediction,
            "probability": round(probability, 4),
            "label":       "LIKELY CANCELLED" if prediction else "BOOKING SAFE",
            "risk_level":  risk_level,
            "threshold":   THRESHOLD,
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}
