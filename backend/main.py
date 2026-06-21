from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import numpy as np
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_models():
    required = [
        "xgb_clean.pkl",
        "feature_cols_clean.pkl",
        "global_tenc_clean.pkl",
        "label_encoders_clean.pkl",
        
        "xgb_road_closure.pkl",
        "feature_cols_B.pkl",
    ]
    missing = []
    for fname in required:
        path = os.path.join(MODEL_DIR, fname)
        key  = fname.replace(".pkl", "")
        if os.path.exists(path):
            MODELS[key] = joblib.load(path)
            logger.info(f"Loaded {fname}")
        else:
            missing.append(fname)
            logger.warning(f"Missing {fname}")

    if missing:
        logger.warning(f"Missing models: {missing}")
    else:
        logger.info("All models loaded successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(
    title="Event Congestion Prediction API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://evensaathi.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# schemas
class PredictionRequest(BaseModel):
    event_type: str
    event_cause: str
    latitude: float
    longitude: float
    hour: int
    day_of_week: int
    zone: Optional[str] = None
    junction: Optional[str] = None
    veh_type: Optional[str] = None
    requires_road_closure: bool = False
    has_end_location: bool = False
    corridor: Optional[str] = None
    month: Optional[int] = None      
    priority: Optional[str] = None  

class ResourceRecommendation(BaseModel):
    manpower_min: int
    manpower_max: int
    barricades: int
    diversion_needed: bool
    diversion_zones: List[str]

class SimilarEvent(BaseModel):
    id: str
    cause: str
    zone: str
    duration_mins: int
    date: str
    severity: str

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class PredictionResponse(BaseModel):
    duration_mins: float
    duration_label: str
    severity: str
    severity_score: int
    road_closure_prob: float
    road_closure_label: str
    resources: ResourceRecommendation
    similar_events: List[SimilarEvent]
    feature_importances: List[FeatureImportance]
    model_used: str
    confidence: str

# feature builder
PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}

def build_feature_row(req: PredictionRequest) -> dict:
    hour       = req.hour
    dow        = req.day_of_week
    month      = req.month if req.month is not None else 1
    lat        = req.latitude
    lon        = req.longitude
    le         = MODELS.get("label_encoders_clean", {})
    tenc       = MODELS.get("global_tenc_clean", {})

    # label encode
    def safe_le(col, val):
        encoder = le.get(col)
        if encoder is None:
            return -999
        try:
            return int(encoder.transform([str(val)])[0])
        except ValueError:
            return int(encoder.transform([encoder.classes_[0]])[0])

    event_type_enc  = safe_le("event_type",  req.event_type)
    event_cause_enc = safe_le("event_cause", req.event_cause)
    veh_type_enc    = safe_le("veh_type", req.veh_type or "Unknown")

    priority_enc = PRIORITY_MAP.get((req.priority or "medium").lower(), 1)

    is_weekend   = int(dow in [5, 6])
    is_peak      = int(hour in [7, 8, 9, 17, 18, 19])
    is_peak_am   = int(hour in [7, 8, 9])
    is_peak_pm   = int(hour in [17, 18, 19])
    is_night     = int(hour in [22, 23, 0, 1, 2])
    is_monday    = int(dow == 0)
    is_friday    = int(dow == 4)
    quarter      = (month - 1) // 3 + 1
    week_of_year = min(52, max(1, int((month - 1) * 4.33 + 1)))

    hour_sin  = np.sin(2 * np.pi * hour  / 24)
    hour_cos  = np.cos(2 * np.pi * hour  / 24)
    dow_sin   = np.sin(2 * np.pi * dow   / 7)
    dow_cos   = np.cos(2 * np.pi * dow   / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # geo
    lat_min, lat_max = 12.80, 13.27
    lon_min, lon_max = 77.31, 77.77
    lat_bin = int(np.clip((lat - lat_min) / (lat_max - lat_min) * 8, 0, 7))
    lon_bin = int(np.clip((lon - lon_min) / (lon_max - lon_min) * 8, 0, 7))
    has_end_location  = int(req.has_end_location)
    dist_from_center  = float(np.sqrt((lat - 12.9716) ** 2 + (lon - 77.5946) ** 2))

    # target encoding
    gmean = float(np.mean([v for d in tenc.values()
                           if isinstance(d, dict) for v in d.values()] or [89.0]))

    def tenc_val(col, val):
        col_map = tenc.get(col, {})
        return float(col_map.get(str(val), gmean))

    zone     = req.zone     or "Unknown"
    junction = req.junction or "Unknown"
    corridor = req.corridor or "Unknown"

    zone_tenc         = tenc_val("zone", zone)
    junction_tenc     = tenc_val("junction", junction)
    corridor_tenc     = tenc_val("corridor", corridor)
    event_cause_tenc  = tenc_val("event_cause", req.event_cause)

    zone_count = junction_count = corridor_count = event_cause_count = 1.0

    cause_x_peak    = event_cause_enc * is_peak
    cause_x_weekend = event_cause_enc * is_weekend
    cause_x_closure = event_cause_enc * int(req.requires_road_closure)
    zone_x_peak     = zone_tenc * is_peak
    closure_x_end   = int(req.requires_road_closure) * has_end_location

    return {
        "hour": hour, "day_of_week": dow, "month": month,
        "week_of_year": week_of_year, "quarter": quarter,
        "is_weekend": is_weekend, "is_peak": is_peak,
        "is_peak_am": is_peak_am, "is_peak_pm": is_peak_pm,
        "is_night": is_night,
        "is_monday": is_monday, "is_friday": is_friday,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "latitude": lat, "longitude": lon,
        "lat_bin": lat_bin, "lon_bin": lon_bin,
        "has_end_location": has_end_location,
        "dist_from_center": dist_from_center,
        "event_type_enc": event_type_enc,
        "event_cause_enc": event_cause_enc,
        "veh_type_enc": veh_type_enc,
        "requires_road_closure": int(req.requires_road_closure),
        "priority_enc": priority_enc,

        "zone_tenc": zone_tenc, "zone_count": zone_count,
        "junction_tenc": junction_tenc, "junction_count": junction_count,
        "corridor_tenc": corridor_tenc, "corridor_count": corridor_count,
        "event_cause_tenc": event_cause_tenc, "event_cause_count": event_cause_count,

        "cause_x_peak": cause_x_peak, "cause_x_weekend": cause_x_weekend,
        "cause_x_closure": cause_x_closure,
        "zone_x_peak": zone_x_peak, "closure_x_end": closure_x_end,
    }

def row_to_array(row: dict, feature_cols: list) -> np.ndarray:
    return np.array([[row.get(c, -999) for c in feature_cols]], dtype=np.float32)

def real_predict(req: PredictionRequest):
    row = build_feature_row(req)

    feat_A = MODELS["feature_cols_clean"]
    X_A    = row_to_array(row, feat_A)
    log_pred = float(MODELS["xgb_clean"].predict(X_A)[0])
    duration  = float(np.expm1(log_pred))
    duration  = max(5.0, round(duration, 1))

    if duration < 30:
        sev_class = 0    # Low
    elif duration < 90:
        sev_class = 1    # Medium
    elif duration < 240:
        sev_class = 2    # High
    else:
        sev_class = 3    # Critical

    feat_B = MODELS["feature_cols_B"]
    X_B    = row_to_array(row, feat_B)
    closure_prob = float(MODELS["xgb_road_closure"].predict_proba(X_B)[0][1])
    closure_prob = round(float(np.clip(closure_prob, 0.0, 1.0)), 3)

    return duration, closure_prob, sev_class

SEV_LABELS  = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
SEV_SCORES  = {0: 18,    1: 42,       2: 72,      3: 92}

DIVERSION_ZONES = {
    "Central Zone": ["Outer Ring Road via Hebbal", "Tumkur Road diversion"],
    "North Zone":   ["Bellary Road alternate", "NH44 bypass"],
    "South Zone":   ["Kanakapura Road bypass", "Bannerghatta Road alternate"],
    "East Zone":    ["Old Madras Road alternate", "Whitefield bypass"],
    "West Zone":    ["Magadi Road bypass", "Mysore Road alternate"],
}

def build_resources(severity: str, closure_prob: float, zone: str) -> ResourceRecommendation:
    manpower_map  = {"Low": (4,6),  "Medium": (8,12),  "High": (14,18),  "Critical": (20,28)}
    barricade_map = {"Low": 2,      "Medium": 4,        "High": 8,        "Critical": 14}
    diversion     = closure_prob > 0.4 or severity in ("High", "Critical")
    zone_key      = next((k for k in DIVERSION_ZONES if zone and k.lower() in zone.lower()), "Central Zone")
    lo, hi        = manpower_map[severity]
    return ResourceRecommendation(
        manpower_min=lo, manpower_max=hi,
        barricades=barricade_map[severity],
        diversion_needed=diversion,
        diversion_zones=DIVERSION_ZONES[zone_key] if diversion else [],
    )

def duration_label(duration: float) -> str:
    if duration < 60:
        return f"{int(duration)} mins"
    h, m = divmod(int(duration), 60)
    return f"{h}h {m}m" if m else f"{h}h"

def closure_label(prob: float) -> str:
    if prob > 0.7:  return "Very likely"
    if prob > 0.4:  return "Likely"
    if prob > 0.2:  return "Possible"
    return "Unlikely"

def get_feature_importances() -> List[FeatureImportance]:
    model  = MODELS.get("xgb_clean")
    feat_A = MODELS.get("feature_cols_clean", [])
    if model is None or not feat_A:
        return []
    scores = model.feature_importances_
    pairs  = sorted(zip(feat_A, scores), key=lambda x: x[1], reverse=True)[:10]
    total  = sum(s for _, s in pairs) or 1.0
    return [FeatureImportance(feature=f, importance=round(float(s/total), 4))
            for f, s in pairs]

# similar data points 
SIMILAR_BANK = [
    SimilarEvent(id="FKID001245", cause="vehicle_breakdown", zone="Central Zone 1", duration_mins=48,  date="2024-03-12", severity="Medium"),
    SimilarEvent(id="FKID002312", cause="vehicle_breakdown", zone="North Zone 2",   duration_mins=35,  date="2024-04-05", severity="Low"),
    SimilarEvent(id="FKID003812", cause="tree_fall",         zone="North Zone 2",   duration_mins=95,  date="2024-02-08", severity="High"),
    SimilarEvent(id="FKID004501", cause="tree_fall",         zone="South Zone 1",   duration_mins=110, date="2024-05-15", severity="High"),
    SimilarEvent(id="FKID005234", cause="accident",          zone="East Zone 1",    duration_mins=145, date="2024-01-22", severity="High"),
    SimilarEvent(id="FKID006134", cause="accident",          zone="South Zone 1",   duration_mins=132, date="2024-04-21", severity="High"),
    SimilarEvent(id="FKID006900", cause="accident",          zone="Central Zone 2", duration_mins=75,  date="2024-06-01", severity="Medium"),
    SimilarEvent(id="FKID002901", cause="festival",          zone="Central Zone 2", duration_mins=210, date="2024-01-14", severity="Critical"),
    SimilarEvent(id="FKID007100", cause="festival",          zone="North Zone 1",   duration_mins=180, date="2024-03-28", severity="High"),
    SimilarEvent(id="FKID007563", cause="construction",      zone="East Zone 1",    duration_mins=300, date="2024-05-03", severity="Critical"),
    SimilarEvent(id="FKID008001", cause="political_rally",   zone="Central Zone 1", duration_mins=240, date="2024-02-18", severity="Critical"),
    SimilarEvent(id="FKID008432", cause="waterlogging",      zone="South Zone 2",   duration_mins=65,  date="2024-07-10", severity="Medium"),
    SimilarEvent(id="FKID009100", cause="others",            zone="West Zone 1",    duration_mins=40,  date="2024-04-30", severity="Medium"),
]

def get_similar_events(cause: str, zone: str, n: int = 3) -> List[SimilarEvent]:
    same_both  = [e for e in SIMILAR_BANK if e.cause == cause and zone and e.zone.split()[0] in zone]
    same_cause = [e for e in SIMILAR_BANK if e.cause == cause and e not in same_both]
    pool = (same_both + same_cause + SIMILAR_BANK)[:n*3]
    seen, result = set(), []
    for e in pool:
        if e.id not in seen:
            seen.add(e.id)
            result.append(e)
        if len(result) == n:
            break
    return result

@app.get("/")
def root():
    return {"status": "ok", "message": "Event Congestion Prediction API v3"}

@app.get("/health")
def health():
    loaded = list(MODELS.keys())
    required_keys = [
        "xgb_clean", "feature_cols_clean", "global_tenc_clean",
        "label_encoders_clean", "xgb_road_closure", "feature_cols_B",
    ]
    all_ok = all(k in MODELS for k in required_keys)
    return {
        "status": "healthy" if all_ok else "degraded",
        "models_loaded": all_ok,
        "loaded_keys": loaded,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        duration, closure_prob, sev_class = real_predict(req)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    row = build_feature_row(req)
    feat_A = MODELS["feature_cols_clean"]
    X_A    = row_to_array(row, feat_A)

    severity       = SEV_LABELS[sev_class]
    severity_score = SEV_SCORES[sev_class]
    resources      = build_resources(severity, closure_prob, req.zone or "Central Zone")

    # Confidence based on how far duration is from a severity boundary
    boundaries = [30, 90, 240]
    min_dist   = min(abs(duration - b) for b in boundaries)
    confidence = "High" if min_dist > 20 else "Medium"

    return PredictionResponse(
        duration_mins=duration,
        duration_label=duration_label(duration),
        severity=severity,
        severity_score=severity_score,
        road_closure_prob=closure_prob,
        road_closure_label=closure_label(closure_prob),
        resources=resources,
        similar_events=get_similar_events(req.event_cause, req.zone or ""),
        feature_importances=get_feature_importances(),
        model_used="XGBoost",
        confidence=confidence,
    )

@app.get("/options")
def options():
    return {
        "event_types": ["planned", "unplanned"],
        "event_causes": [
            "vehicle_breakdown",
            "accident",
            "tree_fall",
            "construction",
            "water_logging",
            "congestion",
            "procession",
            "protest",
            "road_conditions",
            "pot_holes",
            "others",
        ],
        "veh_types": [
            "Unknown",
            "private_car",
            "truck",
            "auto",
            "heavy_vehicle",
            "bmtc_bus",
            "ksrtc_bus",
            "private_bus",
            "lcv",
            "taxi",
            "others",
        ],
        "zones": [
            "Central Zone 1", "Central Zone 2",
            "North Zone 1",   "North Zone 2",
            "South Zone 1",   "South Zone 2",
            "East Zone 1",    "East Zone 2",
            "West Zone 1",    "West Zone 2",
            "Unknown",
        ],
        "junctions": ["Unknown"],   # 229 junctions - too many for dropdown
        "priorities": ["low", "medium", "high", "critical"],
    }