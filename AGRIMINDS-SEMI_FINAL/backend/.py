"""
AgriMinds Backend API
=====================
FastAPI application serving ML predictions and satellite data integration.

Endpoints:
1. POST /predict/crop - Crop recommendation based on soil parameters
2. POST /predict/fertilizer - NPK fertilizer plan with satellite data fusion
3. GET /satellite/ndvi - Mock NDVI retrieval (Sentinel-2 simulation)

Architecture: Async/Await for scalable concurrent request handling
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any
import joblib
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import asyncio

# Initialize FastAPI app
app = FastAPI(
    title="AgriMinds API",
    description="Satellite-Enhanced Precision Agriculture Platform",
    version="1.0.0"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS (Request/Response Schemas)
# ============================================================================

class SoilData(BaseModel):
    """Soil parameters from ground sensors"""
    N: float = Field(..., ge=0, le=200, description="Nitrogen (mg/kg)")
    P: float = Field(..., ge=0, le=200, description="Phosphorus (mg/kg)")
    K: float = Field(..., ge=0, le=300, description="Potassium (mg/kg)")
    temperature: float = Field(..., ge=0, le=50, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=3.5, le=10, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall (mm)")
    
    @field_validator('ph')
    @classmethod
    def validate_ph(cls, v):
        if not 3.5 <= v <= 10:
            raise ValueError('pH must be between 3.5 and 10')
        return v


class FertilizerRequest(BaseModel):
    """Extended request with satellite data for fertilizer recommendation"""
    N: float = Field(..., ge=0, le=200)
    P: float = Field(..., ge=0, le=200)
    K: float = Field(..., ge=0, le=300)
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    NDVI: float = Field(..., ge=-1, le=1, description="Satellite NDVI index")
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture (%)")
    growth_stage: str = Field(..., description="seedling|vegetative|flowering|fruiting|maturity")
    forecast_rain: float = Field(default=0, description="Forecasted rainfall in next 48h (mm)")


class CropPredictionResponse(BaseModel):
    """Response for crop recommendation"""
    recommended_crop: str
    confidence: float
    alternative_crops: List[Dict[str, float]]
    feature_contributions: Dict[str, float]


class FertilizerRecommendation(BaseModel):
    """Detailed fertilizer recommendation with alerts"""
    npk_plan: Dict[str, float]  # {N: kg/ha, P: kg/ha, K: kg/ha}
    application_timing: str
    alerts: List[str]
    green_score: float  # Environmental impact (0-100, higher is better)
    estimated_roi: Dict[str, float]  # {cost_inr: float, yield_increase_kg: float, revenue_inr: float}
    satellite_insights: Dict[str, Any]


class NDVIResponse(BaseModel):
    """NDVI satellite data response"""
    ndvi: float
    interpretation: str
    timestamp: str
    source: str


# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelRegistry:
    """Singleton for lazy-loading ML models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models_loaded = False
        return cls._instance
    
    def load_models(self):
        """Load all trained models into memory"""
        if self.models_loaded:
            return
        
        model_dir = Path('models')
        
        try:
            self.crop_model = joblib.load(model_dir / 'crop_model.pkl')
            self.label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
            self.yield_model = joblib.load(model_dir / 'yield_model.pkl')
            self.yield_scaler = joblib.load(model_dir / 'yield_scaler.pkl')
            
            with open(model_dir / 'crop_labels.json', 'r') as f:
                self.crop_labels = json.load(f)
            
            self.models_loaded = True
            print("✅ Models loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

models = ModelRegistry()


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    models.load_models()


# ============================================================================
# FERTILIZER RECOMMENDATION ENGINE
# ============================================================================

class FertilizerRecommender:
    """
    Advanced fertilizer recommendation engine with satellite data fusion.
    
    Logic Flow:
    1. Analyze ground truth (N, P, K, pH)
    2. Integrate satellite data (NDVI for crop stress detection)
    3. Apply growth stage adjustments
    4. Check weather constraints (rain lock)
    5. Calculate economic and environmental metrics
    
    Mathematical Foundation:
    - Deficiency Detection: If soil_N < threshold[growth_stage], flag for supplementation
    - NDVI Fusion: NDVI < 0.3 indicates vegetation stress → Cross-check with soil N
    - Rain Lock: forecast_rain > 10mm → High runoff risk → Postpone application
    """
    
    # Growth stage-specific nutrient requirements (kg/ha)
    STAGE_REQUIREMENTS = {
        'seedling': {'N': 30, 'P': 20, 'K': 20},
        'vegetative': {'N': 80, 'P': 40, 'K': 60},
        'flowering': {'N': 50, 'P': 60, 'K': 80},
        'fruiting': {'N': 40, 'P': 50, 'K': 100},
        'maturity': {'N': 20, 'P': 30, 'K': 50}
    }
    
    # Soil deficiency thresholds (mg/kg)
    DEFICIENCY_THRESHOLDS = {
        'N': 50,
        'P': 15,
        'K': 100
    }
    
    @staticmethod
    def calculate_npk_plan(request: FertilizerRequest) -> Dict[str, float]:
        """
        Calculate optimal NPK application rates.
        
        Algorithm:
        1. Determine base requirement from growth stage
        2. Adjust for current soil levels (deficit calculation)
        3. Apply NDVI correction factor (stressed crops need more N)
        4. Apply pH correction (nutrient availability factor)
        
        Returns:
            Dict with N, P, K in kg/hectare
        """
        stage_req = FertilizerRecommender.STAGE_REQUIREMENTS[request.growth_stage]
        
        # Deficit calculation (kg/ha)
        # Conversion: mg/kg soil → kg/ha (assume 2M kg soil per hectare, 20cm depth)
        soil_to_ha = 2000  # Simplified conversion factor
        
        n_deficit = max(0, stage_req['N'] - (request.N * soil_to_ha / 1000))
        p_deficit = max(0, stage_req['P'] - (request.P * soil_to_ha / 1000))
        k_deficit = max(0, stage_req['K'] - (request.K * soil_to_ha / 1000))
        
        # NDVI-based stress correction
        # If NDVI < 0.4 (moderate stress), increase N by 20%
        ndvi_factor = 1.0
        if request.NDVI < 0.4:
            ndvi_factor = 1.2
        elif request.NDVI < 0.3:
            ndvi_factor = 1.4  # Severe stress
        
        # pH correction (nutrient availability)
        # Optimal pH: 6.0-7.0. Outside this range, nutrients become less available
        ph_factor = 1.0
        if request.ph < 5.5 or request.ph > 7.5:
            ph_factor = 1.15  # Compensate for reduced availability
        
        # Final recommendations
        npk_plan = {
            'N': round(n_deficit * ndvi_factor * ph_factor, 2),
            'P': round(p_deficit * ph_factor, 2),
            'K': round(k_deficit, 2)
        }
        
        return npk_plan
    
    @staticmethod
    def generate_alerts(request: FertilizerRequest, npk_plan: Dict[str, float]) -> List[str]:
        """Generate actionable alerts based on conditions"""
        alerts = []
        
        # Rain Lock Alert
        if request.forecast_rain > 10:
            alerts.append(
                f"⚠️ RAIN LOCK: {request.forecast_rain}mm forecasted. "
                "Do not fertilize - high runoff risk. Postpone application."
            )
        
        # NDVI-based alerts
        if request.NDVI < 0.3 and request.N < FertilizerRecommender.DEFICIENCY_THRESHOLDS['N']:
            alerts.append(
                "🔴 HIGH PRIORITY: Nitrogen deficiency detected via satellite (NDVI < 0.3) "
                f"and ground sensors (N = {request.N} mg/kg). Immediate intervention required."
            )
        elif request.NDVI < 0.5:
            alerts.append(
                f"🟡 MODERATE STRESS: NDVI = {request.NDVI:.2f}. Monitor crop health closely."
            )
        
        # pH alerts
        if request.ph < 5.5:
            alerts.append(
                f"⚠️ Acidic soil (pH = {request.ph}). Consider lime application to improve nutrient availability."
            )
        elif request.ph > 7.5:
            alerts.append(
                f"⚠️ Alkaline soil (pH = {request.ph}). May need sulfur amendments."
            )
        
        # Excessive application warning
        if npk_plan['N'] > 150:
            alerts.append(
                "⚠️ High nitrogen recommendation. Split application into 2-3 doses to prevent leaching."
            )
        
        return alerts
    
    @staticmethod
    def calculate_green_score(npk_plan: Dict[str, float]) -> float:
        """
        Calculate environmental impact score (0-100, higher is better).
        
        Logic:
        - Lower application = Higher score (reduces runoff, eutrophication)
        - Penalize excessive N (greenhouse gas emissions from N2O)
        
        Formula: 100 - (total_NPK / 3.5)  [Empirical calibration]
        """
        total_npk = sum(npk_plan.values())
        
        # Penalize N more heavily (N2O emissions)
        weighted_total = npk_plan['N'] * 1.5 + npk_plan['P'] + npk_plan['K']
        
        green_score = max(0, 100 - (weighted_total / 3.5))
        return round(green_score, 2)
    
    @staticmethod
    def estimate_roi(npk_plan: Dict[str, float], predicted_yield: float) -> Dict[str, float]:
        """
        Estimate economic returns (placeholder for data.gov.in integration).
        
        Assumptions (Indian market 2024):
        - Urea (46% N): ₹300/50kg bag → ₹6.5/kg N
        - DAP (18% N, 46% P): ₹1350/50kg bag → ₹15/kg P
        - MOP (60% K): ₹850/50kg bag → ₹1.4/kg K
        - Crop revenue: ₹20/kg (average for wheat/rice)
        """
        # Fertilizer costs (INR/kg nutrient)
        cost_per_kg = {'N': 6.5, 'P': 15, 'K': 1.4}
        
        total_cost = sum(npk_plan[nutrient] * cost_per_kg[nutrient] 
                        for nutrient in ['N', 'P', 'K'])
        
        # Yield increase estimation (simplified)
        # Assume 10% yield increase per 50kg/ha NPK applied
        total_npk = sum(npk_plan.values())
        yield_increase_pct = min(0.3, (total_npk / 50) * 0.1)  # Cap at 30%
        yield_increase_kg = predicted_yield * yield_increase_pct
        
        revenue_increase = yield_increase_kg * 20  # ₹20/kg
        
        return {
            'cost_inr': round(total_cost, 2),
            'yield_increase_kg': round(yield_increase_kg, 2),
            'revenue_inr': round(revenue_increase, 2),
            'net_profit_inr': round(revenue_increase - total_cost, 2)
        }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Welcome page with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AgriMinds API</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-align: center;
            }
            .tagline {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .status {
                background: #10b981;
                color: white;
                padding: 10px 20px;
                border-radius: 50px;
                display: inline-block;
                margin-bottom: 30px;
                font-weight: bold;
            }
            .endpoints {
                margin-top: 30px;
            }
            .endpoint {
                background: #f8fafc;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 5px;
            }
            .endpoint-method {
                color: #667eea;
                font-weight: bold;
                margin-right: 10px;
            }
            .endpoint-path {
                font-family: 'Courier New', monospace;
                color: #333;
            }
            .endpoint-desc {
                color: #666;
                margin-top: 5px;
                font-size: 0.9em;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .feature {
                text-align: center;
                padding: 20px;
                background: #f8fafc;
                border-radius: 10px;
            }
            .feature-icon {
                font-size: 2em;
                margin-bottom: 10px;
            }
            .feature-title {
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            .feature-desc {
                color: #666;
                font-size: 0.9em;
            }
            .docs-link {
                text-align: center;
                margin-top: 30px;
            }
            .docs-link a {
                background: #667eea;
                color: white;
                padding: 12px 30px;
                border-radius: 50px;
                text-decoration: none;
                display: inline-block;
                transition: background 0.3s;
            }
            .docs-link a:hover {
                background: #5568d3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌾 AgriMinds API</h1>
            <p class="tagline">Satellite-Enhanced Precision Agriculture Platform</p>
            <div style="text-align: center;">
                <span class="status">✅ ONLINE</span>
            </div>
            
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">🛰️</div>
                    <div class="feature-title">Satellite Integration</div>
                    <div class="feature-desc">Real-time NDVI analysis</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">🌱</div>
                    <div class="feature-title">Crop Prediction</div>
                    <div class="feature-desc">ML-powered recommendations</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">⚗️</div>
                    <div class="feature-title">Smart Fertilizer</div>
                    <div class="feature-desc">NPK optimization engine</div>
                </div>
            </div>

            <div class="endpoints">
                <h2 style="color: #333; margin-bottom: 20px;">API Endpoints</h2>
                
                <div class="endpoint">
                    <div>
                        <span class="endpoint-method">POST</span>
                        <span class="endpoint-path">/predict/crop</span>
                    </div>
                    <div class="endpoint-desc">Get optimal crop recommendations based on soil parameters</div>
                </div>

                <div class="endpoint">
                    <div>
                        <span class="endpoint-method">POST</span>
                        <span class="endpoint-path">/predict/fertilizer</span>
                    </div>
                    <div class="endpoint-desc">Generate NPK fertilizer plan with satellite data fusion</div>
                </div>

                <div class="endpoint">
                    <div>
                        <span class="endpoint-method">GET</span>
                        <span class="endpoint-path">/satellite/ndvi</span>
                    </div>
                    <div class="endpoint-desc">Retrieve NDVI index for given coordinates</div>
                </div>
            </div>

            <div class="docs-link">
                <a href="/docs" target="_blank">📚 View Interactive API Documentation</a>
            </div>

            <div style="text-align: center; margin-top: 30px; color: #999; font-size: 0.9em;">
                <p>Version 1.0.0 | Built with FastAPI</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.post("/predict/crop", response_model=CropPredictionResponse)
async def predict_crop(soil: SoilData):
    """
    Predict optimal crop based on soil parameters.
    
    Process:
    1. Feature engineering (NPK ratio, moisture index)
    2. Random Forest prediction (100 decision trees vote)
    3. Extract confidence from probability distribution
    4. Return top 3 alternatives
    """
    try:
        # Feature engineering (must match training pipeline)
        npk_ratio = soil.N / (soil.P + soil.K + 1e-6)
        moisture_index = soil.humidity * soil.rainfall
        
        features = np.array([[
            soil.N, soil.P, soil.K, soil.temperature, soil.humidity,
            soil.ph, soil.rainfall, npk_ratio, moisture_index
        ]])
        
        # Predict
        prediction = models.crop_model.predict(features)[0]
        probabilities = models.crop_model.predict_proba(features)[0]
        
        # Get crop name
        crop_name = models.label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Top 3 alternatives
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        alternatives = [
            {
                'crop': models.label_encoder.inverse_transform([idx])[0],
                'confidence': float(probabilities[idx])
            }
            for idx in top_3_idx if idx != prediction
        ]
        
        # Feature contributions (simplified)
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        feature_values = [soil.N, soil.P, soil.K, soil.temperature, 
                         soil.humidity, soil.ph, soil.rainfall]
        
        contributions = dict(zip(feature_names, feature_values))
        
        return CropPredictionResponse(
            recommended_crop=crop_name,
            confidence=confidence,
            alternative_crops=alternatives,
            feature_contributions=contributions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/fertilizer", response_model=FertilizerRecommendation)
async def predict_fertilizer(request: FertilizerRequest):
    """
    Generate fertilizer recommendation with satellite data fusion.
    
    The "Brain" of AgriMinds:
    1. Satellite Analysis: NDVI → Crop stress detection
    2. Ground Truth: N-P-K levels → Nutrient deficiency
    3. Cross-Validation: If NDVI < 0.3 AND N < 50 → HIGH PRIORITY alert
    4. Growth Stage Adjustment: Tailor recommendations to crop phenology
    5. Weather Integration: Rain lock mechanism
    6. Economic Analysis: ROI calculation
    """
    try:
        # Calculate NPK plan
        npk_plan = FertilizerRecommender.calculate_npk_plan(request)
        
        # Predict yield (for ROI calculation)
        yield_features = np.array([[
            request.soil_moisture,
            request.NDVI,
            request.temperature,
            request.rainfall,
            request.NDVI * request.soil_moisture,  # VHI
            1 if request.temperature > 35 else 0   # thermal_stress
        ]])
        
        yield_features_scaled = models.yield_scaler.transform(yield_features)
        predicted_yield = models.yield_model.predict(yield_features_scaled)[0]
        
        # Generate alerts
        alerts = FertilizerRecommender.generate_alerts(request, npk_plan)
        
        # Calculate green score
        green_score = FertilizerRecommender.calculate_green_score(npk_plan)
        
        # Estimate ROI
        roi = FertilizerRecommender.estimate_roi(npk_plan, predicted_yield)
        
        # Application timing
        if request.forecast_rain > 10:
            timing = f"POSTPONE: Wait until rain passes (forecasted {request.forecast_rain}mm)"
        else:
            timing = "Apply within 48 hours during morning or evening (avoid midday heat)"
        
        # Satellite insights
        satellite_insights = {
            'ndvi': request.NDVI,
            'stress_level': 'severe' if request.NDVI < 0.3 else 'moderate' if request.NDVI < 0.5 else 'healthy',
            'predicted_yield_kg_per_ha': round(predicted_yield, 2),
            'source': 'Sentinel-2 (simulated)'
        }
        
        return FertilizerRecommendation(
            npk_plan=npk_plan,
            application_timing=timing,
            alerts=alerts,
            green_score=green_score,
            estimated_roi=roi,
            satellite_insights=satellite_insights
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/satellite/ndvi", response_model=NDVIResponse)
async def get_ndvi(lat: float, lon: float):
    """
    Retrieve NDVI value for given coordinates (Sentinel-2 simulation).
    
    In Production:
    - Integrate with Sentinel Hub API or OpenWeather Agromonitoring
    - Fetch actual satellite imagery
    - Calculate NDVI from Red and NIR bands: NDVI = (NIR - Red) / (NIR + Red)
    
    Mock Implementation:
    - Generate realistic NDVI based on lat/lon patterns
    - Simulate seasonal variations
    """
    # Simulate async satellite API call
    await asyncio.sleep(0.5)
    
    # Mock NDVI generation (varies by location and season)
    # Use lat/lon hash to generate consistent but varied values
    base_ndvi = 0.7
    variation = np.sin(lat * lon * 0.01) * 0.2
    seasonal_factor = np.cos((datetime.now().month / 12) * 2 * np.pi) * 0.1
    
    ndvi = np.clip(base_ndvi + variation + seasonal_factor, -1, 1)
    
    # Interpretation
    if ndvi > 0.6:
        interpretation = "Healthy vegetation"
    elif ndvi > 0.4:
        interpretation = "Moderate vegetation health"
    elif ndvi > 0.2:
        interpretation = "Stressed vegetation - action required"
    else:
        interpretation = "Severe stress or bare soil"
    
    return NDVIResponse(
        ndvi=round(ndvi, 3),
        interpretation=interpretation,
        timestamp=datetime.utcnow().isoformat(),
        source="Sentinel-2 (simulated)"
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")