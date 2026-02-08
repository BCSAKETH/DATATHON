from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any
import joblib
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, using environment variables directly")
SENTINEL_AVAILABLE = False
try:
    from sentinelhub import (
        SHConfig,
        SentinelHubRequest,
        DataCollection,
        MimeType,
        BBox,
        CRS,
        bbox_to_dimensions,
    )
    SENTINEL_AVAILABLE = True
    print("✅ Sentinel Hub library available")
except ImportError:
    print("⚠️  sentinelhub not installed - using mock NDVI")
    print("   Install with: pip install sentinelhub --break-system-packages")
class SentinelNDVICalculator:   
    def __init__(self, client_id: str, client_secret: str):
        if not SENTINEL_AVAILABLE:
            raise ImportError("sentinelhub library not installed")
        
        self.config = SHConfig()
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            raise ValueError("Sentinel Hub credentials not provided!")
    
    def get_ndvi(
        self,
        latitude: float,
        longitude: float,
        buffer_km: float = 0.5,
        days_back: int = 30,
        max_cloud_coverage: float = 20.0
    ) -> Dict[str, Any]:
        try:
            # Define bounding box
            buffer_deg = buffer_km / 111.0
            bbox = BBox(
                bbox=[
                    longitude - buffer_deg,
                    latitude - buffer_deg,
                    longitude + buffer_deg,
                    latitude + buffer_deg
                ],
                crs=CRS.WGS84
            )
            
            # Define time range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Evalscript to calculate NDVI
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B04", "B08", "SCL"],
                        units: "DN"
                    }],
                    output: {
                        bands: 2,
                        sampleType: "FLOAT32"
                    }
                };
            }

            function evaluatePixel(sample) {
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.00001);
                let is_cloud = (sample.SCL === 3 || sample.SCL === 8 || sample.SCL === 9 || sample.SCL === 10);
                return [ndvi, is_cloud ? 1 : 0];
            }
            """
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        maxcc=max_cloud_coverage / 100.0,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=bbox_to_dimensions(bbox, resolution=10),
                config=self.config
            )
            
            print(f"📡 Fetching Sentinel-2 data for ({latitude}, {longitude})...")
            data = request.get_data()
            
            if not data or len(data) == 0:
                raise Exception("No satellite data available for this location and time range")
            ndvi_array = data[0][:, :, 0]
            cloud_mask = data[0][:, :, 1]
            
            valid_pixels = cloud_mask == 0
            
            if not valid_pixels.any():
                raise Exception("All pixels are cloudy - try increasing max_cloud_coverage or days_back")
            
            # Calculate mean NDVI
            ndvi_value = float(np.mean(ndvi_array[valid_pixels]))
            ndvi_std = float(np.std(ndvi_array[valid_pixels]))
            cloud_coverage_pct = float((1 - valid_pixels.sum() / valid_pixels.size) * 100)
            
            # Interpret NDVI
            if ndvi_value > 0.7:
                interpretation = "Dense, healthy vegetation (excellent crop health)"
            elif ndvi_value > 0.6:
                interpretation = "Healthy vegetation (good crop health)"
            elif ndvi_value > 0.4:
                interpretation = "Moderate vegetation health (monitor closely)"
            elif ndvi_value > 0.2:
                interpretation = "Stressed vegetation (action required)"
            elif ndvi_value > 0:
                interpretation = "Sparse vegetation or bare soil (critical)"
            else:
                interpretation = "Water, snow, or bare ground (no vegetation)"
            
            result = {
                'ndvi': round(ndvi_value, 3),
                'interpretation': interpretation,
                'timestamp': datetime.now().isoformat(),
                'source': 'Sentinel-2 L2A',
                'metadata': {
                    'date_range': f"{start_date.date()} to {end_date.date()}",
                    'cloud_coverage_pct': round(cloud_coverage_pct, 1),
                    'ndvi_std': round(ndvi_std, 3),
                    'valid_pixels': int(valid_pixels.sum()),
                    'total_pixels': int(valid_pixels.size),
                    'resolution_m': 10,
                    'area_km2': round((buffer_km * 2) ** 2, 2)
                }
            }
            
            print(f"✅ NDVI calculated: {result['ndvi']} ({interpretation})")
            return result
            
        except Exception as e:
            print(f"❌ Sentinel Hub error: {str(e)}")
            raise Exception(f"Failed to fetch NDVI from Sentinel Hub: {str(e)}")


class MockNDVICalculator:

    
    def get_ndvi(
        self,
        latitude: float,
        longitude: float,
        buffer_km: float = 0.5,
        days_back: int = 30,
        max_cloud_coverage: float = 20.0
    ) -> Dict[str, Any]:
        base_ndvi = 0.7
        variation = float(np.sin(latitude * longitude * 0.01) * 0.2)
        seasonal_factor = float(np.cos((datetime.now().month / 12) * 2 * np.pi) * 0.1)
        ndvi = float(np.clip(base_ndvi + variation + seasonal_factor, -1, 1))
        
        if ndvi > 0.6:
            interpretation = "Healthy vegetation"
        elif ndvi > 0.4:
            interpretation = "Moderate vegetation health"
        else:
            interpretation = "Stressed vegetation"
        
        return {
            'ndvi': round(ndvi, 3),
            'interpretation': interpretation,
            'timestamp': datetime.now().isoformat(),
            'source': 'Mock NDVI (for testing)',
            'metadata': {}
        }



class FertilizerRequest(BaseModel):
    crop_type: str = Field(..., description="Crop name (e.g., rice, wheat, maize)")
    N: float = Field(..., ge=0, le=300, description="Nitrogen content in soil (kg/ha)")
    P: float = Field(..., ge=0, le=200, description="Phosphorus content in soil (kg/ha)")
    K: float = Field(..., ge=0, le=400, description="Potassium content in soil (kg/ha)")
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture (%)")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall in mm")
    NDVI: float = Field(..., ge=-1, le=1, description="Normalized Difference Vegetation Index")
    forecast_rain: float = Field(0, ge=0, le=100, description="Forecasted rainfall in mm")


class IrrigationRequirement(BaseModel):
    water_needed_mm: float = Field(..., description="Water requirement in mm")
    irrigation_frequency_days: int = Field(..., description="Recommended irrigation frequency in days")
    irrigation_method: str = Field(..., description="Recommended irrigation method")
    water_stress_level: str = Field(..., description="Current water stress level")
    recommendations: List[str] = Field(..., description="Specific irrigation recommendations")


class FertilizerRecommendation(BaseModel):
    crop_type: str
    npk_plan: Dict[str, Any]
    application_timing: str
    alerts: List[str]
    green_score: float
    estimated_roi: float
    satellite_insights: Dict[str, Any]
    irrigation_requirements: IrrigationRequirement


class NDVIResponse(BaseModel):
    ndvi: float
    interpretation: str
    timestamp: str
    source: str



CROP_REQUIREMENTS = {
    'rice': {'N': 120, 'P': 60, 'K': 40, 'water_mm_per_day': 6.5, 'irrigation_method': 'flood'},
    'wheat': {'N': 100, 'P': 50, 'K': 40, 'water_mm_per_day': 4.5, 'irrigation_method': 'sprinkler'},
    'maize': {'N': 140, 'P': 60, 'K': 60, 'water_mm_per_day': 5.0, 'irrigation_method': 'drip'},
    'cotton': {'N': 120, 'P': 60, 'K': 50, 'water_mm_per_day': 5.5, 'irrigation_method': 'drip'},
    'sugarcane': {'N': 200, 'P': 80, 'K': 100, 'water_mm_per_day': 7.0, 'irrigation_method': 'furrow'},
    'chickpea': {'N': 25, 'P': 60, 'K': 40, 'water_mm_per_day': 3.0, 'irrigation_method': 'sprinkler'},
    'kidney_beans': {'N': 30, 'P': 50, 'K': 50, 'water_mm_per_day': 3.5, 'irrigation_method': 'drip'},
    'pigeon_peas': {'N': 25, 'P': 50, 'K': 50, 'water_mm_per_day': 3.0, 'irrigation_method': 'sprinkler'},
    'moth_beans': {'N': 20, 'P': 40, 'K': 40, 'water_mm_per_day': 2.5, 'irrigation_method': 'drip'},
    'mung_bean': {'N': 25, 'P': 50, 'K': 40, 'water_mm_per_day': 3.0, 'irrigation_method': 'sprinkler'},
    'black_gram': {'N': 25, 'P': 50, 'K': 40, 'water_mm_per_day': 3.2, 'irrigation_method': 'sprinkler'},
    'lentil': {'N': 20, 'P': 50, 'K': 40, 'water_mm_per_day': 2.8, 'irrigation_method': 'sprinkler'},
    'pomegranate': {'N': 100, 'P': 50, 'K': 100, 'water_mm_per_day': 4.0, 'irrigation_method': 'drip'},
    'banana': {'N': 200, 'P': 75, 'K': 300, 'water_mm_per_day': 6.0, 'irrigation_method': 'drip'},
    'mango': {'N': 100, 'P': 50, 'K': 100, 'water_mm_per_day': 4.5, 'irrigation_method': 'drip'},
    'grapes': {'N': 80, 'P': 60, 'K': 120, 'water_mm_per_day': 3.5, 'irrigation_method': 'drip'},
    'watermelon': {'N': 100, 'P': 50, 'K': 80, 'water_mm_per_day': 5.0, 'irrigation_method': 'drip'},
    'muskmelon': {'N': 90, 'P': 50, 'K': 70, 'water_mm_per_day': 4.5, 'irrigation_method': 'drip'},
    'apple': {'N': 80, 'P': 50, 'K': 80, 'water_mm_per_day': 3.5, 'irrigation_method': 'drip'},
    'orange': {'N': 100, 'P': 50, 'K': 90, 'water_mm_per_day': 4.0, 'irrigation_method': 'drip'},
    'papaya': {'N': 150, 'P': 60, 'K': 150, 'water_mm_per_day': 5.5, 'irrigation_method': 'drip'},
    'coconut': {'N': 120, 'P': 50, 'K': 120, 'water_mm_per_day': 5.0, 'irrigation_method': 'basin'},
    'jute': {'N': 80, 'P': 40, 'K': 40, 'water_mm_per_day': 4.5, 'irrigation_method': 'flood'},
    'coffee': {'N': 100, 'P': 50, 'K': 100, 'water_mm_per_day': 4.0, 'irrigation_method': 'drip'},
}




class FertilizerRecommender:
    
    @staticmethod
    def calculate_npk_plan(request: FertilizerRequest) -> Dict[str, Any]:
        """Calculate NPK requirements based on crop type and soil conditions."""
        crop_type = request.crop_type.lower()
        
        if crop_type not in CROP_REQUIREMENTS:
            raise ValueError(f"Crop '{crop_type}' not supported. Available crops: {', '.join(CROP_REQUIREMENTS.keys())}")
        
        crop_req = CROP_REQUIREMENTS[crop_type]
        
        n_deficit = max(0, crop_req['N'] - request.N)
        p_deficit = max(0, crop_req['P'] - request.P)
        k_deficit = max(0, crop_req['K'] - request.K)
        
        ndvi_factor = 1.0 + (0.7 - request.NDVI) * 0.5 if request.NDVI < 0.7 else 1.0
        
        return {
            'nitrogen_kg_per_ha': round(n_deficit * ndvi_factor, 2),
            'phosphorus_kg_per_ha': round(p_deficit * ndvi_factor, 2),
            'potassium_kg_per_ha': round(k_deficit * ndvi_factor, 2),
            'total_fertilizer_kg': round((n_deficit + p_deficit + k_deficit) * ndvi_factor, 2),
            'npk_ratio': f"{crop_req['N']}:{crop_req['P']}:{crop_req['K']}",
            'crop_optimal_npk': crop_req
        }
    
    @staticmethod
    def calculate_irrigation(request: FertilizerRequest) -> IrrigationRequirement:
        """Calculate irrigation requirements based on crop type and conditions."""
        crop_type = request.crop_type.lower()
        
        if crop_type not in CROP_REQUIREMENTS:
            raise ValueError(f"Crop '{crop_type}' not supported")
        
        crop_req = CROP_REQUIREMENTS[crop_type]
        
        daily_et = crop_req['water_mm_per_day']
        
        temp_factor = 1.0
        if request.temperature > 30:
            temp_factor = 1.2
        elif request.temperature < 20:
            temp_factor = 0.8
        
        humidity_factor = 1.0
        if request.humidity < 50:
            humidity_factor = 1.15
        elif request.humidity > 80:
            humidity_factor = 0.85
        
        # Effective rainfall
        effective_rain = request.rainfall * 0.8  # 80% efficiency
        
        # Water needed
        water_needed = (daily_et * temp_factor * humidity_factor) - (effective_rain / 7)
        water_needed = max(0, water_needed)
        
        # Determine stress level based on soil moisture and NDVI
        if request.soil_moisture < 30 or request.NDVI < 0.4:
            stress_level = "High - Immediate irrigation needed"
            frequency = 2
        elif request.soil_moisture < 50 or request.NDVI < 0.6:
            stress_level = "Moderate - Schedule irrigation soon"
            frequency = 4
        else:
            stress_level = "Low - Normal schedule"
            frequency = 7
        
        # Generate recommendations
        recommendations = []
        
        if water_needed > 0:
            recommendations.append(f"Apply {round(water_needed * frequency, 1)}mm of water every {frequency} days")
        
        if request.NDVI < 0.5:
            recommendations.append("Low NDVI indicates water stress - increase irrigation frequency")
        
        if request.temperature > 32:
            recommendations.append("High temperature detected - consider morning irrigation to reduce evaporation")
        
        if request.soil_moisture < 40:
            recommendations.append("Soil moisture is low - immediate irrigation recommended")
        
        if request.forecast_rain > 10:
            recommendations.append(f"Rain forecasted ({request.forecast_rain}mm) - delay irrigation")
        
        if not recommendations:
            recommendations.append("Current water levels are adequate - maintain regular schedule")
        
        return IrrigationRequirement(
            water_needed_mm=round(water_needed, 2),
            irrigation_frequency_days=frequency,
            irrigation_method=crop_req['irrigation_method'],
            water_stress_level=stress_level,
            recommendations=recommendations
        )
    
    @staticmethod
    def generate_alerts(request: FertilizerRequest, npk_plan: Dict) -> List[str]:
        """Generate alerts based on conditions."""
        alerts = []
        
        if request.NDVI < 0.4:
            alerts.append("⚠️ CRITICAL: Very low NDVI indicates severe crop stress")
        
        if request.soil_moisture < 30:
            alerts.append("⚠️ WARNING: Low soil moisture - irrigation needed urgently")
        
        if request.temperature > 35:
            alerts.append("⚠️ CAUTION: High temperature may cause heat stress")
        
        if npk_plan['nitrogen_kg_per_ha'] > 150:
            alerts.append("⚠️ High nitrogen requirement - split applications recommended")
        
        if request.forecast_rain > 20:
            alerts.append("⚠️ Heavy rain forecasted - postpone fertilizer application")
        
        return alerts
    
    @staticmethod
    def calculate_green_score(npk_plan: Dict) -> float:
        """Calculate sustainability score."""
        total = npk_plan['total_fertilizer_kg']
        
        if total < 100:
            return 9.5
        elif total < 200:
            return 8.0
        elif total < 300:
            return 6.5
        else:
            return 5.0
    
    @staticmethod
    def estimate_roi(npk_plan: Dict, predicted_yield: float) -> float:
        """Estimate return on investment."""
        fertilizer_cost = npk_plan['total_fertilizer_kg'] * 2.5  # $2.5 per kg
        expected_revenue = predicted_yield * 0.5  # $0.5 per kg yield
        
        if fertilizer_cost == 0:
            return 0.0
        
        return round(expected_revenue / fertilizer_cost, 2)



app = FastAPI(
    title="AgriMinds API - Modified",
    description="Fertilizer & Irrigation Prediction with NDVI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentinel_client_id = os.getenv("SENTINEL_CLIENT_ID", "")
sentinel_client_secret = os.getenv("SENTINEL_CLIENT_SECRET", "")

try:
    if SENTINEL_AVAILABLE and sentinel_client_id and sentinel_client_secret:
        ndvi_calculator = SentinelNDVICalculator(sentinel_client_id, sentinel_client_secret)
        print("✅ Using real Sentinel Hub NDVI")
    else:
        ndvi_calculator = MockNDVICalculator()
        print("⚠️ Using mock NDVI (set SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET for real data)")
except Exception as e:
    print(f"⚠️ Failed to initialize Sentinel Hub: {e}")
    ndvi_calculator = MockNDVICalculator()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>AgriMinds API</title></head>
        <body style="font-family: Arial; padding: 40px; background: #f5f5f5;">
            <h1>🌾 AgriMinds API - Modified Version</h1>
            <p>Fertilizer & Irrigation Prediction with NDVI Support</p>
            <ul>
                <li><a href="/docs">Interactive API Documentation</a></li>
                <li><strong>POST /predict/fertilizer</strong> - Get fertilizer & irrigation recommendations</li>
                <li><strong>GET /satellite/ndvi</strong> - Get NDVI from satellite</li>
                <li><strong>GET /crops</strong> - List supported crops</li>
            </ul>
            <h3>Supported Crops:</h3>
            <p>rice, wheat, maize, cotton, sugarcane, chickpea, kidney_beans, pigeon_peas, 
               moth_beans, mung_bean, black_gram, lentil, pomegranate, banana, mango, grapes, 
               watermelon, muskmelon, apple, orange, papaya, coconut, jute, coffee</p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ndvi_source": "Sentinel-2" if isinstance(ndvi_calculator, SentinelNDVICalculator) else "Mock",
        "supported_crops": len(CROP_REQUIREMENTS)
    }


@app.get("/crops")
async def get_supported_crops():
    """Get list of supported crops with their requirements."""
    return {
        "supported_crops": list(CROP_REQUIREMENTS.keys()),
        "total_count": len(CROP_REQUIREMENTS),
        "crop_details": CROP_REQUIREMENTS
    }


@app.post("/predict/fertilizer", response_model=FertilizerRecommendation)
async def predict_fertilizer(request: FertilizerRequest):
    """Generate fertilizer and irrigation recommendations for specified crop."""
    try:
        # Calculate fertilizer plan
        npk_plan = FertilizerRecommender.calculate_npk_plan(request)
        
        # Calculate irrigation requirements
        irrigation_req = FertilizerRecommender.calculate_irrigation(request)
        
        # Yield estimation
        def calculate_yield_estimate(ndvi, soil_moisture, temperature, rainfall):
            """Empirical yield estimation using agronomic relationships."""
            base_yield = 5000  # kg/ha baseline
            
            # NDVI factor
            if ndvi < 0.3:
                ndvi_multiplier = 0.5
            elif ndvi > 0.8:
                ndvi_multiplier = 1.4
            else:
                ndvi_multiplier = 0.5 + ((ndvi - 0.3) / 0.5) * 0.9
            
            # Soil moisture
            moisture_deviation = abs(soil_moisture - 70)
            moisture_multiplier = max(0.6, 1.0 - (moisture_deviation / 100))
            
            # Temperature
            if 22 <= temperature <= 30:
                temp_multiplier = 1.0
            elif temperature > 38:
                temp_multiplier = 0.4
            elif temperature < 15:
                temp_multiplier = 0.5
            else:
                temp_multiplier = 0.8
            
            # Rainfall
            if 50 <= rainfall <= 150:
                rain_multiplier = 1.0
            elif rainfall > 300:
                rain_multiplier = 0.7
            elif rainfall < 20:
                rain_multiplier = 0.5
            else:
                rain_multiplier = 0.85
            
            estimated_yield = (base_yield * ndvi_multiplier * 
                             moisture_multiplier * temp_multiplier * 
                             rain_multiplier)
            
            return max(2000, min(8000, estimated_yield))
        
        predicted_yield = calculate_yield_estimate(
            ndvi=request.NDVI,
            soil_moisture=request.soil_moisture,
            temperature=request.temperature,
            rainfall=request.rainfall
        )
        
        # Generate alerts
        alerts = FertilizerRecommender.generate_alerts(request, npk_plan)
        
        # Calculate scores
        green_score = FertilizerRecommender.calculate_green_score(npk_plan)
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
            'source': 'Sentinel-2'
        }
        
        return FertilizerRecommendation(
            crop_type=request.crop_type,
            npk_plan=npk_plan,
            application_timing=timing,
            alerts=alerts,
            green_score=green_score,
            estimated_roi=roi,
            satellite_insights=satellite_insights,
            irrigation_requirements=irrigation_req
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print("❌ Prediction Error:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/satellite/ndvi", response_model=NDVIResponse)
async def get_ndvi(
    lat: float = Query(..., description="Latitude coordinate"), 
    lon: float = Query(..., description="Longitude coordinate"),
    buffer_km: float = Query(0.5, description="Buffer radius in kilometers"),
    days_back: int = Query(30, description="Days to search back for imagery"),
    max_cloud: float = Query(20.0, description="Maximum cloud coverage percentage")
):
    """Retrieve NDVI value for given coordinates from Sentinel-2 satellite."""
    try:
        result = ndvi_calculator.get_ndvi(
            latitude=lat,
            longitude=lon,
            buffer_km=buffer_km,
            days_back=days_back,
            max_cloud_coverage=max_cloud
        )
        
        return NDVIResponse(
            ndvi=result['ndvi'],
            interpretation=result['interpretation'],
            timestamp=result['timestamp'],
            source=result['source']
        )
        
    except Exception as e:
        print(f"⚠️ NDVI calculation failed: {str(e)}")
        print("📝 Using fallback mock NDVI")
        
        base_ndvi = 0.7
        variation = float(np.sin(lat * lon * 0.01) * 0.2)
        seasonal_factor = float(np.cos((datetime.now().month / 12) * 2 * np.pi) * 0.1)
        ndvi = float(np.clip(base_ndvi + variation + seasonal_factor, -1, 1))
        
        if ndvi > 0.6:
            interpretation = "Healthy vegetation"
        elif ndvi > 0.4:
            interpretation = "Moderate vegetation health"
        elif ndvi > 0.2:
            interpretation = "Stressed vegetation - action required"
        else:
            interpretation = "Severe stress or bare soil"
        
        return NDVIResponse(
            ndvi=float(round(ndvi, 3)),
            interpretation=interpretation,
            timestamp=datetime.now().isoformat(),
            source="Sentinel-2 (simulated - fallback)"
        )



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation Error",
            "errors": exc.errors(),
            "body": exc.body
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
