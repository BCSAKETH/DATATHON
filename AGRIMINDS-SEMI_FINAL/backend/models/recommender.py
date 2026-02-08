"""
FertilizerRecommender: The "Brain" of AgriMinds
================================================
This module implements the core recommendation engine that fuses:
1. Ground Truth: Soil NPK values from lab tests/IoT sensors
2. Satellite Data: NDVI from Sentinel-2 imagery
3. Temporal Context: Growth stage and weather forecasts

Mathematical Foundation:
-----------------------
The recommendation algorithm uses a multi-factor decision tree:

    IF (NDVI < threshold_stress) AND (soil_nutrient < optimal_range):
        → Cross-validated deficiency (both sources confirm)
        → Apply correction_factor to increase nutrient application
    
    ELIF (NDVI < threshold_stress) AND (soil_nutrient >= optimal_range):
        → Satellite shows stress BUT ground truth shows adequate nutrients
        → Problem is NOT nutrient deficiency (investigate pests/disease)
        → Do NOT waste money on fertilizers

This prevents the "Blind Application" problem where farmers overapply
fertilizers based on visual inspection alone.
"""

import numpy as np
import joblib
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class GrowthStage(Enum):
    """Crop phenological stages - each requires different NPK ratios"""
    SEEDLING = "seedling"       # 0-3 weeks: High P for root development
    VEGETATIVE = "vegetative"   # 3-8 weeks: High N for leaf/stem growth
    FLOWERING = "flowering"     # 8-12 weeks: Balanced NPK
    FRUITING = "fruiting"       # 12+ weeks: High K for fruit quality
    HARVEST = "harvest"         # Minimal fertilization


@dataclass
class SoilData:
    """Ground truth measurements from soil testing"""
    nitrogen: float      # ppm (parts per million)
    phosphorus: float    # ppm
    potassium: float     # ppm
    ph: float           # pH scale (0-14)
    moisture: float     # Percentage (0-100)
    temperature: float  # Celsius


@dataclass
class SatelliteData:
    """Remote sensing metrics from Sentinel-2/Landsat-8"""
    ndvi: float         # Normalized Difference Vegetation Index (-1 to 1)
    evi: float          # Enhanced Vegetation Index (optional)
    lat: float          # Latitude
    lon: float          # Longitude


@dataclass
class WeatherForecast:
    """Meteorological data for rain lock logic"""
    rainfall_forecast: float  # mm in next 48 hours
    temperature: float        # Celsius
    humidity: float          # Percentage


class FertilizerRecommender:
    """
    Core Recommendation Engine
    
    Uses Random Forest models + rule-based expert system for:
    1. Crop suitability prediction
    2. Yield forecasting
    3. Precision fertilizer recommendations
    
    Mathematical Details:
    --------------------
    Random Forest (Ensemble Learning):
    - n_estimators trees vote on final decision
    - Each tree built using Gini Impurity for splits:
      
      Gini(D) = 1 - Σ(p_i)²
      
      where p_i = probability of class i in dataset D
      
    - Split criterion: Choose feature f and threshold t that maximize:
      
      Information Gain = Gini(parent) - Σ(|D_child|/|D_parent|) × Gini(D_child)
    
    For Yield Regression:
    - Uses Mean Squared Error (MSE) for splits:
      
      MSE = (1/n) × Σ(y_true - y_pred)²
    """
    
    def __init__(self, crop_model_path: str, yield_model_path: str):
        """Load pre-trained Random Forest models"""
        self.crop_model = joblib.load(crop_model_path)
        self.yield_model = joblib.load(yield_model_path)
        
        # NDVI thresholds based on agricultural research
        # Source: Tucker, C.J. (1979), Red and photographic infrared linear combinations
        self.NDVI_HEALTHY = 0.6      # Dense, healthy vegetation
        self.NDVI_MODERATE = 0.4     # Moderate vegetation
        self.NDVI_STRESS = 0.3       # Stressed/sparse vegetation
        self.NDVI_BARE_SOIL = 0.1    # Bare soil/dead vegetation
        
        # Optimal soil nutrient ranges (ppm) for major crops
        self.OPTIMAL_RANGES = {
            'nitrogen': (40, 80),     # 40-80 ppm
            'phosphorus': (20, 40),   # 20-40 ppm
            'potassium': (150, 250)   # 150-250 ppm
        }
        
        # Growth stage nutrient coefficients (multipliers)
        self.STAGE_COEFFICIENTS = {
            GrowthStage.SEEDLING: {'N': 0.5, 'P': 1.5, 'K': 0.8},     # High P for roots
            GrowthStage.VEGETATIVE: {'N': 1.8, 'P': 0.7, 'K': 1.0},   # High N for growth
            GrowthStage.FLOWERING: {'N': 1.2, 'P': 1.2, 'K': 1.2},    # Balanced
            GrowthStage.FRUITING: {'N': 0.8, 'P': 1.0, 'K': 1.8},     # High K for quality
            GrowthStage.HARVEST: {'N': 0.0, 'P': 0.0, 'K': 0.0}       # No fertilization
        }
    
    def predict_crop(self, soil: SoilData, weather: Dict) -> Tuple[str, float]:
        """
        Predict most suitable crop for given conditions
        
        Returns:
            (crop_name, confidence_score)
        """
        # Prepare feature vector [N, P, K, temperature, humidity, ph, rainfall]
        features = np.array([[
            soil.nitrogen,
            soil.phosphorus,
            soil.potassium,
            weather['temperature'],
            weather['humidity'],
            soil.ph,
            weather['rainfall']
        ]])
        
        # Get prediction and probability
        crop = self.crop_model.predict(features)[0]
        proba = self.crop_model.predict_proba(features).max()
        
        return crop, proba
    
    def predict_yield(self, soil: SoilData, satellite: SatelliteData, 
                     weather: Dict) -> float:
        """
        Forecast crop yield (kg/hectare) using satellite + ground data
        
        Feature Engineering:
        - Vegetation Health Index (VHI) = (NDVI + 1) / 2  [normalized to 0-1]
        - Moisture Index = sqrt(soil_moisture × rainfall)
        """
        # Calculate derived features
        vhi = (satellite.ndvi + 1) / 2  # Convert NDVI from [-1,1] to [0,1]
        moisture_index = np.sqrt(soil.moisture * weather['rainfall'])
        
        # Feature vector [soil_moisture, NDVI, temperature, rainfall, VHI, moisture_index]
        features = np.array([[
            soil.moisture,
            satellite.ndvi,
            weather['temperature'],
            weather['rainfall'],
            vhi,
            moisture_index
        ]])
        
        yield_prediction = self.yield_model.predict(features)[0]
        return max(0, yield_prediction)  # Ensure non-negative
    
    def recommend_fertilizer(
        self,
        soil: SoilData,
        satellite: SatelliteData,
        weather_forecast: WeatherForecast,
        growth_stage: GrowthStage,
        field_area_hectares: float
    ) -> Dict:
        """
        *** THE CORE ALGORITHM: Satellite Data Fusion ***
        
        Multi-Source Decision Logic:
        1. Check for rain lock (prevents runoff)
        2. Cross-validate NDVI (satellite) with soil NPK (ground truth)
        3. Apply growth stage adjustments
        4. Calculate precise NPK recommendations
        
        Algorithm:
        ----------
        FOR each nutrient in [N, P, K]:
            deficiency = optimal_range[nutrient] - soil_current[nutrient]
            
            IF deficiency > 0:
                # Soil test shows deficiency
                
                IF NDVI < threshold_stress:
                    # BOTH satellite AND ground confirm stress
                    correction_factor = 1.4  (increase by 40%)
                    alert = "HIGH PRIORITY: Cross-validated deficiency"
                ELSE:
                    # Ground shows deficiency but satellite shows healthy crop
                    correction_factor = 1.0  (standard recommendation)
                    alert = "NORMAL: Soil deficiency detected"
            
            ELSE:
                # Soil is adequate
                
                IF NDVI < threshold_stress:
                    # Satellite shows stress BUT nutrients adequate
                    correction_factor = 0.0  (DO NOT fertilize)
                    alert = "INVESTIGATE: Stress is NOT nutrient-related"
                ELSE:
                    correction_factor = 0.0  (no deficiency)
        
        Returns:
            {
                'recommendations': {'N': kg, 'P': kg, 'K': kg},
                'alerts': [list of warnings],
                'green_score': environmental_impact_score,
                'estimated_cost': INR,
                'can_apply': boolean
            }
        """
        recommendations = {'N': 0.0, 'P': 0.0, 'K': 0.0}
        alerts = []
        
        # ============================================================
        # STEP 1: Rain Lock Check (Prevent Runoff)
        # ============================================================
        if weather_forecast.rainfall_forecast > 10:  # mm
            return {
                'recommendations': recommendations,
                'alerts': [f"⛔ RAIN LOCK ACTIVE: {weather_forecast.rainfall_forecast}mm forecast. "
                          "Fertilizer application postponed to prevent runoff and water pollution."],
                'green_score': 100,  # No application = zero environmental impact
                'estimated_cost': 0,
                'can_apply': False
            }
        
        # ============================================================
        # STEP 2: Calculate Base Nutrient Deficiencies
        # ============================================================
        nutrient_map = {
            'N': ('nitrogen', soil.nitrogen),
            'P': ('phosphorus', soil.phosphorus),
            'K': ('potassium', soil.potassium)
        }
        
        for nutrient_symbol, (nutrient_name, current_value) in nutrient_map.items():
            optimal_min, optimal_max = self.OPTIMAL_RANGES[nutrient_name]
            
            # Calculate deficiency (negative means excess)
            deficiency = optimal_min - current_value
            
            if deficiency > 0:
                # Soil test shows deficiency
                base_recommendation = deficiency * field_area_hectares * 0.5  # kg
                
                # ============================================================
                # STEP 3: Cross-Validation with Satellite Data
                # ============================================================
                if satellite.ndvi < self.NDVI_STRESS:
                    # CRITICAL: Both satellite AND ground confirm deficiency
                    correction_factor = 1.4
                    alert_type = "🔴 HIGH PRIORITY"
                    explanation = (f"{nutrient_name.upper()} deficiency CROSS-VALIDATED "
                                 f"(Satellite NDVI={satellite.ndvi:.2f}, "
                                 f"Soil {nutrient_symbol}={current_value} ppm)")
                    alerts.append(f"{alert_type}: {explanation}")
                    
                elif satellite.ndvi < self.NDVI_MODERATE:
                    # Moderate stress - standard correction
                    correction_factor = 1.2
                    alert_type = "🟡 MODERATE"
                    alerts.append(f"{alert_type}: {nutrient_name.upper()} deficiency detected in soil")
                    
                else:
                    # Satellite shows healthy crop despite soil deficiency
                    # Possible reasons: Recent rain, plant adaptation, measurement error
                    correction_factor = 1.0
                    alerts.append(f"ℹ️ NORMAL: {nutrient_name.upper()} slightly below optimal")
                
                # Apply correction
                base_recommendation *= correction_factor
                
            else:
                # Soil nutrient is adequate or excessive
                if satellite.ndvi < self.NDVI_STRESS:
                    # CRITICAL INSIGHT: Satellite shows stress BUT nutrients are adequate
                    # This means the problem is NOT nutrient deficiency!
                    alerts.append(
                        f"⚠️ INVESTIGATE: Crop stress detected (NDVI={satellite.ndvi:.2f}) "
                        f"but {nutrient_symbol} is adequate ({current_value} ppm). "
                        f"Possible causes: Pests, disease, drought, or soil compaction. "
                        f"DO NOT waste money on {nutrient_name} fertilizer!"
                    )
                    base_recommendation = 0.0
                else:
                    # Everything is fine
                    base_recommendation = 0.0
            
            # ============================================================
            # STEP 4: Growth Stage Adjustment
            # ============================================================
            stage_multiplier = self.STAGE_COEFFICIENTS[growth_stage][nutrient_symbol]
            recommendations[nutrient_symbol] = base_recommendation * stage_multiplier
        
        # ============================================================
        # STEP 5: Calculate Economic & Environmental Metrics
        # ============================================================
        green_score = self._calculate_green_score(recommendations, field_area_hectares)
        estimated_cost = self._calculate_cost(recommendations)
        
        return {
            'recommendations': recommendations,
            'alerts': alerts,
            'green_score': green_score,
            'estimated_cost': estimated_cost,
            'can_apply': True,
            'growth_stage': growth_stage.value,
            'ndvi_status': self._get_ndvi_status(satellite.ndvi)
        }
    
    def _get_ndvi_status(self, ndvi: float) -> str:
        """Classify vegetation health from NDVI"""
        if ndvi >= self.NDVI_HEALTHY:
            return "Excellent (Dense vegetation)"
        elif ndvi >= self.NDVI_MODERATE:
            return "Good (Moderate vegetation)"
        elif ndvi >= self.NDVI_STRESS:
            return "Fair (Some stress)"
        elif ndvi >= self.NDVI_BARE_SOIL:
            return "Poor (Stressed vegetation)"
        else:
            return "Critical (Bare soil/dead crops)"
    
    def _calculate_green_score(self, recommendations: Dict, area: float) -> int:
        """
        Environmental Impact Score (0-100, higher = better)
        
        Formula:
        --------
        Green Score = 100 - (total_NPK_kg / (area × 300)) × 100
        
        Rationale: 300 kg/hectare is excessive application threshold
        Score interpretation:
        - 90-100: Minimal environmental impact
        - 70-89: Moderate impact (acceptable)
        - 50-69: High impact (reconsider)
        - <50: Severe impact (pollution risk)
        """
        total_npk = sum(recommendations.values())
        max_acceptable = area * 300  # kg
        
        if total_npk == 0:
            return 100
        
        impact_ratio = min(total_npk / max_acceptable, 1.0)
        green_score = int(100 - (impact_ratio * 100))
        
        return max(0, green_score)
    
    def _calculate_cost(self, recommendations: Dict) -> float:
        """
        Estimate fertilizer cost in INR
        
        Current market prices (India, 2024):
        - Urea (46% N): ₹266/50kg bag = ₹5.32/kg N
        - DAP (18% N, 46% P): ₹1350/50kg bag = ₹27/kg DAP
        - MOP (60% K): ₹1200/50kg bag = ₹24/kg MOP
        
        Conversion:
        - 1 kg N requires 2.17 kg Urea → ₹5.32/kg N × 2.17 = ₹11.55/kg N
        - 1 kg P requires 2.17 kg DAP → ₹27/2.17 = ₹12.44/kg P  
        - 1 kg K requires 1.67 kg MOP → ₹24/1.67 = ₹14.37/kg K
        """
        prices_per_kg = {
            'N': 11.55,  # INR per kg elemental nitrogen
            'P': 12.44,  # INR per kg elemental phosphorus
            'K': 14.37   # INR per kg elemental potassium
        }
        
        total_cost = sum(
            recommendations[nutrient] * prices_per_kg[nutrient]
            for nutrient in ['N', 'P', 'K']
        )
        
        return round(total_cost, 2)


if __name__ == "__main__":
    # Example usage
    print("FertilizerRecommender module loaded successfully!")
    print("This is the 'Brain' of AgriMinds - ready for satellite data fusion! 🧠🌾")
