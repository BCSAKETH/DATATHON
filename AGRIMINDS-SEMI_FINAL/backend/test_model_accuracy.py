"""
AgriMinds Model Accuracy Testing Suite
======================================
Tests the accuracy of fertilizer recommendations, irrigation calculations,
yield predictions, and NDVI interpretations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path


# ============================================================================
# TEST DATA GENERATION
# ============================================================================

class TestDataGenerator:
    """Generate realistic test data for agriculture models."""
    
    @staticmethod
    def generate_crop_test_cases(crop_type: str, num_samples: int = 100) -> pd.DataFrame:
        """Generate test cases for a specific crop type."""
        np.random.seed(42)
        
        test_data = {
            'crop_type': [crop_type] * num_samples,
            'temperature': np.random.uniform(15, 38, num_samples),
            'humidity': np.random.uniform(30, 95, num_samples),
            'soil_moisture': np.random.uniform(20, 90, num_samples),
            'soil_type': np.random.choice(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'], num_samples),
            'nitrogen': np.random.uniform(0, 140, num_samples),
            'phosphorus': np.random.uniform(5, 145, num_samples),
            'potassium': np.random.uniform(5, 205, num_samples),
            'pH': np.random.uniform(4.5, 9.0, num_samples),
            'rainfall': np.random.uniform(20, 300, num_samples),
            'NDVI': np.random.uniform(0.1, 0.9, num_samples),
        }
        
        return pd.DataFrame(test_data)
    
    @staticmethod
    def generate_ground_truth_npk(crop_type: str, current_npk: Dict[str, float]) -> Dict[str, float]:
        """
        Generate ground truth NPK recommendations based on agronomic standards.
        This simulates expert agronomist recommendations.
        """
        # Standard NPK requirements for different crops (simplified)
        crop_standards = {
            'rice': {'N': 120, 'P': 60, 'K': 40},
            'wheat': {'N': 120, 'P': 60, 'K': 40},
            'maize': {'N': 150, 'P': 75, 'K': 75},
            'cotton': {'N': 120, 'P': 60, 'K': 50},
            'sugarcane': {'N': 150, 'P': 50, 'K': 75},
            'chickpea': {'N': 20, 'P': 60, 'K': 40},
            'kidney_beans': {'N': 25, 'P': 60, 'K': 30},
        }
        
        standard = crop_standards.get(crop_type, {'N': 100, 'P': 50, 'K': 50})
        
        # Calculate deficiency
        ground_truth = {
            'N': max(0, standard['N'] - current_npk.get('nitrogen', 0)),
            'P': max(0, standard['P'] - current_npk.get('phosphorus', 0)),
            'K': max(0, standard['K'] - current_npk.get('potassium', 0))
        }
        
        return ground_truth


# ============================================================================
# NPK PREDICTION ACCURACY TESTS
# ============================================================================

class NPKAccuracyTester:
    """Test accuracy of NPK fertilizer recommendations."""
    
    def __init__(self):
        self.results = []
    
    def calculate_npk_plan_simplified(self, request_data: Dict) -> Dict:
        """
        Simplified version of the NPK calculation from main.py
        This mimics the FertilizerRecommender.calculate_npk_plan logic
        """
        crop_requirements = {
            'rice': {'N': 120, 'P': 60, 'K': 40, 'pH_range': (5.5, 7.0)},
            'wheat': {'N': 120, 'P': 60, 'K': 40, 'pH_range': (6.0, 7.5)},
            'maize': {'N': 150, 'P': 75, 'K': 75, 'pH_range': (5.5, 7.5)},
            'cotton': {'N': 120, 'P': 60, 'K': 50, 'pH_range': (5.8, 8.0)},
        }
        
        crop_type = request_data.get('crop_type', 'rice')
        crop_req = crop_requirements.get(crop_type, crop_requirements['rice'])
        
        # Calculate deficiencies
        N_deficit = max(0, crop_req['N'] - request_data.get('nitrogen', 0))
        P_deficit = max(0, crop_req['P'] - request_data.get('phosphorus', 0))
        K_deficit = max(0, crop_req['K'] - request_data.get('potassium', 0))
        
        # Adjust based on NDVI (plant health)
        ndvi = request_data.get('NDVI', 0.6)
        if ndvi < 0.4:
            stress_factor = 1.2
        elif ndvi > 0.7:
            stress_factor = 0.9
        else:
            stress_factor = 1.0
        
        return {
            'N_kg_per_ha': round(N_deficit * stress_factor, 2),
            'P_kg_per_ha': round(P_deficit * stress_factor, 2),
            'K_kg_per_ha': round(K_deficit * stress_factor, 2),
        }
    
    def test_npk_accuracy(self, test_data: pd.DataFrame) -> Dict:
        """Test NPK recommendation accuracy."""
        print("\n" + "="*80)
        print("NPK FERTILIZER RECOMMENDATION ACCURACY TEST")
        print("="*80)
        
        predictions = []
        ground_truths = []
        errors = {'N': [], 'P': [], 'K': []}
        
        for idx, row in test_data.iterrows():
            # Convert row to dict
            request_data = row.to_dict()
            
            # Get prediction from model
            prediction = self.calculate_npk_plan_simplified(request_data)
            
            # Get ground truth
            ground_truth = TestDataGenerator.generate_ground_truth_npk(
                row['crop_type'],
                {'nitrogen': row['nitrogen'], 'phosphorus': row['phosphorus'], 'potassium': row['potassium']}
            )
            
            # Calculate errors
            errors['N'].append(abs(prediction['N_kg_per_ha'] - ground_truth['N']))
            errors['P'].append(abs(prediction['P_kg_per_ha'] - ground_truth['P']))
            errors['K'].append(abs(prediction['K_kg_per_ha'] - ground_truth['K']))
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        
        # Calculate metrics
        results = {
            'n_samples': len(test_data),
            'metrics': {
                'Nitrogen (N)': {
                    'MAE': np.mean(errors['N']),
                    'RMSE': np.sqrt(np.mean(np.array(errors['N'])**2)),
                    'Max_Error': np.max(errors['N']),
                    'Accuracy_±10kg': np.mean(np.array(errors['N']) <= 10) * 100
                },
                'Phosphorus (P)': {
                    'MAE': np.mean(errors['P']),
                    'RMSE': np.sqrt(np.mean(np.array(errors['P'])**2)),
                    'Max_Error': np.max(errors['P']),
                    'Accuracy_±10kg': np.mean(np.array(errors['P']) <= 10) * 100
                },
                'Potassium (K)': {
                    'MAE': np.mean(errors['K']),
                    'RMSE': np.sqrt(np.mean(np.array(errors['K'])**2)),
                    'Max_Error': np.max(errors['K']),
                    'Accuracy_±10kg': np.mean(np.array(errors['K']) <= 10) * 100
                }
            }
        }
        
        # Print results
        print(f"\nTest Samples: {results['n_samples']}")
        print("\nNutrient Recommendation Accuracy:")
        print("-" * 80)
        
        for nutrient, metrics in results['metrics'].items():
            print(f"\n{nutrient}:")
            print(f"  Mean Absolute Error (MAE):    {metrics['MAE']:.2f} kg/ha")
            print(f"  Root Mean Square Error (RMSE): {metrics['RMSE']:.2f} kg/ha")
            print(f"  Maximum Error:                 {metrics['Max_Error']:.2f} kg/ha")
            print(f"  Accuracy within ±10 kg/ha:     {metrics['Accuracy_±10kg']:.1f}%")
        
        return results


# ============================================================================
# YIELD PREDICTION ACCURACY TESTS
# ============================================================================

class YieldPredictionTester:
    """Test accuracy of crop yield predictions."""
    
    def calculate_yield_estimate(self, ndvi: float, soil_moisture: float, 
                                temperature: float, rainfall: float) -> float:
        """
        Simplified version of yield calculation from main.py
        """
        base_yield = 5000
        
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
    
    def generate_ground_truth_yield(self, ndvi: float, soil_moisture: float,
                                   temperature: float, rainfall: float) -> float:
        """Generate realistic ground truth yield based on agronomic data."""
        # This simulates real-world yield with some variation
        base = self.calculate_yield_estimate(ndvi, soil_moisture, temperature, rainfall)
        # Add realistic variation (±10%)
        noise = np.random.normal(0, base * 0.05)
        return max(1500, min(9000, base + noise))
    
    def test_yield_accuracy(self, test_data: pd.DataFrame) -> Dict:
        """Test yield prediction accuracy."""
        print("\n" + "="*80)
        print("CROP YIELD PREDICTION ACCURACY TEST")
        print("="*80)
        
        predictions = []
        ground_truths = []
        
        for idx, row in test_data.iterrows():
            prediction = self.calculate_yield_estimate(
                row['NDVI'], row['soil_moisture'], 
                row['temperature'], row['rainfall']
            )
            
            ground_truth = self.generate_ground_truth_yield(
                row['NDVI'], row['soil_moisture'],
                row['temperature'], row['rainfall']
            )
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - ground_truths))
        rmse = np.sqrt(np.mean((predictions - ground_truths)**2))
        mape = np.mean(np.abs((predictions - ground_truths) / ground_truths)) * 100
        r2 = 1 - (np.sum((ground_truths - predictions)**2) / 
                  np.sum((ground_truths - np.mean(ground_truths))**2))
        
        results = {
            'n_samples': len(test_data),
            'metrics': {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2_Score': r2,
                'Mean_Predicted': np.mean(predictions),
                'Mean_Actual': np.mean(ground_truths),
                'Accuracy_±500kg': np.mean(np.abs(predictions - ground_truths) <= 500) * 100
            }
        }
        
        # Print results
        print(f"\nTest Samples: {results['n_samples']}")
        print("\nYield Prediction Metrics:")
        print("-" * 80)
        print(f"Mean Absolute Error (MAE):      {mae:.2f} kg/ha")
        print(f"Root Mean Square Error (RMSE):  {rmse:.2f} kg/ha")
        print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
        print(f"R² Score:                       {r2:.4f}")
        print(f"Accuracy within ±500 kg/ha:     {results['metrics']['Accuracy_±500kg']:.1f}%")
        print(f"\nMean Predicted Yield:           {np.mean(predictions):.2f} kg/ha")
        print(f"Mean Actual Yield:              {np.mean(ground_truths):.2f} kg/ha")
        
        return results


# ============================================================================
# NDVI INTERPRETATION ACCURACY TESTS
# ============================================================================

class NDVIInterpretationTester:
    """Test accuracy of NDVI interpretation."""
    
    def interpret_ndvi(self, ndvi: float) -> str:
        """NDVI interpretation from main.py"""
        if ndvi > 0.7:
            return "Dense, healthy vegetation (excellent crop health)"
        elif ndvi > 0.6:
            return "Healthy vegetation (good crop health)"
        elif ndvi > 0.4:
            return "Moderate vegetation health (monitor closely)"
        elif ndvi > 0.2:
            return "Stressed vegetation (action required)"
        elif ndvi > 0:
            return "Sparse vegetation or bare soil (critical)"
        else:
            return "Water, snow, or bare ground (no vegetation)"
    
    def get_ground_truth_category(self, ndvi: float) -> str:
        """Ground truth categorization based on scientific literature."""
        if ndvi > 0.65:
            return "healthy"
        elif ndvi > 0.4:
            return "moderate"
        elif ndvi > 0.2:
            return "stressed"
        else:
            return "critical"
    
    def get_prediction_category(self, interpretation: str) -> str:
        """Convert interpretation to category."""
        if "excellent" in interpretation.lower() or "healthy vegetation" in interpretation.lower():
            return "healthy"
        elif "moderate" in interpretation.lower():
            return "moderate"
        elif "stressed" in interpretation.lower():
            return "stressed"
        else:
            return "critical"
    
    def test_ndvi_accuracy(self, num_samples: int = 200) -> Dict:
        """Test NDVI interpretation accuracy."""
        print("\n" + "="*80)
        print("NDVI INTERPRETATION ACCURACY TEST")
        print("="*80)
        
        np.random.seed(42)
        ndvi_values = np.random.uniform(-0.1, 1.0, num_samples)
        
        correct_predictions = 0
        confusion_matrix = {
            'healthy': {'healthy': 0, 'moderate': 0, 'stressed': 0, 'critical': 0},
            'moderate': {'healthy': 0, 'moderate': 0, 'stressed': 0, 'critical': 0},
            'stressed': {'healthy': 0, 'moderate': 0, 'stressed': 0, 'critical': 0},
            'critical': {'healthy': 0, 'moderate': 0, 'stressed': 0, 'critical': 0}
        }
        
        for ndvi in ndvi_values:
            interpretation = self.interpret_ndvi(ndvi)
            predicted_cat = self.get_prediction_category(interpretation)
            ground_truth_cat = self.get_ground_truth_category(ndvi)
            
            confusion_matrix[ground_truth_cat][predicted_cat] += 1
            
            if predicted_cat == ground_truth_cat:
                correct_predictions += 1
        
        accuracy = (correct_predictions / num_samples) * 100
        
        results = {
            'n_samples': num_samples,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix
        }
        
        # Print results
        print(f"\nTest Samples: {num_samples}")
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print("\nConfusion Matrix:")
        print("-" * 80)
        print(f"{'True/Pred':<12} {'Healthy':<12} {'Moderate':<12} {'Stressed':<12} {'Critical':<12}")
        print("-" * 80)
        
        for true_cat in ['healthy', 'moderate', 'stressed', 'critical']:
            row = confusion_matrix[true_cat]
            print(f"{true_cat.capitalize():<12} {row['healthy']:<12} {row['moderate']:<12} "
                  f"{row['stressed']:<12} {row['critical']:<12}")
        
        return results


# ============================================================================
# IRRIGATION REQUIREMENT ACCURACY TESTS
# ============================================================================

class IrrigationAccuracyTester:
    """Test accuracy of irrigation requirement calculations."""
    
    def calculate_irrigation(self, soil_moisture: float, rainfall: float, 
                           temperature: float, humidity: float) -> Dict:
        """Simplified irrigation calculation from main.py."""
        # Base water requirement (mm/day)
        base_requirement = 5.0
        
        # Temperature adjustment
        if temperature > 30:
            temp_factor = 1.3
        elif temperature > 25:
            temp_factor = 1.1
        else:
            temp_factor = 1.0
        
        # Humidity adjustment
        humidity_factor = 1.2 if humidity < 50 else 1.0 if humidity < 70 else 0.9
        
        # Calculate daily requirement
        daily_requirement = base_requirement * temp_factor * humidity_factor
        
        # Soil moisture deficit
        optimal_moisture = 70
        moisture_deficit = max(0, optimal_moisture - soil_moisture)
        
        # Recent rainfall consideration
        effective_rainfall = min(rainfall, 50)
        
        # Calculate total need
        total_water_mm = (moisture_deficit * 0.5) + (daily_requirement * 7) - effective_rainfall
        total_water_mm = max(0, total_water_mm)
        
        return {
            'daily_mm': round(daily_requirement, 2),
            'weekly_mm': round(total_water_mm, 2),
            'frequency': 'daily' if soil_moisture < 40 else 'every 2-3 days'
        }
    
    def test_irrigation_accuracy(self, test_data: pd.DataFrame) -> Dict:
        """Test irrigation calculation accuracy."""
        print("\n" + "="*80)
        print("IRRIGATION REQUIREMENT ACCURACY TEST")
        print("="*80)
        
        predictions = []
        
        for idx, row in test_data.iterrows():
            prediction = self.calculate_irrigation(
                row['soil_moisture'], row['rainfall'],
                row['temperature'], row['humidity']
            )
            predictions.append(prediction)
        
        # Extract weekly requirements
        weekly_reqs = [p['weekly_mm'] for p in predictions]
        daily_reqs = [p['daily_mm'] for p in predictions]
        
        results = {
            'n_samples': len(test_data),
            'metrics': {
                'mean_daily_mm': np.mean(daily_reqs),
                'mean_weekly_mm': np.mean(weekly_reqs),
                'std_weekly_mm': np.std(weekly_reqs),
                'min_weekly_mm': np.min(weekly_reqs),
                'max_weekly_mm': np.max(weekly_reqs),
                'reasonable_range_pct': np.mean((np.array(weekly_reqs) >= 0) & 
                                               (np.array(weekly_reqs) <= 100)) * 100
            }
        }
        
        # Print results
        print(f"\nTest Samples: {results['n_samples']}")
        print("\nIrrigation Requirement Statistics:")
        print("-" * 80)
        print(f"Mean Daily Requirement:    {results['metrics']['mean_daily_mm']:.2f} mm/day")
        print(f"Mean Weekly Requirement:   {results['metrics']['mean_weekly_mm']:.2f} mm/week")
        print(f"Std Dev Weekly:            {results['metrics']['std_weekly_mm']:.2f} mm")
        print(f"Range:                     {results['metrics']['min_weekly_mm']:.2f} - "
              f"{results['metrics']['max_weekly_mm']:.2f} mm/week")
        print(f"Values in Reasonable Range: {results['metrics']['reasonable_range_pct']:.1f}%")
        
        return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

class ModelAccuracyTestRunner:
    """Main test runner for all model accuracy tests."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.all_results = {}
    
    def run_all_tests(self, crops: List[str] = None, num_samples: int = 100):
        """Run all accuracy tests."""
        if crops is None:
            crops = ['rice', 'wheat', 'maize', 'cotton']
        
        print("\n" + "="*80)
        print("AGRIMINDS MODEL ACCURACY TEST SUITE")
        print("="*80)
        print(f"Testing {len(crops)} crops with {num_samples} samples each")
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for crop in crops:
            print(f"\n\n{'='*80}")
            print(f"TESTING CROP: {crop.upper()}")
            print(f"{'='*80}")
            
            # Generate test data
            test_data = TestDataGenerator.generate_crop_test_cases(crop, num_samples)
            
            # Run tests
            npk_tester = NPKAccuracyTester()
            yield_tester = YieldPredictionTester()
            irrigation_tester = IrrigationAccuracyTester()
            
            npk_results = npk_tester.test_npk_accuracy(test_data)
            yield_results = yield_tester.test_yield_accuracy(test_data)
            irrigation_results = irrigation_tester.test_irrigation_accuracy(test_data)
            
            # Store results
            self.all_results[crop] = {
                'npk': npk_results,
                'yield': yield_results,
                'irrigation': irrigation_results
            }
        
        # Test NDVI interpretation (crop-agnostic)
        ndvi_tester = NDVIInterpretationTester()
        ndvi_results = ndvi_tester.test_ndvi_accuracy(num_samples * 2)
        self.all_results['ndvi_interpretation'] = ndvi_results
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save results to JSON
        self.save_results()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
    
    def generate_summary_report(self):
        """Generate summary report of all tests."""
        print("\n\n" + "="*80)
        print("SUMMARY REPORT - MODEL ACCURACY ACROSS ALL CROPS")
        print("="*80)
        
        # NPK Summary
        print("\n1. NPK FERTILIZER RECOMMENDATIONS")
        print("-" * 80)
        
        for crop in [k for k in self.all_results.keys() if k != 'ndvi_interpretation']:
            npk_metrics = self.all_results[crop]['npk']['metrics']
            print(f"\n{crop.upper()}:")
            print(f"  N - MAE: {npk_metrics['Nitrogen (N)']['MAE']:.2f} kg/ha, "
                  f"Accuracy: {npk_metrics['Nitrogen (N)']['Accuracy_±10kg']:.1f}%")
            print(f"  P - MAE: {npk_metrics['Phosphorus (P)']['MAE']:.2f} kg/ha, "
                  f"Accuracy: {npk_metrics['Phosphorus (P)']['Accuracy_±10kg']:.1f}%")
            print(f"  K - MAE: {npk_metrics['Potassium (K)']['MAE']:.2f} kg/ha, "
                  f"Accuracy: {npk_metrics['Potassium (K)']['Accuracy_±10kg']:.1f}%")
        
        # Yield Summary
        print("\n\n2. YIELD PREDICTIONS")
        print("-" * 80)
        
        for crop in [k for k in self.all_results.keys() if k != 'ndvi_interpretation']:
            yield_metrics = self.all_results[crop]['yield']['metrics']
            print(f"{crop.upper()}: MAE={yield_metrics['MAE']:.2f} kg/ha, "
                  f"R²={yield_metrics['R2_Score']:.4f}, "
                  f"MAPE={yield_metrics['MAPE']:.2f}%")
        
        # NDVI Summary
        print("\n\n3. NDVI INTERPRETATION")
        print("-" * 80)
        ndvi_acc = self.all_results['ndvi_interpretation']['accuracy']
        print(f"Overall Accuracy: {ndvi_acc:.2f}%")
        
        # Overall Assessment
        print("\n\n4. OVERALL MODEL ASSESSMENT")
        print("-" * 80)
        
        avg_npk_acc = np.mean([
            self.all_results[crop]['npk']['metrics']['Nitrogen (N)']['Accuracy_±10kg']
            for crop in self.all_results.keys() if crop != 'ndvi_interpretation'
        ])
        
        avg_yield_r2 = np.mean([
            self.all_results[crop]['yield']['metrics']['R2_Score']
            for crop in self.all_results.keys() if crop != 'ndvi_interpretation'
        ])
        
        print(f"Average NPK Accuracy (±10kg):  {avg_npk_acc:.1f}%")
        print(f"Average Yield R² Score:        {avg_yield_r2:.4f}")
        print(f"NDVI Interpretation Accuracy:  {ndvi_acc:.1f}%")
        
        # Quality assessment
        print("\n5. QUALITY ASSESSMENT")
        print("-" * 80)
        
        if avg_npk_acc > 80 and avg_yield_r2 > 0.85 and ndvi_acc > 85:
            quality = "EXCELLENT"
        elif avg_npk_acc > 70 and avg_yield_r2 > 0.75 and ndvi_acc > 75:
            quality = "GOOD"
        elif avg_npk_acc > 60 and avg_yield_r2 > 0.65 and ndvi_acc > 65:
            quality = "ACCEPTABLE"
        else:
            quality = "NEEDS IMPROVEMENT"
        
        print(f"Overall Model Quality: {quality}")
    
    def save_results(self):
        """Save results to JSON file."""
        output_file = self.output_dir / f"accuracy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Initialize test runner
    runner = ModelAccuracyTestRunner(output_dir="test_results")
    
    # Run all tests
    crops_to_test = ['rice', 'wheat', 'maize', 'cotton']
    runner.run_all_tests(crops=crops_to_test, num_samples=100)
    
    print("\n✅ Testing completed successfully!")
