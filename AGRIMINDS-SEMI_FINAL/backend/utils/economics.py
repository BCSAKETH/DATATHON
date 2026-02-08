"""
Economics & Sustainability Module
==================================
Calculates financial and environmental metrics:
1. Green Score (environmental impact)
2. ROI (Return on Investment)
3. Carbon footprint estimation
4. Cost-benefit analysis
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class FertilizerPricing:
    """Current market prices in India (INR per 50kg bag)"""
    urea_price: float = 266.0      # 46% N
    dap_price: float = 1350.0      # 18% N, 46% P
    mop_price: float = 1200.0      # 60% K
    ssp_price: float = 450.0       # 16% P
    complexes_npk: float = 1500.0  # 10-26-26 or similar


class EconomicsCalculator:
    """
    Financial and environmental analysis for fertilizer recommendations
    """
    
    def __init__(self):
        self.pricing = FertilizerPricing()
        
        # Fertilizer composition (nutrient percentage)
        self.fertilizer_composition = {
            'urea': {'N': 0.46, 'P': 0, 'K': 0},
            'dap': {'N': 0.18, 'P': 0.46, 'K': 0},
            'mop': {'N': 0, 'P': 0, 'K': 0.60},
            'ssp': {'N': 0, 'P': 0.16, 'K': 0}
        }
        
        # Carbon emission factors (kg CO2 per kg fertilizer)
        # Source: IPCC, FAO studies
        self.carbon_emissions = {
            'N': 5.8,  # High energy for Haber-Bosch process
            'P': 0.9,
            'K': 0.6
        }
        
        # Crop yield response to NPK (kg yield increase per kg nutrient)
        # These are averages - actual response curves are non-linear
        self.yield_response = {
            'rice': {'N': 15, 'P': 10, 'K': 8},
            'wheat': {'N': 18, 'P': 12, 'K': 10},
            'maize': {'N': 20, 'P': 15, 'K': 12},
            'cotton': {'N': 8, 'P': 6, 'K': 10},
            'sugarcane': {'N': 25, 'P': 18, 'K': 20},
            'default': {'N': 12, 'P': 8, 'K': 7}
        }
        
        # Market prices for major crops (INR per kg)
        self.crop_prices = {
            'rice': 25,
            'wheat': 22,
            'maize': 18,
            'cotton': 60,  # per kg lint
            'sugarcane': 3.5,
            'default': 20
        }
    
    def calculate_green_score(self, npk_recommendations: Dict, 
                             field_area: float) -> Dict:
        """
        Calculate environmental impact score (0-100, higher = better)
        
        Methodology:
        -----------
        Green Score = 100 - (Composite Environmental Impact Index)
        
        Impact factors:
        1. Overapplication penalty (compare to agronomic optimum)
        2. Carbon footprint from fertilizer production
        3. Nutrient loss potential (leaching, runoff risk)
        4. Soil health degradation risk
        
        Formula:
        --------
        Total NPK = N + P + K (kg/hectare)
        
        Overapplication = Total NPK / Optimal Range
        where Optimal Range = 200-300 kg/ha for most crops
        
        Carbon footprint = Σ(nutrient_kg × emission_factor)
        
        Green Score = 100 - min(100, overapplication_penalty + carbon_penalty)
        
        Returns:
            {
                'green_score': int (0-100),
                'carbon_footprint_kg': float,
                'leaching_risk': str ('low', 'medium', 'high'),
                'sustainability_rating': str
            }
        """
        N = npk_recommendations.get('N', 0)
        P = npk_recommendations.get('P', 0)
        K = npk_recommendations.get('K', 0)
        
        total_npk = N + P + K
        
        if total_npk == 0:
            return {
                'green_score': 100,
                'carbon_footprint_kg': 0,
                'leaching_risk': 'none',
                'sustainability_rating': 'Excellent'
            }
        
        # 1. Overapplication penalty
        npk_per_hectare = total_npk / field_area
        optimal_min = 200  # kg/ha
        optimal_max = 300  # kg/ha
        
        if npk_per_hectare <= optimal_max:
            overapplication_penalty = 0
        else:
            # Exponential penalty for excessive use
            overapplication_penalty = min(50, 
                (npk_per_hectare - optimal_max) / optimal_max * 100)
        
        # 2. Carbon footprint
        carbon_footprint = (
            N * self.carbon_emissions['N'] +
            P * self.carbon_emissions['P'] +
            K * self.carbon_emissions['K']
        )
        
        carbon_per_hectare = carbon_footprint / field_area
        carbon_penalty = min(30, carbon_per_hectare / 10)  # Scale to 0-30
        
        # 3. Leaching risk (high N = high risk)
        n_per_hectare = N / field_area
        if n_per_hectare > 150:
            leaching_risk = 'high'
            leaching_penalty = 20
        elif n_per_hectare > 100:
            leaching_risk = 'medium'
            leaching_penalty = 10
        else:
            leaching_risk = 'low'
            leaching_penalty = 0
        
        # Calculate final score
        total_penalty = overapplication_penalty + carbon_penalty + leaching_penalty
        green_score = max(0, int(100 - total_penalty))
        
        # Sustainability rating
        if green_score >= 80:
            rating = 'Excellent - Environmentally optimized'
        elif green_score >= 60:
            rating = 'Good - Acceptable impact'
        elif green_score >= 40:
            rating = 'Fair - Consider reduction'
        else:
            rating = 'Poor - High environmental risk'
        
        return {
            'green_score': green_score,
            'carbon_footprint_kg': round(carbon_footprint, 2),
            'carbon_per_hectare': round(carbon_per_hectare, 2),
            'leaching_risk': leaching_risk,
            'sustainability_rating': rating,
            'breakdown': {
                'overapplication_penalty': round(overapplication_penalty, 1),
                'carbon_penalty': round(carbon_penalty, 1),
                'leaching_penalty': leaching_penalty
            }
        }
    
    def calculate_fertilizer_cost(self, npk_recommendations: Dict) -> Dict:
        """
        Calculate total fertilizer purchase cost
        
        Strategy:
        --------
        For each nutrient, choose the most cost-effective fertilizer source:
        - N: Urea (cheapest nitrogen source)
        - P: DAP or SSP (choose based on N needs)
        - K: MOP (only economical potash source in India)
        
        Returns:
            {
                'total_cost': float (INR),
                'breakdown': {fertilizer_name: {quantity, cost}},
                'cost_per_hectare': float
            }
        """
        N_needed = npk_recommendations.get('N', 0)
        P_needed = npk_recommendations.get('P', 0)
        K_needed = npk_recommendations.get('K', 0)
        
        breakdown = {}
        total_cost = 0
        
        # 1. Nitrogen from Urea
        if N_needed > 0:
            urea_kg = N_needed / self.fertilizer_composition['urea']['N']
            urea_bags = np.ceil(urea_kg / 50)  # Sold in 50kg bags
            urea_cost = urea_bags * self.pricing.urea_price
            
            breakdown['Urea'] = {
                'quantity_kg': round(urea_kg, 1),
                'bags': int(urea_bags),
                'cost': round(urea_cost, 2)
            }
            total_cost += urea_cost
        
        # 2. Phosphorus from DAP (also provides some N)
        if P_needed > 0:
            dap_kg = P_needed / self.fertilizer_composition['dap']['P']
            dap_bags = np.ceil(dap_kg / 50)
            dap_cost = dap_bags * self.pricing.dap_price
            
            # DAP also provides nitrogen (18%)
            n_from_dap = dap_kg * self.fertilizer_composition['dap']['N']
            
            breakdown['DAP'] = {
                'quantity_kg': round(dap_kg, 1),
                'bags': int(dap_bags),
                'cost': round(dap_cost, 2),
                'bonus_N_kg': round(n_from_dap, 1)
            }
            total_cost += dap_cost
        
        # 3. Potassium from MOP
        if K_needed > 0:
            mop_kg = K_needed / self.fertilizer_composition['mop']['K']
            mop_bags = np.ceil(mop_kg / 50)
            mop_cost = mop_bags * self.pricing.mop_price
            
            breakdown['MOP'] = {
                'quantity_kg': round(mop_kg, 1),
                'bags': int(mop_bags),
                'cost': round(mop_cost, 2)
            }
            total_cost += mop_cost
        
        return {
            'total_cost': round(total_cost, 2),
            'breakdown': breakdown,
            'currency': 'INR'
        }
    
    def calculate_roi(self, npk_recommendations: Dict, field_area: float,
                     crop_type: str = 'default', current_yield: float = None) -> Dict:
        """
        Estimate Return on Investment
        
        ROI Formula:
        -----------
        Expected Yield Increase = Σ(nutrient_kg × yield_response)
        Revenue Increase = Yield Increase × Crop Price
        
        ROI (%) = (Revenue Increase - Fertilizer Cost) / Fertilizer Cost × 100
        
        Payback Period = Fertilizer Cost / (Revenue Increase - Fertilizer Cost)
        
        Args:
            npk_recommendations: NPK in kg
            field_area: hectares
            crop_type: Crop name for yield response curves
            current_yield: Current yield (kg/ha) for baseline
        
        Returns:
            {
                'roi_percentage': float,
                'revenue_increase': float (INR),
                'net_profit': float (INR),
                'payback_seasons': float,
                'recommendation': str
            }
        """
        # Get crop-specific parameters
        crop = crop_type.lower()
        yield_response = self.yield_response.get(crop, self.yield_response['default'])
        crop_price = self.crop_prices.get(crop, self.crop_prices['default'])
        
        # Calculate expected yield increase
        N = npk_recommendations.get('N', 0)
        P = npk_recommendations.get('P', 0)
        K = npk_recommendations.get('K', 0)
        
        # Diminishing returns: Use square root for more realistic response
        yield_increase_per_ha = (
            yield_response['N'] * np.sqrt(N / field_area) +
            yield_response['P'] * np.sqrt(P / field_area) +
            yield_response['K'] * np.sqrt(K / field_area)
        )
        
        total_yield_increase = yield_increase_per_ha * field_area  # kg
        
        # Revenue calculation
        revenue_increase = total_yield_increase * crop_price
        
        # Cost calculation
        cost_data = self.calculate_fertilizer_cost(npk_recommendations)
        fertilizer_cost = cost_data['total_cost']
        
        # ROI
        if fertilizer_cost == 0:
            roi_percentage = 0
            payback_seasons = 0
        else:
            net_profit = revenue_increase - fertilizer_cost
            roi_percentage = (net_profit / fertilizer_cost) * 100
            
            # Payback period (in crop seasons)
            if net_profit > 0:
                payback_seasons = fertilizer_cost / net_profit
            else:
                payback_seasons = float('inf')
        
        # Investment recommendation
        if roi_percentage > 200:
            recommendation = "Highly Profitable - Strong investment"
        elif roi_percentage > 100:
            recommendation = "Profitable - Recommended"
        elif roi_percentage > 50:
            recommendation = "Moderately Profitable - Consider"
        elif roi_percentage > 0:
            recommendation = "Marginally Profitable - Evaluate alternatives"
        else:
            recommendation = "Not Profitable - Reduce application"
        
        return {
            'roi_percentage': round(roi_percentage, 1),
            'revenue_increase': round(revenue_increase, 2),
            'fertilizer_cost': round(fertilizer_cost, 2),
            'net_profit': round(revenue_increase - fertilizer_cost, 2),
            'payback_seasons': round(payback_seasons, 2) if payback_seasons != float('inf') else '>10',
            'yield_increase_kg': round(total_yield_increase, 1),
            'yield_increase_per_ha': round(yield_increase_per_ha, 1),
            'recommendation': recommendation,
            'break_even_analysis': {
                'current_yield_assumed': current_yield or 'N/A',
                'projected_yield_increase_percent': round(
                    (yield_increase_per_ha / (current_yield or 3000)) * 100, 1
                ) if current_yield else 'N/A'
            }
        }
    
    def comparative_analysis(self, recommendations: List[Dict]) -> Dict:
        """
        Compare multiple fertilizer strategies
        
        Args:
            recommendations: List of {npk, area, crop} dictionaries
        
        Returns:
            Best strategy based on ROI and Green Score
        """
        results = []
        
        for i, rec in enumerate(recommendations):
            roi = self.calculate_roi(
                rec['npk'],
                rec['area'],
                rec.get('crop', 'default')
            )
            green = self.calculate_green_score(rec['npk'], rec['area'])
            
            # Composite score (70% ROI, 30% Green)
            composite = 0.7 * roi['roi_percentage'] + 0.3 * green['green_score']
            
            results.append({
                'strategy_id': i,
                'roi': roi,
                'green_score': green,
                'composite_score': composite
            })
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return {
            'best_strategy': results[0],
            'all_strategies': results
        }


if __name__ == "__main__":
    # Example usage
    calc = EconomicsCalculator()
    
    # Test scenario
    npk = {'N': 50, 'P': 30, 'K': 40}
    area = 2.5  # hectares
    
    print("=== Green Score ===")
    green = calc.calculate_green_score(npk, area)
    print(f"Score: {green['green_score']}/100")
    print(f"Rating: {green['sustainability_rating']}")
    print(f"Carbon: {green['carbon_footprint_kg']} kg CO2")
    
    print("\n=== Cost Analysis ===")
    cost = calc.calculate_fertilizer_cost(npk)
    print(f"Total: ₹{cost['total_cost']}")
    for fert, details in cost['breakdown'].items():
        print(f"  {fert}: {details['bags']} bags = ₹{details['cost']}")
    
    print("\n=== ROI Analysis ===")
    roi = calc.calculate_roi(npk, area, 'rice', current_yield=4000)
    print(f"ROI: {roi['roi_percentage']}%")
    print(f"Net Profit: ₹{roi['net_profit']}")
    print(f"Recommendation: {roi['recommendation']}")
    
    print("\n✅ Economics module loaded successfully!")
