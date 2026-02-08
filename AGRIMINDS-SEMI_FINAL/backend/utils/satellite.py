"""
Satellite Data Integration Module
==================================
Interfaces with Sentinel Hub API and OpenWeather Agromonitoring
to fetch NDVI, EVI, and other vegetation indices.

For production: Replace mock functions with actual API calls.
"""

import httpx
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import asyncio


class SatelliteDataProvider:
    """
    Fetches satellite imagery and vegetation indices
    
    Supported APIs:
    1. Sentinel Hub (ESA Copernicus) - High resolution (10m)
    2. OpenWeather Agro API - Real-time monitoring
    3. NASA MODIS - Long-term historical data
    
    NDVI Physics:
    ------------
    NDVI = (NIR - RED) / (NIR + RED)
    
    Where:
    - NIR = Near-Infrared reflectance (Sentinel-2 Band 8, 842nm)
    - RED = Red reflectance (Sentinel-2 Band 4, 665nm)
    
    Healthy vegetation:
    - Absorbs RED light (photosynthesis) → Low RED reflectance
    - Reflects NIR light (cell structure) → High NIR reflectance
    - Result: NDVI close to +1
    
    Stressed/dead vegetation:
    - Reflects both RED and NIR similarly
    - Result: NDVI close to 0 or negative
    """
    
    def __init__(self, sentinel_api_key: Optional[str] = None, 
                 openweather_api_key: Optional[str] = None):
        """
        Initialize satellite data provider
        
        Args:
            sentinel_api_key: Sentinel Hub API key (get from https://www.sentinel-hub.com/)
            openweather_api_key: OpenWeather Agro API key
        """
        self.sentinel_api_key = sentinel_api_key
        self.openweather_api_key = openweather_api_key
        self.sentinel_base_url = "https://services.sentinel-hub.com"
        self.openweather_base_url = "http://api.agromonitoring.com/agro/1.0"
    
    async def fetch_ndvi(self, lat: float, lon: float, 
                        date: Optional[str] = None) -> Dict:
        """
        Fetch NDVI for a given location
        
        Args:
            lat: Latitude
            lon: Longitude  
            date: ISO date string (YYYY-MM-DD). If None, uses most recent available.
        
        Returns:
            {
                'ndvi': float,
                'evi': float,
                'date': str,
                'cloud_coverage': float,
                'resolution': str
            }
        """
        if self.sentinel_api_key:
            return await self._fetch_sentinel_ndvi(lat, lon, date)
        else:
            # Mock data for development/testing
            return self._generate_mock_ndvi(lat, lon)
    
    async def _fetch_sentinel_ndvi(self, lat: float, lon: float, 
                                   date: Optional[str]) -> Dict:
        """
        Actual Sentinel Hub API call (requires authentication)
        
        Sentinel-2 has 5-day revisit time at equator.
        Resolution: 10m per pixel
        """
        # TODO: Implement actual Sentinel Hub API integration
        # Reference: https://docs.sentinel-hub.com/api/latest/
        
        bbox = self._create_bbox(lat, lon, buffer_km=0.5)
        
        if not date:
            date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        
        # Evalscript for NDVI calculation
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08", "SCL"],  // RED, NIR, Scene Classification
                output: { bands: 3 }
            };
        }
        
        function evaluatePixel(sample) {
            // Calculate NDVI
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            
            // Calculate EVI (Enhanced Vegetation Index)
            let evi = 2.5 * ((sample.B08 - sample.B04) / 
                            (sample.B08 + 6 * sample.B04 - 7.5 * sample.B02 + 1));
            
            return [ndvi, evi, sample.SCL];  // Return NDVI, EVI, cloud mask
        }
        """
        
        request_payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z"
                        },
                        "maxCloudCoverage": 30
                    }
                }]
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
            },
            "evalscript": evalscript
        }
        
        # In production, make actual API call:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         f"{self.sentinel_base_url}/api/v1/process",
        #         json=request_payload,
        #         headers={"Authorization": f"Bearer {self.sentinel_api_key}"}
        #     )
        #     # Process TIFF response to extract NDVI statistics
        
        # For now, return mock data
        return self._generate_mock_ndvi(lat, lon)
    
    def _generate_mock_ndvi(self, lat: float, lon: float) -> Dict:
        """
        Generate realistic mock NDVI data for testing
        
        Uses geographical heuristics:
        - Tropical regions (|lat| < 23.5°): Higher NDVI (0.5-0.8)
        - Temperate regions (23.5° < |lat| < 66.5°): Moderate NDVI (0.3-0.6)
        - Polar regions (|lat| > 66.5°): Low NDVI (0.1-0.3)
        - Coastal areas: Variable NDVI
        """
        # Base NDVI from latitude
        abs_lat = abs(lat)
        if abs_lat < 23.5:  # Tropics
            base_ndvi = 0.65
        elif abs_lat < 66.5:  # Temperate
            base_ndvi = 0.45
        else:  # Polar
            base_ndvi = 0.2
        
        # Add some spatial variation based on longitude
        lon_variation = np.sin(np.radians(lon)) * 0.1
        
        # Add random noise (simulating field variability)
        noise = np.random.uniform(-0.05, 0.05)
        
        # Seasonal variation (assume Northern Hemisphere)
        month = datetime.now().month
        if 3 <= month <= 5:  # Spring
            seasonal_factor = 1.1
        elif 6 <= month <= 8:  # Summer
            seasonal_factor = 1.2
        elif 9 <= month <= 11:  # Fall
            seasonal_factor = 0.9
        else:  # Winter
            seasonal_factor = 0.7
        
        ndvi = np.clip(base_ndvi * seasonal_factor + lon_variation + noise, -1, 1)
        
        # EVI is typically similar to NDVI but more sensitive to high biomass
        evi = np.clip(ndvi * 1.15 + noise * 0.5, -1, 1)
        
        return {
            'ndvi': round(float(ndvi), 3),
            'evi': round(float(evi), 3),
            'date': datetime.now().strftime("%Y-%m-%d"),
            'cloud_coverage': round(np.random.uniform(0, 20), 1),
            'resolution': '10m (Sentinel-2)',
            'source': 'mock_data',
            'location': {'lat': lat, 'lon': lon}
        }
    
    def _create_bbox(self, lat: float, lon: float, buffer_km: float = 0.5) -> list:
        """
        Create bounding box around point
        
        Args:
            lat, lon: Center point
            buffer_km: Buffer distance in kilometers
        
        Returns:
            [min_lon, min_lat, max_lon, max_lat]
        """
        # Approximate: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111*cos(lat) km
        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
        
        return [
            lon - lon_buffer,
            lat - lat_buffer,
            lon + lon_buffer,
            lat + lat_buffer
        ]
    
    async def fetch_historical_ndvi(self, lat: float, lon: float, 
                                    start_date: str, end_date: str) -> list:
        """
        Fetch NDVI time series for trend analysis
        
        Args:
            lat, lon: Location
            start_date, end_date: Date range (YYYY-MM-DD)
        
        Returns:
            List of {date, ndvi, evi} dictionaries
        """
        # Mock implementation: Generate 10 data points
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        delta = (end - start) / 10
        
        time_series = []
        for i in range(10):
            date = start + delta * i
            data = self._generate_mock_ndvi(lat, lon)
            data['date'] = date.strftime("%Y-%m-%d")
            time_series.append(data)
        
        return time_series
    
    async def fetch_ndvi_heatmap(self, bbox: list, resolution: int = 256) -> Dict:
        """
        Fetch NDVI raster for map visualization
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            resolution: Image size in pixels
        
        Returns:
            {
                'image_url': str,  # URL to NDVI raster
                'ndvi_stats': {'min', 'max', 'mean', 'std'},
                'colormap': 'RdYlGn'  # Red-Yellow-Green
            }
        """
        # In production, fetch actual Sentinel-2 tile
        # For now, generate synthetic heatmap
        
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        
        # Generate grid of NDVI values
        x = np.linspace(bbox[0], bbox[2], resolution)
        y = np.linspace(bbox[1], bbox[3], resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Create realistic spatial pattern using Perlin-like noise
        base = self._generate_mock_ndvi(center_lat, center_lon)['ndvi']
        
        # Add spatial variation
        pattern = np.sin(xx * 10) * np.cos(yy * 10) * 0.2
        noise = np.random.normal(0, 0.05, (resolution, resolution))
        
        ndvi_grid = np.clip(base + pattern + noise, 0, 1)
        
        return {
            'ndvi_grid': ndvi_grid.tolist(),
            'ndvi_stats': {
                'min': float(ndvi_grid.min()),
                'max': float(ndvi_grid.max()),
                'mean': float(ndvi_grid.mean()),
                'std': float(ndvi_grid.std())
            },
            'colormap': 'RdYlGn',
            'bbox': bbox,
            'resolution': resolution
        }


# Weather data integration
class WeatherProvider:
    """Fetch weather forecasts for rain lock logic"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    async def fetch_forecast(self, lat: float, lon: float) -> Dict:
        """
        Fetch 48-hour weather forecast
        
        Returns:
            {
                'rainfall_forecast': float (mm),
                'temperature': float (°C),
                'humidity': float (%)
            }
        """
        if self.api_key:
            # Real API call
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/forecast",
                    params={
                        'lat': lat,
                        'lon': lon,
                        'appid': self.api_key,
                        'units': 'metric'
                    }
                )
                data = response.json()
                
                # Sum rainfall for next 48 hours
                rainfall = sum(
                    item.get('rain', {}).get('3h', 0) 
                    for item in data['list'][:16]  # Next 48 hours (3h intervals)
                )
                
                return {
                    'rainfall_forecast': rainfall,
                    'temperature': data['list'][0]['main']['temp'],
                    'humidity': data['list'][0]['main']['humidity']
                }
        else:
            # Mock data
            return {
                'rainfall_forecast': np.random.uniform(0, 15),
                'temperature': np.random.uniform(20, 35),
                'humidity': np.random.uniform(40, 90)
            }


if __name__ == "__main__":
    # Test the module
    async def test():
        provider = SatelliteDataProvider()
        
        # Test NDVI fetch
        result = await provider.fetch_ndvi(17.385, 78.486)  # Hyderabad
        print("NDVI Data:", result)
        
        # Test heatmap
        bbox = [78.4, 17.3, 78.5, 17.4]
        heatmap = await provider.fetch_ndvi_heatmap(bbox)
        print("Heatmap Stats:", heatmap['ndvi_stats'])
    
    asyncio.run(test())
    print("✅ Satellite module loaded successfully!")
