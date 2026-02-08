import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polygon, Popup, useMapEvents } from 'react-leaflet';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line } from 'recharts';
import 'leaflet/dist/leaflet.css';

/**
 * AgriMinds Dashboard - Updated Version
 * Takes crop type as input and provides fertilizer + irrigation recommendations
 */

const API_BASE_URL = 'http://localhost:8000';

// ============================================================================
// SUPPORTED CROPS
// ============================================================================

const SUPPORTED_CROPS = [
  'rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'chickpea', 'kidney_beans',
  'pigeon_peas', 'moth_beans', 'mung_bean', 'black_gram', 'lentil',
  'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
  'apple', 'orange', 'papaya', 'coconut', 'jute', 'coffee'
];

// ============================================================================
// MAP COMPONENTS
// ============================================================================

const FieldDrawer = ({ onPolygonComplete }) => {
  const [positions, setPositions] = useState([]);

  useMapEvents({
    click(e) {
      const newPos = [e.latlng.lat, e.latlng.lng];
      setPositions([...positions, newPos]);
      
      // Auto-complete polygon after 4 points
      if (positions.length >= 3) {
        onPolygonComplete([...positions, newPos]);
        setPositions([]);
      }
    },
  });

  return positions.length > 0 ? (
    <Polygon positions={positions} color="blue" fillOpacity={0.2} />
  ) : null;
};

const NDVIHeatmapLayer = ({ fieldBounds }) => {
  const [ndviData, setNdviData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (fieldBounds && fieldBounds.length > 0) {
      const lats = fieldBounds.map(p => p[0]);
      const lngs = fieldBounds.map(p => p[1]);
      const centerLat = lats.reduce((a, b) => a + b) / lats.length;
      const centerLng = lngs.reduce((a, b) => a + b) / lngs.length;

      console.log(`Fetching NDVI for: ${centerLat}, ${centerLng}`);

      fetch(`${API_BASE_URL}/satellite/ndvi?lat=${centerLat}&lon=${centerLng}`)
        .then(res => {
          console.log('NDVI Response status:', res.status);
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }
          return res.json();
        })
        .then(data => {
          console.log('NDVI Data:', data);
          setNdviData(data);
          setError(null);
        })
        .catch(err => {
          console.error('NDVI fetch error:', err);
          setError(err.message);
        });
    }
  }, [fieldBounds]);

  if (error) {
    console.warn('NDVI layer error:', error);
  }

  if (!fieldBounds || !ndviData) return null;

  const getColor = (ndvi) => {
    if (ndvi > 0.6) return '#00ff00';
    if (ndvi > 0.4) return '#ffff00';
    if (ndvi > 0.2) return '#ff9900';
    return '#ff0000';
  };

  return (
    <Polygon 
      positions={fieldBounds} 
      color={getColor(ndviData.ndvi)}
      fillOpacity={0.5}
    >
      <Popup>
        <div className="ndvi-popup">
          <h4>Satellite Analysis</h4>
          <p><strong>NDVI:</strong> {ndviData.ndvi.toFixed(3)}</p>
          <p><strong>Status:</strong> {ndviData.interpretation}</p>
          <p><strong>Source:</strong> {ndviData.source}</p>
        </div>
      </Popup>
    </Polygon>
  );
};

// ============================================================================
// MAIN DASHBOARD COMPONENT
// ============================================================================

const Dashboard = () => {
  // State Management
  const [fieldBounds, setFieldBounds] = useState(null);
  const [soilData, setSoilData] = useState({
    crop_type: 'rice',
    N: 50,
    P: 30,
    K: 80,
    temperature: 25,
    humidity: 65,
    soil_moisture: 50,
    rainfall: 150,
    NDVI: 0.5,
    forecast_rain: 0
  });
  
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availableCrops, setAvailableCrops] = useState([]);

  // Fetch available crops on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/crops`)
      .then(res => res.json())
      .then(data => {
        console.log('Available crops:', data);
        setAvailableCrops(data.supported_crops || SUPPORTED_CROPS);
      })
      .catch(err => {
        console.error('Failed to fetch crops:', err);
        setAvailableCrops(SUPPORTED_CROPS);
      });
  }, []);

  // ============================================================================
  // API CALLS
  // ============================================================================

  const getRecommendation = async () => {
    console.log('🌾 Getting fertilizer + irrigation recommendation...');
    setLoading(true);
    setError(null);
    
    try {
      console.log('Recommendation payload:', soilData);
      
      const response = await fetch(`${API_BASE_URL}/predict/fertilizer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(soilData)
      });
      
      console.log('Recommendation response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Recommendation error response:', errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      const data = await response.json();
      console.log('Recommendation data:', data);
      setRecommendation(data);
    } catch (error) {
      console.error('❌ Recommendation error:', error);
      setError(`Recommendation failed: ${error.message}`);
      alert(`Failed to get recommendation: ${error.message}\n\nCheck console for details.`);
    } finally {
      setLoading(false);
    }
  };

  // Auto-fetch NDVI when field is drawn
  useEffect(() => {
    if (fieldBounds && fieldBounds.length > 0) {
      const lats = fieldBounds.map(p => p[0]);
      const lngs = fieldBounds.map(p => p[1]);
      const centerLat = lats.reduce((a, b) => a + b) / lats.length;
      const centerLng = lngs.reduce((a, b) => a + b) / lngs.length;

      console.log('Auto-fetching NDVI for field center...');

      fetch(`${API_BASE_URL}/satellite/ndvi?lat=${centerLat}&lon=${centerLng}`)
        .then(res => res.json())
        .then(data => {
          console.log('Auto-NDVI data:', data);
          setSoilData(prev => ({ ...prev, NDVI: data.ndvi }));
        })
        .catch(err => console.error('Auto NDVI fetch error:', err));
    }
  }, [fieldBounds]);

  // ============================================================================
  // RENDER HELPER COMPONENTS
  // ============================================================================

  const renderNPKChart = () => {
    if (!recommendation) return null;

    const chartData = [
      { 
        nutrient: 'Nitrogen (N)', 
        current: soilData.N, 
        required: recommendation.npk_plan.nitrogen_kg_per_ha 
      },
      { 
        nutrient: 'Phosphorus (P)', 
        current: soilData.P, 
        required: recommendation.npk_plan.phosphorus_kg_per_ha 
      },
      { 
        nutrient: 'Potassium (K)', 
        current: soilData.K, 
        required: recommendation.npk_plan.potassium_kg_per_ha 
      }
    ];

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="nutrient" />
          <YAxis label={{ value: 'kg/hectare', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="current" fill="#8884d8" name="Current Soil Level" />
          <Bar dataKey="required" fill="#82ca9d" name="Application Needed" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderSoilRadar = () => {
    const radarData = [
      { factor: 'Nitrogen', value: (soilData.N / 200) * 100 },
      { factor: 'Phosphorus', value: (soilData.P / 200) * 100 },
      { factor: 'Potassium', value: (soilData.K / 300) * 100 },
      { factor: 'Moisture', value: soilData.soil_moisture },
      { factor: 'NDVI', value: ((soilData.NDVI + 1) / 2) * 100 }
    ];

    return (
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={radarData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="factor" />
          <PolarRadiusAxis angle={90} domain={[0, 100]} />
          <Radar name="Field Health" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>
    );
  };

  const handleInputChange = (field, value) => {
    setSoilData(prev => ({
      ...prev,
      [field]: field === 'crop_type' ? value : parseFloat(value) || 0
    }));
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <h1>🌾 AgriMinds - Precision Agriculture</h1>
        <p style={styles.subtitle}>Fertilizer & Irrigation Recommendations with Satellite NDVI</p>
      </header>

      {/* Error Banner */}
      {error && (
        <div style={styles.errorBanner}>
          ❌ Error: {error}
        </div>
      )}

      {/* Main Grid */}
      <div style={styles.mainGrid}>
        {/* LEFT PANEL - MAP & VISUALIZATION */}
        <div style={styles.mapPanel}>
          {/* Map */}
          <div style={styles.card}>
            <h3>📍 Field Mapper</h3>
            <p style={styles.instructions}>
              Click on the map to draw your field boundaries (4 points minimum). 
              NDVI will be automatically fetched from satellite imagery.
            </p>
            <MapContainer
              center={[20.5937, 78.9629]}
              zoom={5}
              style={styles.map}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <FieldDrawer onPolygonComplete={setFieldBounds} />
              {fieldBounds && <NDVIHeatmapLayer fieldBounds={fieldBounds} />}
            </MapContainer>

            {/* Legend */}
            <div style={styles.legend}>
              <strong>NDVI Color Legend:</strong>
              <div style={styles.legendItems}>
                <span style={{...styles.legendItem, backgroundColor: '#ff0000'}}>0.0-0.2: Stressed</span>
                <span style={{...styles.legendItem, backgroundColor: '#ff9900'}}>0.2-0.4: Low</span>
                <span style={{...styles.legendItem, backgroundColor: '#ffff00'}}>0.4-0.6: Moderate</span>
                <span style={{...styles.legendItem, backgroundColor: '#00ff00'}}>0.6-1.0: Healthy</span>
              </div>
            </div>
          </div>

          {/* Soil Health Radar */}
          <div style={styles.card}>
            <h3>📊 Field Health Overview</h3>
            {renderSoilRadar()}
          </div>
        </div>

        {/* RIGHT PANEL - INPUTS & RESULTS */}
        <div style={styles.controlPanel}>
          {/* Input Form */}
          <div style={styles.card}>
            <h3>🌱 Crop & Soil Data</h3>
            
            {/* Crop Selection */}
            <div style={styles.inputGroup}>
              <label><strong>Select Crop Type:</strong></label>
              <select
                value={soilData.crop_type}
                onChange={(e) => handleInputChange('crop_type', e.target.value)}
                style={styles.select}
              >
                {availableCrops.map(crop => (
                  <option key={crop} value={crop}>
                    {crop.charAt(0).toUpperCase() + crop.slice(1).replace('_', ' ')}
                  </option>
                ))}
              </select>
            </div>

            <div style={styles.formGrid}>
              {/* Soil NPK */}
              <div style={styles.inputGroup}>
                <label>Nitrogen (N) - kg/ha:</label>
                <input
                  type="number"
                  value={soilData.N}
                  onChange={(e) => handleInputChange('N', e.target.value)}
                  style={styles.input}
                  min="0"
                  max="300"
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Phosphorus (P) - kg/ha:</label>
                <input
                  type="number"
                  value={soilData.P}
                  onChange={(e) => handleInputChange('P', e.target.value)}
                  style={styles.input}
                  min="0"
                  max="200"
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Potassium (K) - kg/ha:</label>
                <input
                  type="number"
                  value={soilData.K}
                  onChange={(e) => handleInputChange('K', e.target.value)}
                  style={styles.input}
                  min="0"
                  max="400"
                />
              </div>

              {/* Weather */}
              <div style={styles.inputGroup}>
                <label>Temperature (°C):</label>
                <input
                  type="number"
                  value={soilData.temperature}
                  onChange={(e) => handleInputChange('temperature', e.target.value)}
                  style={styles.input}
                  min="-10"
                  max="50"
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Humidity (%):</label>
                <input
                  type="number"
                  value={soilData.humidity}
                  onChange={(e) => handleInputChange('humidity', e.target.value)}
                  style={styles.input}
                  min="0"
                  max="100"
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Soil Moisture (%):</label>
                <input
                  type="number"
                  value={soilData.soil_moisture}
                  onChange={(e) => handleInputChange('soil_moisture', e.target.value)}
                  style={styles.input}
                  min="0"
                  max="100"
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Rainfall (mm):</label>
                <input
                  type="number"
                  value={soilData.rainfall}
                  onChange={(e) => handleInputChange('rainfall', e.target.value)}
                  style={styles.input}
                  min="0"
                  max="500"
                />
              </div>

              <div style={styles.inputGroup}>
                <label>NDVI:</label>
                <input
                  type="number"
                  value={soilData.NDVI}
                  onChange={(e) => handleInputChange('NDVI', e.target.value)}
                  style={styles.input}
                  min="-1"
                  max="1"
                  step="0.01"
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Forecast Rain (mm):</label>
                <input
                  type="number"
                  value={soilData.forecast_rain}
                  onChange={(e) => handleInputChange('forecast_rain', e.target.value)}
                  style={styles.input}
                  min="0"
                  max="100"
                />
              </div>
            </div>

            {/* Action Button */}
            <button
              onClick={getRecommendation}
              disabled={loading}
              style={{
                ...styles.primaryButton,
                opacity: loading ? 0.6 : 1,
                cursor: loading ? 'not-allowed' : 'pointer'
              }}
            >
              {loading ? '⏳ Generating...' : '🎯 Get AI Recommendation'}
            </button>
          </div>

          {/* Results */}
          {recommendation && (
            <div style={styles.card}>
              <h3>📋 Recommendations for {recommendation.crop_type.toUpperCase()}</h3>

              {/* Alerts */}
              {recommendation.alerts && recommendation.alerts.length > 0 && (
                <div style={styles.alertsContainer}>
                  {recommendation.alerts.map((alert, idx) => (
                    <div key={idx} style={styles.alert}>
                      {alert}
                    </div>
                  ))}
                </div>
              )}

              {/* Fertilizer Plan */}
              <div style={styles.npkPlan}>
                <h4>⚗️ Fertilizer Application Plan</h4>
                <div style={styles.npkGrid}>
                  <div style={styles.npkItem}>
                    <span className="label">Nitrogen (N)</span>
                    <span className="value">{recommendation.npk_plan.nitrogen_kg_per_ha} kg/ha</span>
                  </div>
                  <div style={styles.npkItem}>
                    <span className="label">Phosphorus (P)</span>
                    <span className="value">{recommendation.npk_plan.phosphorus_kg_per_ha} kg/ha</span>
                  </div>
                  <div style={styles.npkItem}>
                    <span className="label">Potassium (K)</span>
                    <span className="value">{recommendation.npk_plan.potassium_kg_per_ha} kg/ha</span>
                  </div>
                </div>

                <div style={styles.totalFertilizer}>
                  <strong>Total Fertilizer:</strong> {recommendation.npk_plan.total_fertilizer_kg} kg
                </div>

                <p style={styles.timing}>⏰ <strong>Timing:</strong> {recommendation.application_timing}</p>
              </div>

              {/* Irrigation Requirements */}
              <div style={styles.irrigationPlan}>
                <h4>💧 Irrigation Requirements</h4>
                <div style={styles.irrigationGrid}>
                  <div style={styles.irrigationItem}>
                    <span className="label">Water Needed</span>
                    <span className="value">{recommendation.irrigation_requirements.water_needed_mm} mm</span>
                  </div>
                  <div style={styles.irrigationItem}>
                    <span className="label">Frequency</span>
                    <span className="value">Every {recommendation.irrigation_requirements.irrigation_frequency_days} days</span>
                  </div>
                  <div style={styles.irrigationItem}>
                    <span className="label">Method</span>
                    <span className="value">{recommendation.irrigation_requirements.irrigation_method}</span>
                  </div>
                  <div style={styles.irrigationItem}>
                    <span className="label">Water Stress</span>
                    <span className="value">{recommendation.irrigation_requirements.water_stress_level}</span>
                  </div>
                </div>

                <div style={styles.irrigationRecommendations}>
                  <strong>Recommendations:</strong>
                  <ul>
                    {recommendation.irrigation_requirements.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* NPK CHART */}
              <div style={styles.chartSection}>
                <h3>📊 Current vs Required:</h3>
                {renderNPKChart()}
              </div>

              {/* ECONOMIC & ENVIRONMENTAL */}
              <div style={styles.metricsGrid}>
                <div style={styles.metricCard}>
                  <h4>🌿 Green Score</h4>
                  <div style={styles.scoreCircle}>
                    {recommendation.green_score.toFixed(1)}
                  </div>
                  <p>Environmental Impact</p>
                </div>

                <div style={styles.metricCard}>
                  <h4>💰 Estimated ROI</h4>
                  <div style={styles.roiValue}>
                    {recommendation.estimated_roi.toFixed(2)}x
                  </div>
                  <p>Return on Investment</p>
                </div>
              </div>

              {/* SATELLITE INSIGHTS */}
              <div style={styles.satelliteInsights}>
                <h4>🛰️ Satellite Insights:</h4>
                <p>NDVI: {recommendation.satellite_insights.ndvi.toFixed(3)} ({recommendation.satellite_insights.stress_level})</p>
                <p>Predicted Yield: {recommendation.satellite_insights.predicted_yield_kg_per_ha} kg/ha</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// STYLES
// ============================================================================

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f5f7fa',
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  },
  header: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    padding: '2rem',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: '1.2rem',
    marginTop: '0.5rem',
    opacity: 0.9,
  },
  errorBanner: {
    background: '#fee2e2',
    color: '#991b1b',
    padding: '1rem',
    margin: '1rem',
    borderRadius: '8px',
    border: '2px solid #ef4444',
    fontWeight: 'bold'
  },
  mainGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '2rem',
    padding: '2rem',
    maxWidth: '1800px',
    margin: '0 auto',
  },
  mapPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  controlPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  card: {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '1.5rem',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
  },
  instructions: {
    color: '#666',
    fontSize: '0.9rem',
    marginBottom: '1rem',
  },
  map: {
    height: '500px',
    borderRadius: '8px',
    border: '2px solid #e0e0e0',
  },
  legend: {
    marginTop: '1rem',
    padding: '1rem',
    backgroundColor: '#f9f9f9',
    borderRadius: '8px',
  },
  legendItems: {
    display: 'flex',
    gap: '1rem',
    flexWrap: 'wrap',
    marginTop: '0.5rem',
  },
  legendItem: {
    padding: '0.5rem 1rem',
    borderRadius: '4px',
    color: '#333',
    fontWeight: '500',
    fontSize: '0.85rem',
  },
  formGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1rem',
    marginBottom: '1.5rem',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  input: {
    padding: '0.75rem',
    border: '2px solid #e0e0e0',
    borderRadius: '6px',
    fontSize: '1rem',
    transition: 'border-color 0.3s',
  },
  select: {
    padding: '0.75rem',
    border: '2px solid #e0e0e0',
    borderRadius: '6px',
    fontSize: '1rem',
    backgroundColor: 'white',
    cursor: 'pointer',
    marginBottom: '1rem',
  },
  primaryButton: {
    width: '100%',
    padding: '1rem',
    backgroundColor: '#667eea',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  alertsContainer: {
    marginBottom: '1.5rem',
  },
  alert: {
    padding: '1rem',
    marginBottom: '0.75rem',
    backgroundColor: '#fff3cd',
    border: '1px solid #ffc107',
    borderRadius: '6px',
    fontSize: '0.95rem',
  },
  npkPlan: {
    marginBottom: '1.5rem',
  },
  npkGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '1rem',
    margin: '1rem 0',
  },
  npkItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '1rem',
    backgroundColor: '#e8f5e9',
    borderRadius: '8px',
  },
  totalFertilizer: {
    padding: '1rem',
    backgroundColor: '#e3f2fd',
    borderRadius: '6px',
    textAlign: 'center',
    marginTop: '1rem',
    fontSize: '1.1rem',
  },
  timing: {
    padding: '1rem',
    backgroundColor: '#fff9e6',
    borderRadius: '6px',
    marginTop: '1rem',
  },
  irrigationPlan: {
    marginBottom: '1.5rem',
    padding: '1rem',
    backgroundColor: '#e0f2f7',
    borderRadius: '8px',
  },
  irrigationGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1rem',
    margin: '1rem 0',
  },
  irrigationItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '1rem',
    backgroundColor: 'white',
    borderRadius: '8px',
  },
  irrigationRecommendations: {
    marginTop: '1rem',
    padding: '1rem',
    backgroundColor: 'white',
    borderRadius: '6px',
  },
  chartSection: {
    marginTop: '1.5rem',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1rem',
    marginTop: '1.5rem',
  },
  metricCard: {
    padding: '1rem',
    backgroundColor: '#f9f9f9',
    borderRadius: '8px',
    textAlign: 'center',
  },
  scoreCircle: {
    width: '80px',
    height: '80px',
    borderRadius: '50%',
    backgroundColor: '#4CAF50',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '2rem',
    fontWeight: 'bold',
    margin: '1rem auto',
  },
  roiValue: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: '#4CAF50',
    margin: '1rem 0',
  },
  satelliteInsights: {
    marginTop: '1.5rem',
    padding: '1rem',
    backgroundColor: '#e3f2fd',
    borderRadius: '8px',
  },
};

export default Dashboard;
