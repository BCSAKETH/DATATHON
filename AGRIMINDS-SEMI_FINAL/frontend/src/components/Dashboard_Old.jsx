import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polygon, Popup, useMapEvents } from 'react-leaflet';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import 'leaflet/dist/leaflet.css';

/**
 * AgriMinds Dashboard
 * ====================
 * Main interface for satellite-enhanced precision agriculture.
 * 
 * Features:
 * 1. Interactive map with NDVI overlay
 * 2. Field polygon drawing
 * 3. Soil parameter input
 * 4. Real-time fertilizer recommendations
 * 5. Economic and environmental impact visualization
 */

const API_BASE_URL = 'http://localhost:8000';

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

  useEffect(() => {
    if (fieldBounds && fieldBounds.length > 0) {
      // Calculate field center
      const lats = fieldBounds.map(p => p[0]);
      const lngs = fieldBounds.map(p => p[1]);
      const centerLat = lats.reduce((a, b) => a + b) / lats.length;
      const centerLng = lngs.reduce((a, b) => a + b) / lngs.length;

      // Fetch NDVI for field center
      fetch(`${API_BASE_URL}/satellite/ndvi?lat=${centerLat}&lon=${centerLng}`)
        .then(res => res.json())
        .then(data => setNdviData(data))
        .catch(err => console.error('NDVI fetch error:', err));
    }
  }, [fieldBounds]);

  if (!fieldBounds || !ndviData) return null;

  // Color based on NDVI value
  const getColor = (ndvi) => {
    if (ndvi > 0.6) return '#00ff00'; // Healthy - Green
    if (ndvi > 0.4) return '#ffff00'; // Moderate - Yellow
    if (ndvi > 0.2) return '#ff9900'; // Stressed - Orange
    return '#ff0000'; // Severe - Red
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
    N: 50,
    P: 30,
    K: 80,
    temperature: 25,
    humidity: 65,
    ph: 6.5,
    rainfall: 150,
    NDVI: 0.5,
    soil_moisture: 50,
    growth_stage: 'vegetative',
    forecast_rain: 0
  });
  
  const [cropPrediction, setCropPrediction] = useState(null);
  const [fertilizerRec, setFertilizerRec] = useState(null);
  const [loading, setLoading] = useState(false);

  // ============================================================================
  // API CALLS
  // ============================================================================

  const predictCrop = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict/crop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          N: soilData.N,
          P: soilData.P,
          K: soilData.K,
          temperature: soilData.temperature,
          humidity: soilData.humidity,
          ph: soilData.ph,
          rainfall: soilData.rainfall
        })
      });
      
      const data = await response.json();
      setCropPrediction(data);
    } catch (error) {
      console.error('Crop prediction error:', error);
      alert('Failed to get crop prediction. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getFertilizerRecommendation = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict/fertilizer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(soilData)
      });
      
      const data = await response.json();
      setFertilizerRec(data);
    } catch (error) {
      console.error('Fertilizer recommendation error:', error);
      alert('Failed to get fertilizer recommendation. Ensure backend is running.');
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

      fetch(`${API_BASE_URL}/satellite/ndvi?lat=${centerLat}&lon=${centerLng}`)
        .then(res => res.json())
        .then(data => setSoilData(prev => ({ ...prev, NDVI: data.ndvi })))
        .catch(err => console.error('Auto NDVI fetch error:', err));
    }
  }, [fieldBounds]);

  // ============================================================================
  // RENDER HELPER COMPONENTS
  // ============================================================================

  const renderNPKChart = () => {
    if (!fertilizerRec) return null;

    const chartData = [
      { nutrient: 'Nitrogen (N)', current: soilData.N, recommended: fertilizerRec.npk_plan.N },
      { nutrient: 'Phosphorus (P)', current: soilData.P, recommended: fertilizerRec.npk_plan.P },
      { nutrient: 'Potassium (K)', current: soilData.K, recommended: fertilizerRec.npk_plan.K }
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
          <Bar dataKey="recommended" fill="#82ca9d" name="Recommended Application" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderSoilRadar = () => {
    const radarData = [
      { factor: 'Nitrogen', value: (soilData.N / 200) * 100 },
      { factor: 'Phosphorus', value: (soilData.P / 200) * 100 },
      { factor: 'Potassium', value: (soilData.K / 300) * 100 },
      { factor: 'pH', value: ((soilData.ph - 3.5) / 6.5) * 100 },
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

  // ============================================================================
  // MAIN RENDER
  // ============================================================================

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1>🌾 AgriMinds - Satellite-Enhanced Precision Agriculture</h1>
        <p style={styles.subtitle}>Sense • Analyze • Act</p>
      </header>

      <div style={styles.mainGrid}>
        {/* LEFT PANEL: MAP */}
        <div style={styles.mapPanel}>
          <div style={styles.card}>
            <h2>🗺️ Field Mapping & NDVI Analysis</h2>
            <p style={styles.instructions}>
              Click on the map to draw your field boundary (4+ points auto-complete)
            </p>
            
            <MapContainer 
              center={[17.385, 78.486]} // Hyderabad coordinates
              zoom={13} 
              style={styles.map}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              />
              <FieldDrawer onPolygonComplete={setFieldBounds} />
              <NDVIHeatmapLayer fieldBounds={fieldBounds} />
            </MapContainer>

            {fieldBounds && (
              <div style={styles.legend}>
                <h4>NDVI Legend:</h4>
                <div style={styles.legendItems}>
                  <span style={{ ...styles.legendItem, background: '#00ff00' }}>Healthy (&gt;0.6)</span>
                  <span style={{ ...styles.legendItem, background: '#ffff00' }}>Moderate (0.4-0.6)</span>
                  <span style={{ ...styles.legendItem, background: '#ff9900' }}>Stressed (0.2-0.4)</span>
                  <span style={{ ...styles.legendItem, background: '#ff0000' }}>Severe (&lt;0.2)</span>
                </div>
              </div>
            )}
          </div>

          {/* SOIL RADAR CHART */}
          <div style={styles.card}>
            <h3>📊 Field Health Profile</h3>
            {renderSoilRadar()}
          </div>
        </div>

        {/* RIGHT PANEL: INPUTS & RECOMMENDATIONS */}
        <div style={styles.controlPanel}>
          {/* SOIL INPUT FORM */}
          <div style={styles.card}>
            <h2>🌱 Ground Truth: Soil Parameters</h2>
            
            <div style={styles.formGrid}>
              <div style={styles.inputGroup}>
                <label>Nitrogen (N) - mg/kg</label>
                <input 
                  type="number" 
                  value={soilData.N} 
                  onChange={(e) => setSoilData({...soilData, N: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Phosphorus (P) - mg/kg</label>
                <input 
                  type="number" 
                  value={soilData.P} 
                  onChange={(e) => setSoilData({...soilData, P: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Potassium (K) - mg/kg</label>
                <input 
                  type="number" 
                  value={soilData.K} 
                  onChange={(e) => setSoilData({...soilData, K: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Temperature - °C</label>
                <input 
                  type="number" 
                  value={soilData.temperature} 
                  onChange={(e) => setSoilData({...soilData, temperature: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Humidity - %</label>
                <input 
                  type="number" 
                  value={soilData.humidity} 
                  onChange={(e) => setSoilData({...soilData, humidity: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Soil pH</label>
                <input 
                  type="number" 
                  step="0.1"
                  value={soilData.ph} 
                  onChange={(e) => setSoilData({...soilData, ph: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Rainfall - mm</label>
                <input 
                  type="number" 
                  value={soilData.rainfall} 
                  onChange={(e) => setSoilData({...soilData, rainfall: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Soil Moisture - %</label>
                <input 
                  type="number" 
                  value={soilData.soil_moisture} 
                  onChange={(e) => setSoilData({...soilData, soil_moisture: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label>Growth Stage</label>
                <select 
                  value={soilData.growth_stage}
                  onChange={(e) => setSoilData({...soilData, growth_stage: e.target.value})}
                  style={styles.input}
                >
                  <option value="seedling">Seedling</option>
                  <option value="vegetative">Vegetative</option>
                  <option value="flowering">Flowering</option>
                  <option value="fruiting">Fruiting</option>
                  <option value="maturity">Maturity</option>
                </select>
              </div>

              <div style={styles.inputGroup}>
                <label>Forecast Rain (48h) - mm</label>
                <input 
                  type="number" 
                  value={soilData.forecast_rain} 
                  onChange={(e) => setSoilData({...soilData, forecast_rain: parseFloat(e.target.value)})}
                  style={styles.input}
                />
              </div>
            </div>

            <div style={styles.buttonGroup}>
              <button 
                onClick={predictCrop} 
                disabled={loading}
                style={styles.primaryButton}
              >
                {loading ? '⏳ Analyzing...' : '🌾 Recommend Crop'}
              </button>

              <button 
                onClick={getFertilizerRecommendation} 
                disabled={loading}
                style={styles.secondaryButton}
              >
                {loading ? '⏳ Analyzing...' : '💊 Get Fertilizer Plan'}
              </button>
            </div>
          </div>

          {/* CROP PREDICTION CARD */}
          {cropPrediction && (
            <div style={styles.card}>
              <h2>🌾 Crop Recommendation</h2>
              <div style={styles.resultCard}>
                <h3 style={styles.recommendedCrop}>{cropPrediction.recommended_crop}</h3>
                <p style={styles.confidence}>Confidence: {(cropPrediction.confidence * 100).toFixed(1)}%</p>
                
                <h4>Alternative Options:</h4>
                <ul style={styles.alternativeList}>
                  {cropPrediction.alternative_crops.map((alt, idx) => (
                    <li key={idx}>
                      {alt.crop} - {(alt.confidence * 100).toFixed(1)}%
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* FERTILIZER RECOMMENDATION CARD */}
          {fertilizerRec && (
            <div style={styles.card}>
              <h2>💊 Fertilizer Recommendation</h2>
              
              {/* ALERTS */}
              {fertilizerRec.alerts.length > 0 && (
                <div style={styles.alertsContainer}>
                  {fertilizerRec.alerts.map((alert, idx) => (
                    <div key={idx} style={styles.alert}>
                      {alert}
                    </div>
                  ))}
                </div>
              )}

              {/* NPK PLAN */}
              <div style={styles.npkPlan}>
                <h3>📋 Application Plan:</h3>
                <div style={styles.npkGrid}>
                  <div style={styles.npkItem}>
                    <span className="label">Nitrogen (N)</span>
                    <span className="value">{fertilizerRec.npk_plan.N} kg/ha</span>
                  </div>
                  <div style={styles.npkItem}>
                    <span className="label">Phosphorus (P)</span>
                    <span className="value">{fertilizerRec.npk_plan.P} kg/ha</span>
                  </div>
                  <div style={styles.npkItem}>
                    <span className="label">Potassium (K)</span>
                    <span className="value">{fertilizerRec.npk_plan.K} kg/ha</span>
                  </div>
                </div>

                <p style={styles.timing}>⏰ <strong>Timing:</strong> {fertilizerRec.application_timing}</p>
              </div>

              {/* NPK CHART */}
              <div style={styles.chartSection}>
                <h3>📊 Current vs Recommended:</h3>
                {renderNPKChart()}
              </div>

              {/* ECONOMIC & ENVIRONMENTAL */}
              <div style={styles.metricsGrid}>
                <div style={styles.metricCard}>
                  <h4>🌿 Green Score</h4>
                  <div style={styles.scoreCircle}>
                    {fertilizerRec.green_score.toFixed(1)}
                  </div>
                  <p>Environmental Impact</p>
                </div>

                <div style={styles.metricCard}>
                  <h4>💰 Estimated ROI</h4>
                  <p>Cost: ₹{fertilizerRec.estimated_roi.cost_inr}</p>
                  <p>Yield Gain: {fertilizerRec.estimated_roi.yield_increase_kg} kg/ha</p>
                  <p style={{ color: '#4CAF50', fontWeight: 'bold' }}>
                    Net Profit: ₹{fertilizerRec.estimated_roi.net_profit_inr}
                  </p>
                </div>
              </div>

              {/* SATELLITE INSIGHTS */}
              <div style={styles.satelliteInsights}>
                <h4>🛰️ Satellite Insights:</h4>
                <p>NDVI: {fertilizerRec.satellite_insights.ndvi.toFixed(3)} ({fertilizerRec.satellite_insights.stress_level})</p>
                <p>Predicted Yield: {fertilizerRec.satellite_insights.predicted_yield_kg_per_ha} kg/ha</p>
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
  buttonGroup: {
    display: 'flex',
    gap: '1rem',
  },
  primaryButton: {
    flex: 1,
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
  secondaryButton: {
    flex: 1,
    padding: '1rem',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  resultCard: {
    padding: '1rem',
    backgroundColor: '#f0f4ff',
    borderRadius: '8px',
  },
  recommendedCrop: {
    fontSize: '2rem',
    color: '#667eea',
    marginBottom: '0.5rem',
  },
  confidence: {
    fontSize: '1.1rem',
    color: '#4CAF50',
    fontWeight: '600',
  },
  alternativeList: {
    listStyle: 'none',
    padding: 0,
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
  timing: {
    padding: '1rem',
    backgroundColor: '#fff9e6',
    borderRadius: '6px',
    marginTop: '1rem',
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
  satelliteInsights: {
    marginTop: '1.5rem',
    padding: '1rem',
    backgroundColor: '#e3f2fd',
    borderRadius: '8px',
  },
};

export default Dashboard;
