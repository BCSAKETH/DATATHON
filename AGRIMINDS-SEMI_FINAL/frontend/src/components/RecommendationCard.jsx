import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

/**
 * RecommendationCard Component
 * Displays fertilizer recommendations with:
 * 1. NPK breakdown
 * 2. Visual charts
 * 3. Alerts and warnings
 * 4. Economic metrics
 * 5. Green score
 */
const RecommendationCard = ({ 
  recommendations,
  alerts = [],
  greenScore,
  estimatedCost,
  canApply = true,
  ndviStatus,
  growthStage,
  roi,
  onApplyRecommendation
}) => {
  if (!recommendations) {
    return (
      <div style={styles.card}>
        <div style={styles.emptyState}>
          <span style={{ fontSize: '3em' }}>🌱</span>
          <h3>No Recommendations Yet</h3>
          <p>Enter soil data and field information to get AI-powered fertilizer recommendations</p>
        </div>
      </div>
    );
  }

  // Prepare data for charts
  const npkData = [
    { nutrient: 'Nitrogen (N)', amount: recommendations.N || 0, color: '#3b82f6' },
    { nutrient: 'Phosphorus (P)', amount: recommendations.P || 0, color: '#10b981' },
    { nutrient: 'Potassium (K)', amount: recommendations.K || 0, color: '#f59e0b' }
  ];

  const radarData = [
    { subject: 'N', value: Math.min(recommendations.N || 0, 100), fullMark: 100 },
    { subject: 'P', value: Math.min(recommendations.P || 0, 100), fullMark: 100 },
    { subject: 'K', value: Math.min(recommendations.K || 0, 100), fullMark: 100 }
  ];

  const totalNPK = (recommendations.N || 0) + (recommendations.P || 0) + (recommendations.K || 0);

  // Green score color
  const getGreenScoreColor = (score) => {
    if (score >= 80) return '#10b981';
    if (score >= 60) return '#f59e0b';
    return '#ef4444';
  };

  // Alert severity styling
  const getAlertStyle = (alert) => {
    if (alert.includes('🔴') || alert.includes('⛔')) {
      return { ...styles.alert, background: '#fee2e2', borderLeft: '4px solid #ef4444' };
    } else if (alert.includes('🟡') || alert.includes('⚠️')) {
      return { ...styles.alert, background: '#fef3c7', borderLeft: '4px solid #f59e0b' };
    } else if (alert.includes('ℹ️')) {
      return { ...styles.alert, background: '#dbeafe', borderLeft: '4px solid #3b82f6' };
    }
    return styles.alert;
  };

  return (
    <div style={styles.card}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <h2 style={styles.title}>🎯 AI Recommendation</h2>
          <p style={styles.subtitle}>Satellite-Enhanced Precision Agriculture</p>
        </div>
        <div style={{
          ...styles.badge,
          background: canApply ? '#d1fae5' : '#fee2e2',
          color: canApply ? '#065f46' : '#991b1b'
        }}>
          {canApply ? '✓ Ready to Apply' : '⛔ Application Blocked'}
        </div>
      </div>

      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div style={styles.alertsContainer}>
          {alerts.map((alert, idx) => (
            <div key={idx} style={getAlertStyle(alert)}>
              {alert}
            </div>
          ))}
        </div>
      )}

      {/* Main Content Grid */}
      <div style={styles.grid}>
        {/* Left Column: NPK Recommendations */}
        <div style={styles.column}>
          <h3 style={styles.sectionTitle}>Fertilizer Application Plan</h3>
          
          {/* NPK Cards */}
          <div style={styles.npkGrid}>
            <div style={{ ...styles.npkCard, borderTop: '4px solid #3b82f6' }}>
              <div style={styles.npkLabel}>Nitrogen (N)</div>
              <div style={styles.npkValue}>{recommendations.N?.toFixed(1) || 0} kg</div>
              <div style={styles.npkNote}>Apply as Urea (46% N)</div>
            </div>
            
            <div style={{ ...styles.npkCard, borderTop: '4px solid #10b981' }}>
              <div style={styles.npkLabel}>Phosphorus (P)</div>
              <div style={styles.npkValue}>{recommendations.P?.toFixed(1) || 0} kg</div>
              <div style={styles.npkNote}>Apply as DAP (46% P)</div>
            </div>
            
            <div style={{ ...styles.npkCard, borderTop: '4px solid #f59e0b' }}>
              <div style={styles.npkLabel}>Potassium (K)</div>
              <div style={styles.npkValue}>{recommendations.K?.toFixed(1) || 0} kg</div>
              <div style={styles.npkNote}>Apply as MOP (60% K)</div>
            </div>
          </div>

          {/* Bar Chart */}
          <div style={{ marginTop: '20px' }}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={npkData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="nutrient" tick={{ fontSize: 12 }} />
                <YAxis label={{ value: 'kg', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="amount" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Column: Metrics */}
        <div style={styles.column}>
          <h3 style={styles.sectionTitle}>Impact Analysis</h3>
          
          {/* Green Score */}
          <div style={styles.metricCard}>
            <div style={styles.metricLabel}>🌿 Environmental Score</div>
            <div style={styles.metricValue}>
              <span style={{ 
                fontSize: '2.5em', 
                fontWeight: 'bold',
                color: getGreenScoreColor(greenScore)
              }}>
                {greenScore || 0}
              </span>
              <span style={{ fontSize: '1.2em', color: '#666' }}>/100</span>
            </div>
            <div style={styles.progressBar}>
              <div style={{
                ...styles.progressFill,
                width: `${greenScore}%`,
                background: getGreenScoreColor(greenScore)
              }} />
            </div>
          </div>

          {/* Economic Metrics */}
          <div style={styles.metricsGrid}>
            <div style={styles.miniMetric}>
              <div style={styles.miniMetricLabel}>💰 Estimated Cost</div>
              <div style={styles.miniMetricValue}>₹{estimatedCost?.toFixed(2) || 0}</div>
            </div>
            
            <div style={styles.miniMetric}>
              <div style={styles.miniMetricLabel}>📊 Total NPK</div>
              <div style={styles.miniMetricValue}>{totalNPK.toFixed(1)} kg</div>
            </div>

            {roi && (
              <>
                <div style={styles.miniMetric}>
                  <div style={styles.miniMetricLabel}>📈 Expected ROI</div>
                  <div style={styles.miniMetricValue}>{roi.roi_percentage}%</div>
                </div>
                
                <div style={styles.miniMetric}>
                  <div style={styles.miniMetricLabel}>💵 Net Profit</div>
                  <div style={styles.miniMetricValue}>₹{roi.net_profit?.toFixed(2) || 0}</div>
                </div>
              </>
            )}
          </div>

          {/* Radar Chart */}
          <div style={{ marginTop: '20px' }}>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="NPK Balance" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Context Information */}
      <div style={styles.contextBar}>
        <div style={styles.contextItem}>
          <span style={styles.contextLabel}>📡 Satellite NDVI:</span>
          <span style={styles.contextValue}>{ndviStatus || 'N/A'}</span>
        </div>
        <div style={styles.contextItem}>
          <span style={styles.contextLabel}>🌱 Growth Stage:</span>
          <span style={styles.contextValue}>{growthStage || 'N/A'}</span>
        </div>
        <div style={styles.contextItem}>
          <span style={styles.contextLabel}>📅 Generated:</span>
          <span style={styles.contextValue}>{new Date().toLocaleDateString()}</span>
        </div>
      </div>

      {/* Action Button */}
      {canApply && onApplyRecommendation && (
        <button 
          style={styles.applyButton}
          onClick={() => onApplyRecommendation(recommendations)}
        >
          ✓ Accept & Generate Application Plan
        </button>
      )}

      {/* Disclaimer */}
      <div style={styles.disclaimer}>
        <strong>⚠️ Important:</strong> These are AI-generated recommendations based on satellite data and soil analysis. 
        Always consult with local agricultural experts and conduct soil tests before application. 
        Weather conditions may affect optimal timing.
      </div>
    </div>
  );
};

// Styles
const styles = {
  card: {
    background: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    padding: '24px',
    marginTop: '20px'
  },
  emptyState: {
    textAlign: 'center',
    padding: '60px 20px',
    color: '#666'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '20px',
    paddingBottom: '20px',
    borderBottom: '2px solid #f0f0f0'
  },
  title: {
    margin: 0,
    fontSize: '1.8em',
    color: '#1f2937'
  },
  subtitle: {
    margin: '5px 0 0 0',
    color: '#6b7280',
    fontSize: '0.9em'
  },
  badge: {
    padding: '8px 16px',
    borderRadius: '20px',
    fontWeight: '600',
    fontSize: '0.9em'
  },
  alertsContainer: {
    marginBottom: '20px'
  },
  alert: {
    padding: '12px 16px',
    marginBottom: '10px',
    borderRadius: '8px',
    fontSize: '0.95em',
    lineHeight: '1.5'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '30px',
    marginBottom: '20px'
  },
  column: {
    display: 'flex',
    flexDirection: 'column'
  },
  sectionTitle: {
    fontSize: '1.2em',
    marginBottom: '15px',
    color: '#374151'
  },
  npkGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '15px',
    marginBottom: '20px'
  },
  npkCard: {
    background: '#f9fafb',
    padding: '15px',
    borderRadius: '8px',
    textAlign: 'center'
  },
  npkLabel: {
    fontSize: '0.85em',
    color: '#6b7280',
    marginBottom: '8px'
  },
  npkValue: {
    fontSize: '1.8em',
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: '8px'
  },
  npkNote: {
    fontSize: '0.75em',
    color: '#9ca3af'
  },
  metricCard: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    padding: '20px',
    borderRadius: '12px',
    marginBottom: '20px'
  },
  metricLabel: {
    fontSize: '0.9em',
    marginBottom: '10px',
    opacity: 0.9
  },
  metricValue: {
    marginBottom: '10px'
  },
  progressBar: {
    width: '100%',
    height: '8px',
    background: 'rgba(255,255,255,0.3)',
    borderRadius: '4px',
    overflow: 'hidden'
  },
  progressFill: {
    height: '100%',
    transition: 'width 0.5s ease'
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '10px',
    marginBottom: '20px'
  },
  miniMetric: {
    background: '#f9fafb',
    padding: '15px',
    borderRadius: '8px'
  },
  miniMetricLabel: {
    fontSize: '0.8em',
    color: '#6b7280',
    marginBottom: '5px'
  },
  miniMetricValue: {
    fontSize: '1.4em',
    fontWeight: 'bold',
    color: '#1f2937'
  },
  contextBar: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '15px',
    background: '#f3f4f6',
    borderRadius: '8px',
    marginBottom: '20px'
  },
  contextItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  contextLabel: {
    fontSize: '0.85em',
    color: '#6b7280'
  },
  contextValue: {
    fontSize: '0.85em',
    fontWeight: '600',
    color: '#1f2937'
  },
  applyButton: {
    width: '100%',
    padding: '15px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1.1em',
    fontWeight: '600',
    cursor: 'pointer',
    marginBottom: '15px',
    transition: 'transform 0.2s'
  },
  disclaimer: {
    padding: '12px',
    background: '#fef3c7',
    border: '1px solid #fbbf24',
    borderRadius: '6px',
    fontSize: '0.85em',
    lineHeight: '1.5',
    color: '#78350f'
  }
};

export default RecommendationCard;
