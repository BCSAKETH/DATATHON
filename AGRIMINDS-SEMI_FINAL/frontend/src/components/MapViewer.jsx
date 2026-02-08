import React, { useState, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, FeatureGroup, Rectangle, useMap } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';

/**
 * NDVIOverlay Component
 * Renders a color-coded heatmap overlay on the map
 * 
 * NDVI Color Mapping:
 * - Red (0.0-0.3): Bare soil, stressed crops
 * - Yellow (0.3-0.5): Moderate vegetation
 * - Green (0.5-1.0): Healthy, dense vegetation
 */
const NDVIOverlay = ({ bounds, ndviData, opacity = 0.6 }) => {
  const canvasRef = useRef(null);
  const map = useMap();

  useEffect(() => {
    if (!ndviData || !bounds) return;

    // Create canvas overlay
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Draw NDVI heatmap
    const imageData = ctx.createImageData(256, 256);
    const data = imageData.data;

    for (let i = 0; i < ndviData.length; i++) {
      for (let j = 0; j < ndviData[i].length; j++) {
        const ndvi = ndviData[i][j];
        const idx = (i * 256 + j) * 4;

        // Color mapping: Red -> Yellow -> Green
        let r, g, b;
        if (ndvi < 0.3) {
          // Red zone (stressed)
          r = 255;
          g = Math.floor((ndvi / 0.3) * 200);
          b = 0;
        } else if (ndvi < 0.6) {
          // Yellow to light green
          r = Math.floor((1 - (ndvi - 0.3) / 0.3) * 255);
          g = 255;
          b = 0;
        } else {
          // Dark green (healthy)
          r = 0;
          g = Math.floor(150 + (ndvi - 0.6) / 0.4 * 105);
          b = 0;
        }

        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = Math.floor(opacity * 255);
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Add to map
    const imageUrl = canvas.toDataURL();
    const imageOverlay = L.imageOverlay(imageUrl, bounds, {
      opacity: opacity,
      interactive: false
    }).addTo(map);

    return () => {
      map.removeLayer(imageOverlay);
    };
  }, [ndviData, bounds, opacity, map]);

  return null;
};

/**
 * MapViewer Component
 * Interactive map with:
 * 1. Field polygon drawing
 * 2. NDVI heatmap overlay
 * 3. Location marker
 */
const MapViewer = ({ 
  center = [17.385, 78.486], // Default: Hyderabad
  zoom = 13,
  onFieldDrawn,
  ndviData,
  showNDVILayer = false
}) => {
  const [drawnField, setDrawnField] = useState(null);
  const [mapBounds, setMapBounds] = useState(null);
  const mapRef = useRef(null);

  const handleFieldCreated = (e) => {
    const { layer, layerType } = e;
    
    if (layerType === 'polygon' || layerType === 'rectangle') {
      const coordinates = layer.getLatLngs()[0].map(coord => ({
        lat: coord.lat,
        lng: coord.lng
      }));

      // Calculate area (approximate using Haversine formula)
      const area = calculatePolygonArea(coordinates);
      
      const fieldData = {
        type: layerType,
        coordinates,
        area_hectares: area,
        bounds: layer.getBounds()
      };

      setDrawnField(fieldData);
      setMapBounds(layer.getBounds());
      
      if (onFieldDrawn) {
        onFieldDrawn(fieldData);
      }
    }
  };

  const handleFieldEdited = (e) => {
    const layers = e.layers;
    layers.eachLayer((layer) => {
      const coordinates = layer.getLatLngs()[0].map(coord => ({
        lat: coord.lat,
        lng: coord.lng
      }));
      
      const area = calculatePolygonArea(coordinates);
      
      const fieldData = {
        type: 'polygon',
        coordinates,
        area_hectares: area,
        bounds: layer.getBounds()
      };

      setDrawnField(fieldData);
      setMapBounds(layer.getBounds());
      
      if (onFieldDrawn) {
        onFieldDrawn(fieldData);
      }
    });
  };

  const handleFieldDeleted = () => {
    setDrawnField(null);
    setMapBounds(null);
    if (onFieldDrawn) {
      onFieldDrawn(null);
    }
  };

  /**
   * Calculate polygon area using Shoelace formula
   * Returns area in hectares
   */
  const calculatePolygonArea = (coordinates) => {
    let area = 0;
    const n = coordinates.length;

    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const lat1 = coordinates[i].lat * Math.PI / 180;
      const lat2 = coordinates[j].lat * Math.PI / 180;
      const lng1 = coordinates[i].lng * Math.PI / 180;
      const lng2 = coordinates[j].lng * Math.PI / 180;

      area += (lng2 - lng1) * (2 + Math.sin(lat1) + Math.sin(lat2));
    }

    area = Math.abs(area * 6378137 * 6378137 / 2); // Square meters
    return area / 10000; // Convert to hectares
  };

  return (
    <div className="map-container" style={{ position: 'relative', height: '500px', width: '100%' }}>
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%', borderRadius: '8px' }}
        ref={mapRef}
      >
        {/* Base layer */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Satellite imagery layer (optional) */}
        {/* <TileLayer
          attribution='&copy; <a href="https://www.esri.com">Esri</a>'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        /> */}

        {/* Drawing controls */}
        <FeatureGroup>
          <EditControl
            position="topright"
            onCreated={handleFieldCreated}
            onEdited={handleFieldEdited}
            onDeleted={handleFieldDeleted}
            draw={{
              rectangle: {
                shapeOptions: {
                  color: '#3388ff',
                  fillOpacity: 0.2
                }
              },
              polygon: {
                allowIntersection: false,
                shapeOptions: {
                  color: '#3388ff',
                  fillOpacity: 0.2
                }
              },
              circle: false,
              circlemarker: false,
              marker: false,
              polyline: false
            }}
          />
        </FeatureGroup>

        {/* NDVI Overlay */}
        {showNDVILayer && ndviData && mapBounds && (
          <NDVIOverlay 
            bounds={[
              [mapBounds.getSouth(), mapBounds.getWest()],
              [mapBounds.getNorth(), mapBounds.getEast()]
            ]}
            ndviData={ndviData}
            opacity={0.6}
          />
        )}
      </MapContainer>

      {/* Map legend */}
      {showNDVILayer && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          background: 'white',
          padding: '15px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          zIndex: 1000
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '10px' }}>NDVI Index</div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
            <div style={{ width: '20px', height: '20px', background: '#ff0000', marginRight: '10px' }}></div>
            <span>0.0 - 0.3 (Stressed)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
            <div style={{ width: '20px', height: '20px', background: '#ffff00', marginRight: '10px' }}></div>
            <span>0.3 - 0.6 (Moderate)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ width: '20px', height: '20px', background: '#00ff00', marginRight: '10px' }}></div>
            <span>0.6 - 1.0 (Healthy)</span>
          </div>
        </div>
      )}

      {/* Field info panel */}
      {drawnField && (
        <div style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          background: 'white',
          padding: '15px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          zIndex: 1000,
          minWidth: '200px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>📍 Field Information</div>
          <div>Type: {drawnField.type}</div>
          <div>Area: {drawnField.area_hectares.toFixed(2)} hectares</div>
          <div style={{ fontSize: '0.9em', color: '#666', marginTop: '8px' }}>
            {drawnField.coordinates.length} vertices
          </div>
        </div>
      )}

      {/* Instructions */}
      {!drawnField && (
        <div style={{
          position: 'absolute',
          top: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(51, 136, 255, 0.9)',
          color: 'white',
          padding: '10px 20px',
          borderRadius: '20px',
          zIndex: 1000,
          fontSize: '0.9em'
        }}>
          📐 Draw your field using the polygon or rectangle tool →
        </div>
      )}
    </div>
  );
};

export default MapViewer;
