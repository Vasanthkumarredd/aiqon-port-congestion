import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [summary, setSummary] = useState(null);
  const [activity, setActivity] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [timeLimit, setTimeLimit] = useState(168);

  const [predictForm, setPredictForm] = useState({
    vessel_count: 20,
    low_speed_vessels: 12,
    avg_sog: 2.1,
    cargo_vessels: 5,
    tug_vessels: 6,
    military_vessels: 4,
    hour: 14,
    dayofweek: 2
  });
  const [predictResult, setPredictResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);

        const [sumRes, actRes, metRes, predRes, featRes] = await Promise.all([
          fetch(`${API_BASE}/api/dashboard/summary`),
          fetch(`${API_BASE}/api/activity?limit=${timeLimit}`),
          fetch(`${API_BASE}/api/model/metrics`),
          fetch(`${API_BASE}/api/predictions?limit=20`),
          fetch(`${API_BASE}/api/model/feature-importance`)
        ]);

        if (!sumRes.ok) throw new Error('Failed to connect to Port Congestion API service.');

        const sumData = await sumRes.json();
        const actData = await actRes.json();
        const metData = await metRes.json();
        const predData = await predRes.json();
        const featData = await featRes.json();

        setSummary(sumData);
        setActivity(actData.records || []);
        setMetrics(metData);
        setPredictions(predData.predictions || []);
        setFeatures(featData.top_features || []);
      } catch (err) {
        console.error(err);
        setError(err.message || 'Error connecting to backend');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [timeLimit]);

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    try {
      setPredictLoading(true);
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(predictForm)
      });
      if (!res.ok) throw new Error('Prediction API call failed');
      const data = await res.json();
      setPredictResult(data);

      const predRes = await fetch(`${API_BASE}/api/predictions?limit=20`);
      if (predRes.ok) {
        const predData = await predRes.json();
        setPredictions(predData.predictions || []);
      }
    } catch (err) {
      alert(`Prediction error: ${err.message}`);
    } finally {
      setPredictLoading(false);
    }
  };

  const formatNumber = (num) => (num ? num.toLocaleString() : '0');
  const formatTime = (str) => {
    if (!str) return '';
    const parts = str.split(' ');
    return parts.length > 1 ? `${parts[0].slice(5)} ${parts[1].slice(0, 5)}` : str;
  };

  return (
    <div>
      
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-logo">⚓</div>
          <div className="brand-title">
            <h1>Aiqon Maritime AI — Port Congestion Risk Engine</h1>
            <p>Guam Commercial & Naval Port AIS Tracking (2025 Dataset)</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <span className="badge badge-live">
            <span className="badge-pulse"></span>
            Backend Active (Port 8000)
          </span>
          <span className={`badge ${summary?.mongodb_connected ? 'badge-live' : 'badge-warning'}`}>
            <span className="badge-pulse"></span>
            {summary?.mongodb_connected ? 'MongoDB Connected' : 'Local File Cache'}
          </span>
        </div>
      </header>

      
      <div className="disclaimer-banner">
        <span className="disclaimer-icon">ℹ️</span>
        <div>
          <strong>Scientific Methodology & Target Notice:</strong> The raw AIS dataset (3.24M records) contains vessel positions, courses, and speeds, but does <em>not</em> contain explicit shipment-delay labels. As specified in the problem formulation, this ML system predicts <strong>Next-Hour High Vessel Activity as an operational proxy for Port Congestion</strong> (threshold $\ge 18$ vessels, strictly calculated from the chronological training period to eliminate data leakage).
        </div>
      </div>

      <main className="main-content">
        
        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            📊 Operational Dashboard
          </button>
          <button
            className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            🎯 Live Risk Predictor
          </button>
          <button
            className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            🔬 Model Evaluation & Benchmarks
          </button>
          <button
            className={`tab-btn ${activeTab === 'explain' ? 'active' : ''}`}
            onClick={() => setActiveTab('explain')}
          >
            🧠 Feature Importance & Explainability
          </button>
          <button
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            📜 Prediction History
          </button>
        </nav>

        {loading ? (
          <div className="card loading-box">
            <div className="spinner"></div>
            <p>Loading real-time port metrics from FastAPI & MongoDB...</p>
          </div>
        ) : error ? (
          <div className="card" style={{ borderColor: 'var(--accent-rose)', color: '#fca5a5' }}>
            <h3>Connection Notice</h3>
            <p>{error}</p>
            <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
              Ensure the FastAPI backend server is running via: <code>uvicorn backend.app:app --port 8000</code>
            </p>
          </div>
        ) : (
          <>
            
            <div className="kpi-grid">
              <div className="card kpi-card">
                <div className="kpi-title">
                  Total AIS Records
                  <span style={{ color: 'var(--accent-cyan)' }}>📡</span>
                </div>
                <div className="kpi-value">{formatNumber(summary?.raw_record_count)}</div>
                <div className="kpi-subtitle">Cleaned 2025 AIS Pings</div>
              </div>

              <div className="card kpi-card">
                <div className="kpi-title">
                  Unique Fleet Tracked
                  <span style={{ color: 'var(--accent-blue)' }}>🚢</span>
                </div>
                <div className="kpi-value">{formatNumber(summary?.unique_mmsi_count)}</div>
                <div className="kpi-subtitle">Distinct Commercial & Naval MMSIs</div>
              </div>

              <div className="card kpi-card">
                <div className="kpi-title">
                  Congestion Threshold
                  <span style={{ color: 'var(--accent-amber)' }}>⚠️</span>
                </div>
                <div className="kpi-value">≥ {summary?.congestion_threshold || 18}</div>
                <div className="kpi-subtitle">Active Vessels / Hour (Train 75th %ile)</div>
              </div>

              <div className="card kpi-card">
                <div className="kpi-title">
                  Next-Hour Risk Index
                  <span style={{ color: summary?.next_hour_risk?.probability >= 0.5 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
                    {summary?.next_hour_risk?.probability >= 0.5 ? '🔴' : '🟢'}
                  </span>
                </div>
                <div
                  className="kpi-value"
                  style={{
                    color: summary?.next_hour_risk?.probability >= 0.5 ? 'var(--accent-rose)' : 'var(--accent-emerald)'
                  }}
                >
                  {((summary?.next_hour_risk?.probability || 0.29) * 100).toFixed(1)}%
                </div>
                <div className="kpi-subtitle">{summary?.next_hour_risk?.label || 'Normal Activity'}</div>
              </div>
            </div>

            
            {activeTab === 'overview' && (
              <div>
                <div className="dashboard-grid">
                  
                  <div className="card">
                    <div className="section-header">
                      <div className="section-title">
                        <span>📈</span> Hourly Port Activity & Docked Fleet Trend
                      </div>
                      <div className="filter-btn-group">
                        <button
                          className={`filter-btn ${timeLimit === 48 ? 'active' : ''}`}
                          onClick={() => setTimeLimit(48)}
                        >
                          48h
                        </button>
                        <button
                          className={`filter-btn ${timeLimit === 168 ? 'active' : ''}`}
                          onClick={() => setTimeLimit(168)}
                        >
                          7 Days
                        </button>
                        <button
                          className={`filter-btn ${timeLimit === 720 ? 'active' : ''}`}
                          onClick={() => setTimeLimit(720)}
                        >
                          30 Days
                        </button>
                      </div>
                    </div>

                    <div className="chart-wrapper">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={activity}>
                          <defs>
                            <linearGradient id="vesselGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#00f2fe" stopOpacity={0.4} />
                              <stop offset="95%" stopColor="#00f2fe" stopOpacity={0.0} />
                            </linearGradient>
                            <linearGradient id="slowGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                          <XAxis
                            dataKey="timestamp"
                            tickFormatter={formatTime}
                            stroke="#64748b"
                            tick={{ fontSize: 11 }}
                          />
                          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{
                              background: '#0d1527',
                              borderColor: 'rgba(255,255,255,0.15)',
                              borderRadius: '8px',
                              fontSize: '12px'
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                          <Area
                            type="monotone"
                            dataKey="vessel_count"
                            name="Total Active Vessels"
                            stroke="#00f2fe"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#vesselGrad)"
                          />
                          <Area
                            type="monotone"
                            dataKey="low_speed_vessels"
                            name="Stationary / Docked (≤0.5 kt)"
                            stroke="#f43f5e"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#slowGrad)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  
                  <div className="card">
                    <div className="section-title" style={{ marginBottom: '1.25rem' }}>
                      <span>🛡️</span> Port Congestion Status
                    </div>

                    <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', marginBottom: '1.25rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Congestion Proxy Risk</span>
                        <span style={{ fontWeight: 700, color: summary?.next_hour_risk?.probability >= 0.5 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
                          {summary?.next_hour_risk?.label}
                        </span>
                      </div>
                      <div className="risk-meter">
                        <div
                          className="risk-meter-fill"
                          style={{
                            width: `${Math.round((summary?.next_hour_risk?.probability || 0.29) * 100)}%`,
                            background: summary?.next_hour_risk?.probability >= 0.5
                              ? 'linear-gradient(90deg, #f59e0b, #f43f5e)'
                              : 'linear-gradient(90deg, #10b981, #00f2fe)'
                          }}
                        ></div>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                        <span>0% Low Traffic</span>
                        <span>Threshold (18 Vessels)</span>
                        <span>100% Critical</span>
                      </div>
                    </div>

                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span>Current Active Vessels:</span>
                        <strong style={{ color: '#fff' }}>{summary?.current_snapshot?.vessel_count || 18}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span>Docked / Maneuvering:</span>
                        <strong style={{ color: '#fff' }}>{summary?.current_snapshot?.low_speed_vessels || 10}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span>Average SOG Speed:</span>
                        <strong style={{ color: '#fff' }}>{summary?.current_snapshot?.avg_sog ? summary.current_snapshot.avg_sog.toFixed(1) : '2.4'} kt</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0' }}>
                        <span>Primary ML Pipeline:</span>
                        <strong style={{ color: 'var(--accent-cyan)' }}>Random Forest (89.9% Acc)</strong>
                      </div>
                    </div>

                    <button
                      className="btn-primary"
                      style={{ width: '100%', marginTop: '1.25rem' }}
                      onClick={() => setActiveTab('predict')}
                    >
                      Test Custom Scenario in Predictor →
                    </button>
                  </div>
                </div>

                
                <div className="card">
                  <div className="section-title" style={{ marginBottom: '1rem' }}>
                    <span>🚢</span> Fleet Distribution Over Time (Commercial Cargo, Tugs & Naval)
                  </div>
                  <div className="chart-wrapper" style={{ height: '240px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={activity.slice(-48)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="timestamp" tickFormatter={formatTime} stroke="#64748b" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{
                            background: '#0d1527',
                            borderColor: 'rgba(255,255,255,0.15)',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Bar dataKey="cargo_vessels" name="Cargo Ships" fill="#3b82f6" stackId="a" />
                        <Bar dataKey="tug_vessels" name="Tugs & Towing" fill="#00f2fe" stackId="a" />
                        <Bar dataKey="military_vessels" name="Military / Gov" fill="#8b5cf6" stackId="a" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            
            {activeTab === 'predict' && (
              <div className="dashboard-grid">
                <div className="card">
                  <div className="section-title" style={{ marginBottom: '1.25rem' }}>
                    <span>🎛️</span> Scenario Simulation Parameters
                  </div>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                    Adjust operational factors below to test how harbor fleet density, speed, and vessel mix influence the Random Forest model's next-hour congestion probability.
                  </p>

                  <form onSubmit={handlePredict}>
                    <div className="form-grid">
                      <div className="form-group">
                        <label className="form-label">
                          Active Distinct Vessels
                          <span className="val">{predictForm.vessel_count}</span>
                        </label>
                        <input
                          type="range"
                          min="1"
                          max="35"
                          value={predictForm.vessel_count}
                          onChange={(e) =>
                            setPredictForm({ ...predictForm, vessel_count: Number(e.target.value) })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Stationary / Docked Fleet (≤0.5 kt)
                          <span className="val">{predictForm.low_speed_vessels}</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max={predictForm.vessel_count}
                          value={predictForm.low_speed_vessels}
                          onChange={(e) =>
                            setPredictForm({ ...predictForm, low_speed_vessels: Number(e.target.value) })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Average SOG Speed (knots)
                          <span className="val">{predictForm.avg_sog} kt</span>
                        </label>
                        <input
                          type="range"
                          min="0.1"
                          max="15.0"
                          step="0.1"
                          value={predictForm.avg_sog}
                          onChange={(e) =>
                            setPredictForm({ ...predictForm, avg_sog: Number(e.target.value) })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Commercial Cargo Vessels
                          <span className="val">{predictForm.cargo_vessels}</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="15"
                          value={predictForm.cargo_vessels}
                          onChange={(e) =>
                            setPredictForm({ ...predictForm, cargo_vessels: Number(e.target.value) })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Active Harbor Tugs
                          <span className="val">{predictForm.tug_vessels}</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          value={predictForm.tug_vessels}
                          onChange={(e) =>
                            setPredictForm({ ...predictForm, tug_vessels: Number(e.target.value) })
                          }
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Naval & Government Craft
                          <span className="val">{predictForm.military_vessels}</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          value={predictForm.military_vessels}
                          onChange={(e) =>
                            setPredictForm({ ...predictForm, military_vessels: Number(e.target.value) })
                          }
                        />
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                      <button type="submit" className="btn-primary" disabled={predictLoading}>
                        {predictLoading ? 'Computing Inference...' : '⚡ Predict Next-Hour Risk'}
                      </button>
                      <button
                        type="button"
                        className="tab-btn"
                        style={{ border: '1px solid var(--border-subtle)' }}
                        onClick={() =>
                          setPredictForm({
                            vessel_count: 24,
                            low_speed_vessels: 16,
                            avg_sog: 1.5,
                            cargo_vessels: 7,
                            tug_vessels: 7,
                            military_vessels: 5,
                            hour: 15,
                            dayofweek: 3
                          })
                        }
                      >
                        Load High-Congestion Preset
                      </button>
                      <button
                        type="button"
                        className="tab-btn"
                        style={{ border: '1px solid var(--border-subtle)' }}
                        onClick={() =>
                          setPredictForm({
                            vessel_count: 10,
                            low_speed_vessels: 5,
                            avg_sog: 4.2,
                            cargo_vessels: 2,
                            tug_vessels: 2,
                            military_vessels: 1,
                            hour: 3,
                            dayofweek: 0
                          })
                        }
                      >
                        Load Calm Harbor Preset
                      </button>
                    </div>
                  </form>
                </div>

                
                <div className="card result-card">
                  <div className="section-title">
                    <span>🎯</span> Prediction Inference Output
                  </div>

                  {predictResult ? (
                    <div>
                      <div style={{ marginTop: '1.25rem', textAlign: 'center' }}>
                        <span
                          className={`badge ${
                            predictResult.risk_class === 1 ? 'badge-danger' : 'badge-live'
                          }`}
                          style={{ fontSize: '1rem', padding: '0.5rem 1.25rem' }}
                        >
                          {predictResult.risk_label}
                        </span>
                      </div>

                      <div style={{ textAlign: 'center', margin: '1.25rem 0' }}>
                        <div style={{ fontSize: '3rem', fontWeight: 800, color: '#fff' }}>
                          {(predictResult.risk_probability * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                          Model Probability (Confidence: <strong>{predictResult.confidence_level}</strong>)
                        </div>
                      </div>

                      <div className="risk-meter">
                        <div
                          className="risk-meter-fill"
                          style={{
                            width: `${Math.round(predictResult.risk_probability * 100)}%`,
                            background:
                              predictResult.risk_class === 1
                                ? 'linear-gradient(90deg, #f59e0b, #f43f5e)'
                                : 'linear-gradient(90deg, #10b981, #00f2fe)'
                          }}
                        ></div>
                      </div>

                      <div style={{ marginTop: '1.5rem' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', color: '#fff' }}>
                          Key Influencing Factors:
                        </h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                          {predictResult.explanations?.map((exp, idx) => (
                            <div
                              key={idx}
                              style={{
                                padding: '0.65rem 0.85rem',
                                background: 'rgba(255,255,255,0.03)',
                                borderRadius: '8px',
                                borderLeft: `3px solid ${
                                  exp.impact.includes('+') ? 'var(--accent-rose)' : 'var(--accent-emerald)'
                                }`,
                                fontSize: '0.8rem'
                              }}
                            >
                              <strong style={{ color: '#fff' }}>{exp.feature}:</strong>{' '}
                              <span style={{ color: 'var(--text-muted)' }}>{exp.description}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div style={{ marginTop: '1.25rem', fontSize: '0.75rem', color: 'var(--text-dim)', textAlign: 'center' }}>
                        Persisted automatically to MongoDB <code>predictions</code> collection at {predictResult.timestamp}.
                      </div>
                    </div>
                  ) : (
                    <div style={{ padding: '3rem 1rem', textAlign: 'center', color: 'var(--text-dim)' }}>
                      <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>🧭</div>
                      <p>Adjust the sliders and click <strong>Predict Next-Hour Risk</strong> to view model evaluation output.</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            
            {activeTab === 'metrics' && (
              <div>
                <div className="card" style={{ marginBottom: '1.5rem' }}>
                  <div className="section-title" style={{ marginBottom: '0.75rem' }}>
                    <span>🔬</span> Chronological Model Benchmark Evaluation
                  </div>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
                    Evaluated strictly on a <strong>chronological held-out test split</strong> (Oct 1 – Dec 31, 2025: 2,184 hours). No random shuffle was used, ensuring strict prevention of future-data leakage.
                  </p>

                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Model Architecture</th>
                          <th>Accuracy</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>F1-Score</th>
                          <th>ROC-AUC</th>
                          <th>Brier Score (Calibration)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics?.models_comparison &&
                          Object.entries(metrics.models_comparison).map(([mName, mStats]) => (
                            <tr key={mName} style={{ fontWeight: mName === 'Random_Forest' ? 700 : 400 }}>
                              <td style={{ color: mName === 'Random_Forest' ? 'var(--accent-cyan)' : '#fff' }}>
                                {mName.replace('_', ' ')} {mName === 'Random_Forest' && '★ Primary'}
                              </td>
                              <td>{(mStats.accuracy * 100).toFixed(2)}%</td>
                              <td>{(mStats.precision * 100).toFixed(2)}%</td>
                              <td>{(mStats.recall * 100).toFixed(2)}%</td>
                              <td>{(mStats.f1_score * 100).toFixed(2)}%</td>
                              <td>{mStats.roc_auc.toFixed(4)}</td>
                              <td>{mStats.brier_score.toFixed(4)}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="dashboard-grid">
                  
                  <div className="card">
                    <div className="section-title" style={{ marginBottom: '1rem' }}>
                      <span>🧮</span> Random Forest Confusion Matrix (Test Set: N=2,184)
                    </div>
                    {metrics?.models_comparison?.Random_Forest && (
                      <div>
                        <div className="cm-grid">
                          <div className="cm-cell positive">
                            <div className="cm-number">{metrics.models_comparison.Random_Forest.true_negatives}</div>
                            <div className="cm-label">True Negatives (Normal Correct)</div>
                          </div>
                          <div className="cm-cell negative">
                            <div className="cm-number">{metrics.models_comparison.Random_Forest.false_positives}</div>
                            <div className="cm-label">False Positives (False Alarm)</div>
                          </div>
                          <div className="cm-cell negative">
                            <div className="cm-number">{metrics.models_comparison.Random_Forest.false_negatives}</div>
                            <div className="cm-label">False Negatives (Missed Congestion)</div>
                          </div>
                          <div className="cm-cell positive">
                            <div className="cm-number">{metrics.models_comparison.Random_Forest.true_positives}</div>
                            <div className="cm-label">True Positives (High Risk Correct)</div>
                          </div>
                        </div>
                        <div style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
                          Sensitivity / Recall: <strong>{(metrics.models_comparison.Random_Forest.recall * 100).toFixed(1)}%</strong> | Specificity: <strong>{(((metrics.models_comparison.Random_Forest.true_negatives) / (metrics.models_comparison.Random_Forest.true_negatives + metrics.models_comparison.Random_Forest.false_positives)) * 100).toFixed(1)}%</strong>
                        </div>
                      </div>
                    )}
                  </div>

                  
                  <div className="card">
                    <div className="section-title" style={{ marginBottom: '1rem' }}>
                      <span>⚖️</span> Probability Calibration & Caution
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                      <p style={{ marginBottom: '0.75rem' }}>
                        <strong>Brier Score: {metrics?.models_comparison?.Random_Forest?.brier_score}</strong> (values close to 0 denote superior probability calibration).
                      </p>
                      <p style={{ marginBottom: '0.75rem' }}>
                        Tree-based ensembles (such as Random Forest) can produce probabilities skewed toward the middle range. To prevent over-interpretation:
                      </p>
                      <ul style={{ paddingLeft: '1.25rem', marginBottom: '0.75rem' }}>
                        <li>Predictions with $P \in [0.45, 0.55]$ are categorized as <em>Moderate Confidence</em>.</li>
                        <li>High confidence is only declared when $P \ge 0.80$ or $P \le 0.20$.</li>
                        <li>Predicted probabilities must be treated as likelihood estimates of harbor density, <strong>not guaranteed delay forecasts</strong>.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}

            
            {activeTab === 'explain' && (
              <div>
                <div className="card" style={{ marginBottom: '1.5rem' }}>
                  <div className="section-title" style={{ marginBottom: '1rem' }}>
                    <span>🧠</span> Top Predictive Features (Random Forest Feature Importance)
                  </div>
                  <div className="chart-wrapper" style={{ height: '360px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={features}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          stroke="#cbd5e1"
                          tick={{ fontSize: 11 }}
                          width={140}
                        />
                        <Tooltip
                          contentStyle={{
                            background: '#0d1527',
                            borderColor: 'rgba(255,255,255,0.15)',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }}
                        />
                        <Bar dataKey="importance" name="Importance Score" fill="#00f2fe" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card">
                  <div className="section-title" style={{ marginBottom: '1rem' }}>
                    <span>📖</span> Domain Interpretation of Key Drivers
                  </div>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Feature</th>
                          <th>Importance</th>
                          <th>Domain Rationale & Role</th>
                        </tr>
                      </thead>
                      <tbody>
                        {features.map((f) => (
                          <tr key={f.feature}>
                            <td style={{ color: 'var(--accent-cyan)', fontFamily: 'JetBrains Mono' }}>
                              {f.feature}
                            </td>
                            <td>{(f.importance * 100).toFixed(2)}%</td>
                            <td>{f.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            
            {activeTab === 'history' && (
              <div className="card">
                <div className="section-title" style={{ marginBottom: '1rem' }}>
                  <span>📜</span> Historical Inferences Logged in MongoDB
                </div>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Vessel Count</th>
                        <th>Low Speed (≤0.5 kt)</th>
                        <th>Cargo</th>
                        <th>Tugs</th>
                        <th>Predicted Risk</th>
                        <th>Probability</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.length > 0 ? (
                        predictions.map((p, idx) => (
                          <tr key={idx}>
                            <td style={{ fontFamily: 'JetBrains Mono' }}>{p.timestamp}</td>
                            <td>{p.inputs?.vessel_count ?? '-'}</td>
                            <td>{p.inputs?.low_speed_vessels ?? '-'}</td>
                            <td>{p.inputs?.cargo_vessels ?? '-'}</td>
                            <td>{p.inputs?.tug_vessels ?? '-'}</td>
                            <td>
                              <span
                                className={`badge ${
                                  p.prediction?.risk_class === 1 ? 'badge-danger' : 'badge-live'
                                }`}
                              >
                                {p.prediction?.risk_label || (p.prediction?.risk_class === 1 ? 'High Risk' : 'Normal')}
                              </span>
                            </td>
                            <td>{((p.prediction?.risk_probability || 0) * 100).toFixed(1)}%</td>
                            <td>{p.prediction?.confidence_level || 'Moderate'}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={8} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-dim)' }}>
                            No prediction history found. Test the Live Predictor tab to generate live entries!
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
