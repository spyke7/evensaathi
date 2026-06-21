import React, { useState, useEffect } from 'react'
import { fetchOptions, predict } from './api.js'
import ImportanceChart from './components/ImportanceChart.jsx'
import SimilarEvents from './components/SimilarEvents.jsx'
import ResourceCards from './components/ResourceCards.jsx'

// ── tiny helpers ──────────────────────────────────────────────────────────────
const SEV_COLOR = {
  Low: '#3fb950', Medium: '#d29922', High: '#f78166', Critical: '#f85149'
}
const SEV_TAG = {
  Low: 'tag-low', Medium: 'tag-medium', High: 'tag-high', Critical: 'tag-critical'
}

function Label({ children, required }) {
  return (
    <label style={{ fontSize: 12, color: '#7d8590', fontWeight: 500, letterSpacing: '0.4px', textTransform: 'uppercase', display: 'block', marginBottom: 6 }}>
      {children}{required && <span style={{ color: '#f85149', marginLeft: 2 }}>*</span>}
    </label>
  )
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: '#161b22',
      border: '1px solid #30363d',
      borderRadius: 12,
      padding: '20px 24px',
      ...style
    }}>
      {children}
    </div>
  )
}

function Section({ title, children, accent }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
        {accent && <div style={{ width: 3, height: 16, background: accent, borderRadius: 2 }} />}
        <h3 style={{ fontSize: 13, fontWeight: 600, color: '#e6edf3', letterSpacing: '0.3px' }}>{title}</h3>
      </div>
      {children}
    </div>
  )
}

function FormField({ label, required, children }) {
  return (
    <div>
      <Label required={required}>{label}</Label>
      {children}
    </div>
  )
}

// ── default form state ────────────────────────────────────────────────────────
const DEFAULT_FORM = {
  event_type: 'unplanned',
  event_cause: 'vehicle_breakdown',
  latitude: '12.9716',
  longitude: '77.5946',
  hour: new Date().getHours(),
  day_of_week: new Date().getDay() === 0 ? 6 : new Date().getDay() - 1,
  zone: 'Central Zone 1',
  junction: 'Unknown',
  veh_type: 'unknown',
  requires_road_closure: false,
  has_end_location: false,
  corridor: '',
}

// ── main component ────────────────────────────────────────────────────────────
export default function App() {
  const [form, setForm]       = useState(DEFAULT_FORM)
  const [options, setOptions] = useState(null)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [activeTab, setActiveTab] = useState('prediction')

  useEffect(() => {
    fetchOptions()
      .then(setOptions)
      .catch(() => setOptions({
    event_types:  ['planned', 'unplanned'],
    event_causes: [
        'vehicle_breakdown', 'accident', 'tree_fall', 'construction',
        'water_logging', 'congestion', 'procession', 'protest',
        'road_conditions', 'pot_holes', 'others'
    ],
    veh_types: [
        'Unknown', 'private_car', 'truck', 'auto', 'heavy_vehicle',
        'bmtc_bus', 'ksrtc_bus', 'private_bus', 'lcv', 'taxi', 'others'
    ],
    zones: [
        'Central Zone 1', 'Central Zone 2',
        'North Zone 1', 'North Zone 2',
        'South Zone 1', 'South Zone 2',
        'East Zone 1', 'East Zone 2',
        'West Zone 1', 'West Zone 2', 'Unknown'
    ],
    junctions: ['Unknown'],
}))
  }, [])

  const set = (k) => (e) => setForm(f => ({
    ...f, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value
  }))

  const handleSubmit = async () => {
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      const payload = {
        ...form,
        latitude:   parseFloat(form.latitude)  || 12.9716,
        longitude:  parseFloat(form.longitude) || 77.5946,
        hour:       parseInt(form.hour),
        day_of_week:parseInt(form.day_of_week),
      }
      const res = await predict(payload)
      setResult(res)
      setActiveTab('prediction')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const DAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

  return (
    <div style={{ minHeight: '100vh', background: '#0d1117' }}>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header style={{
        background: '#161b22',
        borderBottom: '1px solid #30363d',
        padding: '0 32px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 56,
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'linear-gradient(135deg, #1a2d4a, #58a6ff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 16,
          }}>🚦</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#e6edf3', letterSpacing: '-0.3px' }}>
              EventSaathi
            </div>
            <div style={{ fontSize: 10, color: '#7d8590', marginTop: -2 }}>
              Event Congestion Predictor
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            fontSize: 11, color: '#3fb950', background: '#1a3a23',
            border: '1px solid #3fb95033', borderRadius: 20,
            padding: '3px 10px', fontFamily: "'JetBrains Mono'",
          }}>
            ● API connected
          </div>
          <div style={{ fontSize: 11, color: '#7d8590', fontFamily: "'JetBrains Mono'" }}>
            Bengaluru Traffic
          </div>
        </div>
      </header>

      {/* ── Body layout ────────────────────────────────────────────────── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '380px 1fr',
        minHeight: 'calc(100vh - 56px)',
        gap: 0,
      }}>

        {/* ── LEFT PANEL: Input Form ─────────────────────────────────── */}
        <aside style={{
          background: '#161b22',
          borderRight: '1px solid #30363d',
          padding: '24px 24px 32px',
          overflowY: 'auto',
          position: 'sticky', top: 56,
          height: 'calc(100vh - 56px)',
        }}>
          <div style={{ marginBottom: 20 }}>
            <h2 style={{ fontSize: 16, fontWeight: 600, color: '#e6edf3' }}>Event Details</h2>
            <p style={{ fontSize: 12, color: '#7d8590', marginTop: 4 }}>
              Enter incident or event parameters to forecast impact
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Event classification */}
            <Section title="Classification" accent="#58a6ff">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <FormField label="Event type" required>
                  <select value={form.event_type} onChange={set('event_type')}>
                    {(options?.event_types || ['planned','unplanned']).map(v => (
                      <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>
                    ))}
                  </select>
                </FormField>
                <FormField label="Vehicle type">
                  <select value={form.veh_type} onChange={set('veh_type')}>
                    {(options?.veh_types || []).map(v => (
                      <option key={v} value={v}>{v.replace(/-/g,' ')}</option>
                    ))}
                  </select>
                </FormField>
              </div>
              <div style={{ marginTop: 12 }}>
                <FormField label="Event cause" required>
                  <select value={form.event_cause} onChange={set('event_cause')}>
                    {(options?.event_causes || []).map(v => (
                      <option key={v} value={v}>{v.replace(/_/g,' ')}</option>
                    ))}
                  </select>
                </FormField>
              </div>
            </Section>

            {/* Location */}
            <Section title="Location" accent="#bc8cff">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <FormField label="Latitude" required>
                  <input type="number" step="0.0001" value={form.latitude} onChange={set('latitude')} placeholder="12.9716" />
                </FormField>
                <FormField label="Longitude" required>
                  <input type="number" step="0.0001" value={form.longitude} onChange={set('longitude')} placeholder="77.5946" />
                </FormField>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
                <FormField label="Zone">
                  <select value={form.zone} onChange={set('zone')}>
                    {(options?.zones || []).map(v => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                </FormField>
                <FormField label="Junction">
                  <select value={form.junction} onChange={set('junction')}>
                    {(options?.junctions || []).map(v => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                </FormField>
              </div>
            </Section>

            {/* Time */}
            <Section title="Time context" accent="#d29922">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <FormField label="Hour of day (0–23)" required>
                  <input type="number" min="0" max="23" value={form.hour} onChange={set('hour')} />
                </FormField>
                <FormField label="Day of week" required>
                  <select value={form.day_of_week} onChange={set('day_of_week')}>
                    {DAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
                  </select>
                </FormField>
              </div>
              {/* Quick time presets */}
              <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
                {[['Morning peak', 8], ['Afternoon', 13], ['Evening peak', 18], ['Night', 23]].map(([label, h]) => (
                  <button
                    key={label}
                    onClick={() => setForm(f => ({ ...f, hour: h }))}
                    style={{
                      fontSize: 11, padding: '4px 10px',
                      background: form.hour === h ? '#1a2d4a' : '#21262d',
                      border: `1px solid ${form.hour === h ? '#58a6ff' : '#30363d'}`,
                      color: form.hour === h ? '#58a6ff' : '#7d8590',
                      borderRadius: 20, cursor: 'pointer',
                    }}>
                    {label}
                  </button>
                ))}
              </div>
            </Section>

            {/* Flags */}
            <Section title="Incident flags" accent="#f78166">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {[
                  ['requires_road_closure', 'Road closure required'],
                  ['has_end_location',      'Incident spans two points'],
                ].map(([key, lbl]) => (
                  <label key={key} style={{
                    display: 'flex', alignItems: 'center', gap: 10,
                    cursor: 'pointer', padding: '10px 12px',
                    background: form[key] ? '#3d1f1a' : '#1c2128',
                    border: `1px solid ${form[key] ? '#f7816633' : '#30363d'}`,
                    borderRadius: 8, transition: 'all 0.15s',
                  }}>
                    <input type="checkbox" checked={form[key]} onChange={set(key)} />
                    <span style={{ fontSize: 13, color: form[key] ? '#f78166' : '#e6edf3' }}>{lbl}</span>
                  </label>
                ))}
              </div>
            </Section>

          </div>

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{
              width: '100%', marginTop: 8,
              padding: '12px 0',
              background: loading ? '#21262d' : '#1a2d4a',
              border: `1px solid ${loading ? '#30363d' : '#58a6ff'}`,
              borderRadius: 8, color: loading ? '#7d8590' : '#58a6ff',
              fontSize: 14, fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer',
              letterSpacing: '0.3px', transition: 'all 0.15s',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            }}>
            {loading ? (
              <>
                <span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⟳</span>
                Predicting…
              </>
            ) : '🔮 Predict congestion impact'}
          </button>

          {error && (
            <div style={{
              marginTop: 12, padding: '10px 14px',
              background: '#3d1615', border: '1px solid #f8514933',
              borderRadius: 8, fontSize: 12, color: '#f85149',
            }}>
              ⚠ {error}
            </div>
          )}
        </aside>

        {/* ── RIGHT PANEL: Output Dashboard ─────────────────────────── */}
        <main style={{ padding: '24px 28px', overflowY: 'auto' }}>

          {!result && !loading && (
            <div style={{
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
              minHeight: 500, textAlign: 'center', gap: 16,
            }}>
              <div style={{ fontSize: 56 }}>🚦</div>
              <h2 style={{ fontSize: 20, fontWeight: 600, color: '#e6edf3' }}>
                Ready to predict
              </h2>
              <p style={{ fontSize: 14, color: '#7d8590', maxWidth: 380, lineHeight: 1.7 }}>
                Fill in the event details on the left panel and click{' '}
                <strong style={{ color: '#58a6ff' }}>Predict congestion impact</strong>{' '}
                to get duration estimates, severity scores, and resource recommendations.
              </p>
              <div style={{
                display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12,
                maxWidth: 500, marginTop: 16,
              }}>
                {[
                  ['⏱', 'Duration forecast', 'Estimated incident resolution time'],
                  ['📊', 'Severity scoring', 'Low → Critical classification'],
                  ['👮', 'Resource plan', 'Manpower, barricades, diversions'],
                ].map(([icon, title, desc]) => (
                  <div key={title} style={{
                    background: '#161b22', border: '1px solid #30363d',
                    borderRadius: 10, padding: '16px 14px', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 24, marginBottom: 8 }}>{icon}</div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#e6edf3', marginBottom: 4 }}>{title}</div>
                    <div style={{ fontSize: 11, color: '#7d8590' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {loading && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              minHeight: 400, flexDirection: 'column', gap: 16,
            }}>
              <div style={{ fontSize: 40, animation: 'spin 1.2s linear infinite' }}>⟳</div>
              <p style={{ color: '#7d8590', fontSize: 14 }}>Running prediction models…</p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* ── Top summary bar ── */}
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                marginBottom: 20, flexWrap: 'wrap', gap: 12,
              }}>
                <div>
                  <h2 style={{ fontSize: 18, fontWeight: 700, color: '#e6edf3' }}>
                    Prediction results
                  </h2>
                  <p style={{ fontSize: 12, color: '#7d8590', marginTop: 2 }}>
                    Model: {result.model_used} · Confidence: {result.confidence}
                  </p>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span className={`tag ${SEV_TAG[result.severity]}`}>
                    ● {result.severity} severity
                  </span>
                  <span className="tag tag-blue">
                    {form.event_cause.replace(/_/g,' ')}
                  </span>
                  <span style={{
                    fontSize: 11, color: '#7d8590', fontFamily: "'JetBrains Mono'",
                    padding: '2px 8px',
                  }}>
                    {form.hour}:00 · {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][form.day_of_week]}
                  </span>
                </div>
              </div>

              {/* ── Tabs ── */}
              <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '1px solid #30363d' }}>
                {[
                  ['prediction', '📊 Prediction'],
                  ['resources',  '👮 Resources'],
                  ['explain',    '🔍 Explainability'],
                  ['similar',    '📋 Similar events'],
                ].map(([id, label]) => (
                  <button key={id} onClick={() => setActiveTab(id)} style={{
                    padding: '10px 18px', background: 'none',
                    border: 'none', cursor: 'pointer',
                    fontSize: 13, fontWeight: 500,
                    color: activeTab === id ? '#58a6ff' : '#7d8590',
                    borderBottom: `2px solid ${activeTab === id ? '#58a6ff' : 'transparent'}`,
                    marginBottom: -1, transition: 'all 0.15s',
                  }}>
                    {label}
                  </button>
                ))}
              </div>

              {/* ── TAB: Prediction ── */}
              {activeTab === 'prediction' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

                  {/* Hero metrics */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16 }}>
                    {[
                      {
                        label: 'Est. duration',
                        value: result.duration_label,
                        sub: `${result.duration_mins} minutes`,
                        color: '#d29922', dim: '#2d2008',
                      },
                      {
                        label: 'Severity level',
                        value: result.severity,
                        sub: `Score: ${result.severity_score}/100`,
                        color: SEV_COLOR[result.severity], dim: '#1c2128',
                      },
                      {
                        label: 'Road closure',
                        value: result.road_closure_label,
                        sub: `${(result.road_closure_prob * 100).toFixed(0)}% probability`,
                        color: result.road_closure_prob > 0.5 ? '#f78166' : '#3fb950',
                        dim: result.road_closure_prob > 0.5 ? '#3d1f1a' : '#1a3a23',
                      },
                      {
                        label: 'Manpower',
                        value: `${result.resources.manpower_min}–${result.resources.manpower_max}`,
                        sub: 'personnel needed',
                        color: '#bc8cff', dim: '#271c40',
                      },
                    ].map(({ label, value, sub, color, dim }) => (
                      <div key={label} style={{
                        background: dim, border: `1px solid ${color}33`,
                        borderRadius: 10, padding: '16px 18px',
                      }}>
                        <div style={{ fontSize: 11, color, fontWeight: 600, letterSpacing: '0.4px', marginBottom: 6, textTransform: 'uppercase' }}>
                          {label}
                        </div>
                        <div style={{
                          fontSize: 24, fontWeight: 700, color,
                          fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.2,
                        }}>
                          {value}
                        </div>
                        <div style={{ fontSize: 11, color: '#7d8590', marginTop: 4 }}>{sub}</div>
                      </div>
                    ))}
                  </div>

                  {/* Road closure prob */}
                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#7d8590', marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                      Road closure probability
                    </div>
                    <div style={{ textAlign: 'center', marginBottom: 20 }}>
                      <div style={{
                        fontSize: 48, fontWeight: 700,
                        fontFamily: "'JetBrains Mono', monospace",
                        color: result.road_closure_prob > 0.5 ? '#f78166' : '#3fb950',
                      }}>
                        {(result.road_closure_prob * 100).toFixed(0)}%
                      </div>
                      <div style={{ fontSize: 13, color: '#7d8590', marginTop: 4 }}>
                        {result.road_closure_label}
                      </div>
                    </div>
                    {/* Prob bar */}
                    <div style={{ height: 8, background: '#21262d', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{
                        width: `${result.road_closure_prob * 100}%`,
                        height: '100%', borderRadius: 4,
                        background: result.road_closure_prob > 0.5
                          ? 'linear-gradient(90deg, #3d1f1a, #f78166)'
                          : 'linear-gradient(90deg, #1a3a23, #3fb950)',
                        transition: 'width 0.8s ease',
                      }} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 10, color: '#484f58' }}>
                      <span>Unlikely</span><span>Very likely</span>
                    </div>

                    {/* Diversion routes if needed */}
                    {result.resources.diversion_needed && result.resources.diversion_zones.length > 0 && (
                      <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #21262d' }}>
                        <div style={{ fontSize: 11, color: '#7d8590', marginBottom: 8, fontWeight: 600 }}>
                          SUGGESTED DIVERSIONS
                        </div>
                        {result.resources.diversion_zones.map(z => (
                          <div key={z} style={{
                            fontSize: 12, color: '#58a6ff',
                            padding: '4px 0', display: 'flex', alignItems: 'center', gap: 6,
                          }}>
                            🔀 {z}
                          </div>
                        ))}
                      </div>
                    )}
                  </Card>
                </div>
              )}

              {/* ── TAB: Resources ── */}
              {activeTab === 'resources' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#7d8590', marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                      Deployment recommendation
                    </div>
                    <ResourceCards resources={result.resources} severity={result.severity} />
                  </Card>

                  {result.resources.diversion_needed && (
                    <Card>
                      <div style={{ fontSize: 12, fontWeight: 600, color: '#7d8590', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                        Diversion routes
                      </div>
                      {result.resources.diversion_zones.map((z, i) => (
                        <div key={z} style={{
                          display: 'flex', alignItems: 'center', gap: 12,
                          padding: '12px 0',
                          borderBottom: i < result.resources.diversion_zones.length - 1 ? '1px solid #21262d' : 'none',
                        }}>
                          <div style={{
                            width: 28, height: 28, borderRadius: '50%',
                            background: '#1a2d4a', color: '#58a6ff',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: 12, fontWeight: 700, flexShrink: 0,
                          }}>
                            {i + 1}
                          </div>
                          <div style={{ fontSize: 13, color: '#e6edf3' }}>🔀 {z}</div>
                        </div>
                      ))}
                    </Card>
                  )}

                  {/* Action checklist */}
                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#7d8590', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                      Deployment checklist
                    </div>
                    {[
                      `Deploy ${result.resources.manpower_min}–${result.resources.manpower_max} personnel immediately`,
                      `Position ${result.resources.barricades} barricade units at incident perimeter`,
                      result.resources.diversion_needed
                        ? `Activate ${result.resources.diversion_zones.length} diversion route(s)`
                        : 'Monitor traffic flow — no diversion required',
                      result.severity === 'Critical' || result.severity === 'High'
                        ? 'Notify traffic control center — high impact event'
                        : 'Standard incident protocol applicable',
                      `Estimated clearance in ${result.duration_label}`,
                    ].map((item, i) => (
                      <div key={i} style={{
                        display: 'flex', alignItems: 'flex-start', gap: 10,
                        padding: '10px 0',
                        borderBottom: i < 4 ? '1px solid #1c2128' : 'none',
                        fontSize: 13, color: '#e6edf3',
                      }}>
                        <span style={{ color: '#3fb950', fontSize: 14, marginTop: 1, flexShrink: 0 }}>✓</span>
                        {item}
                      </div>
                    ))}
                  </Card>
                </div>
              )}

              {/* ── TAB: Explainability ── */}
              {activeTab === 'explain' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#7d8590', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                      Feature importance
                    </div>
                    <p style={{ fontSize: 12, color: '#484f58', marginBottom: 16 }}>
                      Relative contribution of each feature to this prediction
                    </p>
                    <ImportanceChart data={result.feature_importances} />
                  </Card>

                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#7d8590', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                      Prediction breakdown
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                      {[
                        ['Input: event cause', form.event_cause.replace(/_/g,' ')],
                        ['Input: hour of day', `${form.hour}:00${form.hour < 12 ? ' AM' : ' PM'}`],
                        ['Input: zone', form.zone || 'Unknown'],
                        ['Input: day', ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][form.day_of_week]],
                        ['Output: duration', result.duration_label],
                        ['Output: severity', result.severity],
                      ].map(([lbl, val]) => (
                        <div key={lbl} style={{
                          background: '#1c2128', borderRadius: 8,
                          padding: '10px 14px',
                        }}>
                          <div style={{ fontSize: 10, color: '#484f58', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                            {lbl}
                          </div>
                          <div style={{
                            fontSize: 14, fontWeight: 600, color: '#e6edf3',
                            fontFamily: "'JetBrains Mono', monospace",
                          }}>
                            {val}
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              )}

              {/* ── TAB: Similar events ── */}
              {activeTab === 'similar' && (
                <Card>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#7d8590', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                    Similar historical events
                  </div>
                  <p style={{ fontSize: 12, color: '#484f58', marginBottom: 16 }}>
                    Past incidents with matching cause and zone
                  </p>
                  <SimilarEvents events={result.similar_events} />
                </Card>
              )}
            </>
          )}
        </main>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}