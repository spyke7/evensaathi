import React from 'react'

function ResourceCard({ icon, label, value, sub, color = '#58a6ff', dimColor = '#1a2d4a' }) {
  return (
    <div style={{
      background: dimColor,
      border: `1px solid ${color}33`,
      borderRadius: 10,
      padding: '14px 16px',
      display: 'flex', flexDirection: 'column', gap: 4,
    }}>
      <div style={{ fontSize: 20 }}>{icon}</div>
      <div style={{
        fontSize: 22, fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        color,
      }}>{value}</div>
      <div style={{ fontSize: 12, fontWeight: 600, color }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#7d8590', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export default function ResourceCards({ resources, severity }) {
  if (!resources) return null

  const colors = {
    Low:      { fg: '#3fb950', bg: '#1a3a23' },
    Medium:   { fg: '#d29922', bg: '#2d2008' },
    High:     { fg: '#f78166', bg: '#3d1f1a' },
    Critical: { fg: '#f85149', bg: '#3d1615' },
  }
  const c = colors[severity] || colors.Medium

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
      <ResourceCard
        icon="👮"
        label="Manpower needed"
        value={`${resources.manpower_min}–${resources.manpower_max}`}
        sub="personnel"
        color={c.fg} dimColor={c.bg}
      />
      <ResourceCard
        icon="🚧"
        label="Barricades"
        value={resources.barricades}
        sub="units required"
        color="#d29922" dimColor="#2d2008"
      />
      <ResourceCard
        icon={resources.diversion_needed ? "🔀" : "✅"}
        label="Diversion"
        value={resources.diversion_needed ? "Required" : "Not needed"}
        sub={resources.diversion_needed ? `${resources.diversion_zones.length} route(s)` : "Normal traffic flow"}
        color={resources.diversion_needed ? '#f78166' : '#3fb950'}
        dimColor={resources.diversion_needed ? '#3d1f1a' : '#1a3a23'}
      />
    </div>
  )
}
