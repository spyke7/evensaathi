import React from 'react'

export default function ImportanceChart({ data = [] }) {
  const sorted = [...data].sort((a, b) => b.importance - a.importance).slice(0, 9)
  const max    = sorted[0]?.importance || 1

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {sorted.map((d) => {
        const pct = (d.importance / max) * 100
        return (
          <div key={d.feature} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{
              width: 140, fontSize: 11, color: '#7d8590',
              fontFamily: "'JetBrains Mono', monospace",
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              textAlign: 'right', flexShrink: 0,
            }}>
              {d.feature}
            </span>
            <div style={{
              flex: 1, height: 6, background: '#21262d',
              borderRadius: 3, overflow: 'hidden',
            }}>
              <div style={{
                width: `${pct}%`, height: '100%',
                background: 'linear-gradient(90deg, #1a2d4a, #58a6ff)',
                borderRadius: 3,
                transition: 'width 0.6s ease',
              }} />
            </div>
            <span style={{
              width: 36, fontSize: 11, color: '#58a6ff',
              fontFamily: "'JetBrains Mono', monospace",
              flexShrink: 0,
            }}>
              {(d.importance * 100).toFixed(0)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}
