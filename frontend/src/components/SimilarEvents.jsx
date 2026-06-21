import React from 'react'

const SEV_CLASS = {
  Low:      'tag-low',
  Medium:   'tag-medium',
  High:     'tag-high',
  Critical: 'tag-critical',
}

export default function SimilarEvents({ events = [] }) {
  if (!events.length) return (
    <p style={{ color: '#7d8590', fontSize: 13 }}>No similar events found.</p>
  )

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #21262d' }}>
            {['ID', 'Cause', 'Zone', 'Duration', 'Date', 'Severity'].map(h => (
              <th key={h} style={{
                padding: '6px 12px', textAlign: 'left',
                color: '#7d8590', fontWeight: 500, fontSize: 11,
                whiteSpace: 'nowrap',
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={e.id} style={{
              borderBottom: i < events.length - 1 ? '1px solid #161b22' : 'none',
              transition: 'background 0.1s',
            }}
              onMouseEnter={ev => ev.currentTarget.style.background = '#1c2128'}
              onMouseLeave={ev => ev.currentTarget.style.background = 'transparent'}
            >
              <td style={{ padding: '8px 12px', fontFamily: "'JetBrains Mono'", fontSize: 11, color: '#58a6ff' }}>
                {e.id}
              </td>
              <td style={{ padding: '8px 12px', color: '#e6edf3' }}>
                {e.cause.replace(/_/g, ' ')}
              </td>
              <td style={{ padding: '8px 12px', color: '#7d8590' }}>{e.zone}</td>
              <td style={{ padding: '8px 12px', fontFamily: "'JetBrains Mono'", color: '#d29922' }}>
                {e.duration_mins} min
              </td>
              <td style={{ padding: '8px 12px', color: '#7d8590', fontSize: 11 }}>{e.date}</td>
              <td style={{ padding: '8px 12px' }}>
                <span className={`tag ${SEV_CLASS[e.severity] || 'tag-blue'}`}>
                  {e.severity}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
