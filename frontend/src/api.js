const BASE = 'https://evensaathi.onrender.com'

export async function fetchOptions() {
  const r = await fetch('https://evensaathi.onrender.com/options')
  if (!r.ok) throw new Error('Failed to fetch options')
  return r.json()
}

export async function predict(payload) {
  const r = await fetch('https://evensaathi.onrender.com/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error(err.detail || 'Prediction failed')
  }
  return r.json()
}

export async function checkHealth() {
  const r = await fetch('https://evensaathi.onrender.com/health')
  return r.json()
}
