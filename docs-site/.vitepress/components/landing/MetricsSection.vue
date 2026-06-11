<template>
  <div class="metrics-section">
    <div class="section-header">
      <span class="eyebrow">Performance</span>
      <h2 class="heading">Built for speed</h2>
      <p class="lede">No cloud queues. No API limits. Your hardware, your images.</p>
    </div>

    <div class="metrics-grid">
      <div class="metric">
        <span class="metric-value">~14s</span>
        <span class="metric-label">512x512 on M2 Max</span>
      </div>
      <div class="metric">
        <span class="metric-value">~8GB</span>
        <span class="metric-label">Minimum VRAM</span>
      </div>
      <div class="metric">
        <span class="metric-value">6</span>
        <span class="metric-label">Reference images</span>
      </div>
      <div class="metric">
        <span class="metric-value">2K</span>
        <span class="metric-label">Max resolution</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.metrics-section {
  max-width: 80rem;
  margin: 0 auto 7rem;
  padding: 0 1rem;
}

.section-header {
  text-align: center;
  margin-bottom: 3.5rem;
}

.eyebrow {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  background: hsl(var(--primary) / 0.1);
  color: hsl(var(--primary));
  border: 1px solid hsl(var(--primary) / 0.15);
}

.heading {
  font-size: clamp(1.75rem, 5vw, 3rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.15;
  margin-bottom: 0.75rem;
  color: hsl(var(--foreground));
}

.lede {
  font-size: 1rem;
  color: hsl(var(--muted-foreground));
  max-width: 36rem;
  margin: 0 auto;
  line-height: 1.6;
}

@media (min-width: 640px) {
  .lede { font-size: 1.125rem; }
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
  max-width: 48rem;
  margin: 0 auto;
}

@media (min-width: 768px) {
  .metrics-grid { grid-template-columns: repeat(4, 1fr); gap: 2.5rem; }
}

.metric {
  text-align: center;
  padding: 1.5rem 0.75rem;
  border-radius: 0.875rem;
  background: hsl(var(--card));
  border: 1px solid hsl(var(--border) / 0.4);
  transition: border-color 0.3s ease;
}

.metric:hover {
  border-color: hsl(var(--primary) / 0.2);
}

.metric-value {
  display: block;
  font-size: clamp(2rem, 6vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
  margin-bottom: 0.5rem;
  background: linear-gradient(160deg, hsl(var(--foreground)) 20%, hsl(var(--primary)) 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.metric-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: hsl(var(--muted-foreground) / 0.7);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
</style>
