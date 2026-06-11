<template>
  <div class="cta-section">
    <div class="cta-glow-orbs">
      <div class="orb one"></div>
      <div class="orb two"></div>
    </div>
    <h2 class="heading">Start generating in seconds</h2>
    <p class="lede">Open source. Local-first. Zero API keys. Get up and running in under a minute.</p>
    <div class="cta-row">
      <a href="/ultra-fast-image-gen/guide/quick-start.html" class="btn-glow">Get Started</a>
      <a href="/ultra-fast-image-gen/guide/mcp.html" class="btn-ghost">Configure MCP</a>
    </div>
  </div>
</template>

<style scoped>
.cta-section {
  position: relative;
  max-width: 80rem;
  margin: 0 auto 5rem;
  padding: 4rem 2rem;
  text-align: center;
  border-radius: 1.25rem;
  border: 1px solid hsl(var(--primary) / 0.12);
  background: linear-gradient(160deg, hsl(var(--primary) / 0.06) 0%, hsl(var(--card)) 40%, hsl(var(--primary) / 0.04) 100%);
  overflow: hidden;
}

@media (min-width: 640px) {
  .cta-section { padding: 5rem 3rem; }
}

.cta-glow-orbs {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
}

.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
}

.orb.one {
  width: 400px;
  height: 400px;
  top: -200px;
  right: -100px;
  background: hsl(var(--primary) / 0.08);
}

.orb.two {
  width: 300px;
  height: 300px;
  bottom: -150px;
  left: -80px;
  background: hsl(var(--primary) / 0.06);
}

.heading {
  position: relative;
  font-size: clamp(1.75rem, 5vw, 2.75rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 0.75rem;
  color: hsl(var(--foreground));
}

.lede {
  position: relative;
  font-size: 1rem;
  color: hsl(var(--muted-foreground));
  max-width: 32rem;
  margin: 0 auto 2.5rem;
  line-height: 1.6;
}

@media (min-width: 640px) {
  .lede { font-size: 1.125rem; }
}

.cta-row {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  justify-content: center;
}

@media (min-width: 640px) {
  .cta-row { flex-direction: row; }
}

.btn-glow {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.625rem;
  font-size: 0.875rem;
  font-weight: 600;
  height: 2.75rem;
  padding: 0 1.75rem;
  background: hsl(var(--primary));
  color: hsl(var(--primary-foreground));
  box-shadow:
    0 0 0 1px hsl(var(--primary) / 0.3),
    0 2px 0 0 hsl(0 0% 0% / 0.1),
    0 0 28px -4px hsl(var(--primary) / 0.4);
  transition: all 0.25s ease;
  text-decoration: none;
  letter-spacing: -0.01em;
}

.btn-glow:hover {
  box-shadow:
    0 0 0 2px hsl(var(--primary) / 0.45),
    0 2px 0 0 hsl(0 0% 0% / 0.15),
    0 0 42px -2px hsl(var(--primary) / 0.55);
  transform: translateY(-1px);
  filter: brightness(1.1);
}

.btn-ghost {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.625rem;
  font-size: 0.875rem;
  font-weight: 600;
  height: 2.75rem;
  padding: 0 1.75rem;
  border: 1px solid hsl(var(--border));
  background: hsl(var(--card));
  color: hsl(var(--foreground));
  transition: all 0.25s ease;
  text-decoration: none;
  letter-spacing: -0.01em;
}

.btn-ghost:hover {
  border-color: hsl(var(--primary) / 0.35);
  background: hsl(var(--accent));
  color: hsl(var(--accent-foreground));
}
</style>
