<template>
  <div class="feature-bento">
    <div class="section-header">
      <span class="eyebrow">Models</span>
      <h2 class="heading">Every model, one tool</h2>
      <p class="lede">Optimized for your hardware — from 8GB laptops to 48GB workstations.</p>
    </div>

    <div class="bento-grid">
      <Card class="bento-card">
        <CardHeader class="bento-header">
          <span class="card-icon-box">⚡</span>
          <CardTitle class="card-heading">Z-Image Turbo</CardTitle>
          <CardDescription class="card-sub">Ultra-fast text-to-image</CardDescription>
        </CardHeader>
        <CardContent>
          <span class="card-body">~8GB VRAM. 5 steps. Fastest generation for rapid prototyping and standard tasks.</span>
          <Badge class="card-badge" variant="secondary">Fastest</Badge>
        </CardContent>
      </Card>

      <Card class="bento-card">
        <CardHeader class="bento-header">
          <span class="card-icon-box">🎨</span>
          <CardTitle class="card-heading">FLUX.2-klein-4B</CardTitle>
          <CardDescription class="card-sub">Text-to-image + Image editing</CardDescription>
        </CardHeader>
        <CardContent>
          <span class="card-body">&lt;8GB @ 512px, &lt;16GB @ 1024px. Supports up to 6 reference images for precise control.</span>
          <Badge class="card-badge" variant="secondary">Low Memory</Badge>
        </CardContent>
      </Card>

      <Card class="bento-card">
        <CardHeader class="bento-header">
          <span class="card-icon-box">✨</span>
          <CardTitle class="card-heading">FLUX.2-klein-9B</CardTitle>
          <CardDescription class="card-sub">Text-to-image + Image editing</CardDescription>
        </CardHeader>
        <CardContent>
          <span class="card-body">~12GB @ 512px, ~20GB @ 1024px. Highest quality output for professional and demanding tasks.</span>
          <Badge class="card-badge" variant="secondary">High Quality</Badge>
        </CardContent>
      </Card>

      <Card class="bento-card">
        <CardHeader class="bento-header">
          <span class="card-icon-box">📐</span>
          <CardTitle class="card-heading">MFLUX HS 2K</CardTitle>
          <CardDescription class="card-sub">Uncensored 2K text-to-image</CardDescription>
        </CardHeader>
        <CardContent>
          <span class="card-body">~7GB RSS @ 2K. Fastest current 2K FLUX lane with uncensored GGUF TE for high-res output.</span>
          <Badge class="card-badge" variant="secondary">2K Resolution</Badge>
        </CardContent>
      </Card>

      <Card class="bento-card">
        <CardHeader class="bento-header">
          <span class="card-icon-box">🍎</span>
          <CardTitle class="card-heading">Anima Turbo AIO Q4</CardTitle>
          <CardDescription class="card-sub">Metal runner, baked Turbo LoRA</CardDescription>
        </CardHeader>
        <CardContent>
          <span class="card-body">~3GB model + unified memory. ~16s @ 512x768 / 8 steps. Optimized for Apple Silicon devices.</span>
          <Badge class="card-badge" variant="secondary">Apple Silicon</Badge>
        </CardContent>
      </Card>

      <Card class="bento-card">
        <CardHeader class="bento-header">
          <span class="card-icon-box">🔌</span>
          <CardTitle class="card-heading">MCP Server</CardTitle>
          <CardDescription class="card-sub">AI agent integration</CardDescription>
        </CardHeader>
        <CardContent>
          <span class="card-body">Built-in Model Context Protocol server. Let AI coding agents generate and edit project assets locally.</span>
          <Badge class="card-badge" variant="secondary">Developer Tools</Badge>
        </CardContent>
      </Card>
    </div>
  </div>
</template>

<style scoped>
.feature-bento {
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

.bento-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}

@media (min-width: 768px) {
  .bento-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (min-width: 1024px) {
  .bento-grid { grid-template-columns: repeat(3, 1fr); }
}

.bento-card {
  position: relative;
  border-radius: 0.875rem;
  border: 1px solid hsl(var(--border) / 0.45);
  background: hsl(var(--card));
  transition: all 0.35s ease;
  overflow: hidden;
}

.bento-card::before {
  content: '';
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0;
  background: radial-gradient(600px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), hsl(var(--primary) / 0.06), transparent 40%);
  transition: opacity 0.35s ease;
}

.bento-card:hover::before {
  opacity: 1;
}

.bento-card:hover {
  border-color: hsl(var(--primary) / 0.25);
  transform: translateY(-2px);
  box-shadow:
    0 0 0 1px hsl(var(--primary) / 0.08),
    0 12px 40px -8px hsl(var(--primary) / 0.1),
    0 4px 12px -4px hsl(0 0% 0% / 0.1);
}

.bento-header {
  padding: 1.25rem 1.25rem 0.5rem;
}

.card-icon-box {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 0.625rem;
  background: hsl(var(--primary) / 0.08);
  font-size: 1.25rem;
  margin-bottom: 0.75rem;
  border: 1px solid hsl(var(--primary) / 0.1);
}

.card-heading {
  font-size: 1.0625rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin-bottom: 0.25rem;
  color: hsl(var(--foreground));
}

.card-sub {
  font-size: 0.8125rem;
  color: hsl(var(--muted-foreground));
}

.card-body {
  font-size: 0.8125rem;
  color: hsl(var(--muted-foreground) / 0.85);
  line-height: 1.55;
  display: block;
}

.card-badge {
  margin-top: 0.875rem;
}
</style>
