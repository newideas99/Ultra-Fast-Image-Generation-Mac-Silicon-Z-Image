<template>
  <div class="terminal-showcase">
    <div class="terminal-frame">
      <div class="terminal-bar">
        <span class="t-circle r"></span>
        <span class="t-circle y"></span>
        <span class="t-circle g"></span>
        <span class="t-title">terminal — ultra-fast-image-gen</span>
      </div>
      <div class="terminal-body">
        <span class="t-line"><span class="t-prompt">$</span> ultra-fast-image-gen --model zimage-turbo</span>
        <span class="t-line dim"><span class="t-arrow">→</span> Loading model (quantized, ~8GB VRAM)...</span>
        <span class="t-line dim"><span class="t-arrow">→</span> Prompt: "a cyberpunk city at sunset"</span>
        <span class="t-line dim"><span class="t-arrow">→</span> Steps: 5 | Size: 512x512 | Device: MPS</span>
        <span class="t-line green"><span class="t-check">✓</span> Generated in 14.2s</span>
        <span class="t-line green"><span class="t-check">✓</span> Saved to output/image_001.png</span>
        <span class="t-line cursor"> </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.terminal-showcase {
  max-width: 42rem;
  margin: 0 auto 6rem;
  padding: 0 1rem;
  animation: fadeUp 0.8s ease-out 0.15s both;
}

.terminal-frame {
  border-radius: 0.75rem;
  overflow: hidden;
  border: 1px solid hsl(var(--border) / 0.35);
  background: hsl(var(--card) / 0.7);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow:
    0 0 0 1px hsl(var(--primary) / 0.05),
    0 8px 32px -8px hsl(0 0% 0% / 0.2),
    inset 0 1px 0 hsl(var(--foreground) / 0.04);
}

.terminal-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem;
  border-bottom: 1px solid hsl(var(--border) / 0.2);
  background: hsl(var(--muted) / 0.5);
}

.t-circle {
  width: 0.65rem;
  height: 0.65rem;
  border-radius: 50%;
  flex-shrink: 0;
}

.t-circle.r { background: hsl(0 78% 60%); }
.t-circle.y { background: hsl(44 90% 55%); }
.t-circle.g { background: hsl(148 55% 48%); }

.t-title {
  margin-left: 0.5rem;
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.6875rem;
  color: hsl(var(--muted-foreground) / 0.5);
}

.terminal-body {
  padding: 1.25rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.t-line {
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.8125rem;
  line-height: 1.65;
  color: hsl(var(--foreground) / 0.85);
}

.t-prompt {
  color: hsl(var(--primary));
  font-weight: 700;
  margin-right: 0.5rem;
}

.t-arrow {
  color: hsl(var(--muted-foreground));
  margin-right: 0.25rem;
}

.t-check {
  color: hsl(148 55% 48%);
  font-weight: 700;
  margin-right: 0.25rem;
}

.t-line.dim {
  color: hsl(var(--muted-foreground) / 0.6);
}

.t-line.green {
  color: hsl(148 55% 48%);
}

.t-line.cursor {
  display: inline-block;
  width: 0.5rem;
  height: 1rem;
  background: hsl(var(--primary));
  animation: blink 1s step-end infinite;
  margin-top: 0.5rem;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

@media (max-width: 640px) {
  .terminal-body { padding: 1rem; }
  .t-line { font-size: 0.75rem; }
}
</style>
