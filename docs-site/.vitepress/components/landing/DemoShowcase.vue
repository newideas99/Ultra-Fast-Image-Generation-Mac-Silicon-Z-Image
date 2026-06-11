<template>
  <div class="demo-showcase">
    <div class="section-header">
      <span class="eyebrow">The Experience</span>
      <h2 class="heading">Type a prompt. Get an image.</h2>
      <p class="lede">OpenCode agents call Ultra Fast Image Gen locally — no cloud, no API keys, no latency.</p>
    </div>

    <div class="demo-split">
      <div class="demo-chat">
        <div class="chat-titlebar">
          <span class="ct-dot r"></span>
          <span class="ct-dot y"></span>
          <span class="ct-dot g"></span>
          <span class="ct-label">opencode — ultra-fast-image-gen</span>
        </div>
        <div class="chat-body">
          <div class="chat-msg user">
            <span class="msg-marker">▸</span>
            <span class="msg-text">generate a hero banner for my SaaS landing page — dark theme, neon green accents, abstract geometric shapes</span>
          </div>
          <div class="chat-msg ai thinking">
            <span class="msg-marker">◉</span>
            <span class="msg-text">Loading model zimage-turbo (quantized, ~8GB)…</span>
          </div>
          <div class="chat-msg ai thinking">
            <span class="msg-marker">◉</span>
            <span class="msg-text">Prompt set. Steps: 5 · Size: 1024x512 · Device: MPS</span>
          </div>
          <div class="chat-msg ai thinking">
            <span class="msg-marker">◉</span>
            <span class="msg-text">Generating… ████████████████ 100%</span>
          </div>
          <div class="chat-msg ai done">
            <span class="msg-marker">✓</span>
            <span class="msg-text">Generated in 14.2s → saved to public/images/hero.png</span>
          </div>
        </div>
      </div>

      <div class="demo-result">
        <div class="result-label">Generated Output</div>
        <div class="result-image-wrap">
          <div class="result-image">
            <svg viewBox="0 0 1024 512" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#0a0f0a"/>
                  <stop offset="60%" stop-color="#061208"/>
                  <stop offset="100%" stop-color="#0a1a0c"/>
                </linearGradient>
                <linearGradient id="glow" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stop-color="hsl(150 65% 45% / 0.3)"/>
                  <stop offset="50%" stop-color="hsl(150 65% 45% / 0.08)"/>
                  <stop offset="100%" stop-color="transparent"/>
                </linearGradient>
                <linearGradient id="accentLine" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stop-color="hsl(150 65% 45%)"/>
                  <stop offset="100%" stop-color="hsl(150 65% 45% / 0.1)"/>
                </linearGradient>
              </defs>
              <rect width="1024" height="512" fill="url(#bg)"/>
              <rect x="0" y="400" width="1024" height="112" fill="url(#glow)"/>
              
              <!-- Geometric shapes -->
              <polygon points="620,180 680,120 740,180 680,240" fill="none" stroke="hsl(150 65% 45% / 0.25)" stroke-width="1"/>
              <polygon points="660,200 680,170 700,200 680,230" fill="none" stroke="hsl(150 65% 45% / 0.4)" stroke-width="1"/>
              <circle cx="680" cy="200" r="80" fill="none" stroke="hsl(150 65% 45% / 0.15)" stroke-width="0.5"/>
              <circle cx="680" cy="200" r="110" fill="none" stroke="hsl(150 65% 45% / 0.08)" stroke-width="0.5"/>
              <circle cx="680" cy="200" r="140" fill="none" stroke="hsl(150 65% 45% / 0.04)" stroke-width="0.5"/>
              
              <!-- Accent lines -->
              <line x1="120" y1="340" x2="320" y2="340" stroke="url(#accentLine)" stroke-width="2"/>
              <line x1="200" y1="360" x2="380" y2="360" stroke="url(#accentLine)" stroke-width="1" opacity="0.5"/>
              
              <!-- Text -->
              <text x="160" y="280" font-family="system-ui, sans-serif" font-size="42" font-weight="800" fill="hsl(150 20% 90%)" letter-spacing="-1">Build Faster</text>
              <text x="160" y="320" font-family="system-ui, sans-serif" font-size="18" fill="hsl(150 10% 60% / 0.8)">Local AI image generation for your projects</text>
              
              <!-- Small dot grid accent -->
              <g opacity="0.15">
                <circle cx="900" cy="100" r="1.5" fill="hsl(150 65% 45%)"/>
                <circle cx="920" cy="120" r="1.5" fill="hsl(150 65% 45%)"/>
                <circle cx="940" cy="100" r="1.5" fill="hsl(150 65% 45%)"/>
                <circle cx="910" cy="140" r="1.5" fill="hsl(150 65% 45%)"/>
                <circle cx="930" cy="140" r="1.5" fill="hsl(150 65% 45%)"/>
              </g>
            </svg>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.demo-showcase {
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
  max-width: 34rem;
  margin: 0 auto;
  line-height: 1.6;
}

@media (min-width: 640px) {
  .lede { font-size: 1.125rem; }
}

.demo-split {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

@media (min-width: 1024px) {
  .demo-split {
    flex-direction: row;
  }
  .demo-split > * {
    flex: 1;
  }
}

/* Chat panel */
.demo-chat {
  border-radius: 0.875rem;
  overflow: hidden;
  border: 1px solid hsl(var(--border) / 0.4);
  background: hsl(var(--card));
  box-shadow: 0 4px 16px -4px hsl(0 0% 0% / 0.1);
}

.chat-titlebar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1rem;
  border-bottom: 1px solid hsl(var(--border) / 0.25);
  background: hsl(var(--muted) / 0.5);
}

.ct-dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 50%;
}

.ct-dot.r { background: hsl(0 78% 60%); }
.ct-dot.y { background: hsl(44 90% 55%); }
.ct-dot.g { background: hsl(148 55% 48%); }

.ct-label {
  margin-left: 0.375rem;
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.625rem;
  color: hsl(var(--muted-foreground) / 0.5);
  letter-spacing: 0.03em;
}

.chat-body {
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.625rem;
  min-height: 260px;
}

.chat-msg {
  display: flex;
  gap: 0.5rem;
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.75rem;
  line-height: 1.55;
  animation: msgIn 0.3s ease-out;
}

.chat-msg:nth-child(2) { animation-delay: 0.3s; }
.chat-msg:nth-child(3) { animation-delay: 0.6s; }
.chat-msg:nth-child(4) { animation-delay: 0.9s; }
.chat-msg:nth-child(5) { animation-delay: 1.2s; }

.msg-marker {
  flex-shrink: 0;
  width: 1.125rem;
  font-weight: 700;
  font-size: 0.6875rem;
  line-height: 1.55;
}

.chat-msg.user .msg-marker {
  color: hsl(var(--primary));
}

.chat-msg.ai.thinking .msg-marker {
  color: hsl(var(--muted-foreground) / 0.5);
}

.chat-msg.ai.done .msg-marker {
  color: hsl(148 55% 48%);
}

.chat-msg.user .msg-text {
  color: hsl(var(--foreground) / 0.9);
}

.chat-msg.ai.thinking .msg-text {
  color: hsl(var(--muted-foreground) / 0.55);
}

.chat-msg.ai.done .msg-text {
  color: hsl(148 55% 48%);
}

@keyframes msgIn {
  from { opacity: 0; transform: translateX(-6px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* Result panel */
.demo-result {
  display: flex;
  flex-direction: column;
}

.result-label {
  font-size: 0.6875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: hsl(var(--muted-foreground) / 0.5);
  margin-bottom: 0.5rem;
  padding-left: 0.25rem;
}

.result-image-wrap {
  border-radius: 0.875rem;
  overflow: hidden;
  border: 1px solid hsl(var(--primary) / 0.2);
  box-shadow:
    0 0 0 1px hsl(var(--primary) / 0.08),
    0 12px 40px -8px hsl(var(--primary) / 0.12);
  flex: 1;
  position: relative;
}

.result-image-wrap::after {
  content: '';
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(135deg, hsl(var(--primary) / 0.03) 0%, transparent 50%);
}

.result-image {
  width: 100%;
  height: 100%;
  min-height: 260px;
}

.result-image svg {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
}
</style>
