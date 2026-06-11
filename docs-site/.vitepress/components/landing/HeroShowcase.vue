<template>
  <div class="hero-showcase">
    <div class="hero-mockup">
      <div class="mockup-glow"></div>
      <img src="/screenshot.png" alt="Ultra Fast Image Gen UI" />
    </div>
  </div>
</template>

<style scoped>
.hero-showcase {
  max-width: 52rem;
  margin: 2.5rem auto 4rem;
  padding: 0 1rem;
  animation: fadeUp 0.7s ease-out;
}

.hero-mockup {
  position: relative;
  border-radius: 0.875rem;
  overflow: hidden;
  background: hsl(var(--card));
  border: 1px solid hsl(var(--border) / 0.5);
  box-shadow:
    0 0 0 1px hsl(var(--primary) / 0.08),
    0 8px 32px -4px hsl(var(--primary) / 0.1),
    0 2px 8px -2px hsl(0 0% 0% / 0.15);
  transition: box-shadow 0.5s ease, border-color 0.5s ease;
}

.hero-mockup:hover {
  border-color: hsl(var(--primary) / 0.3);
  box-shadow:
    0 0 0 1px hsl(var(--primary) / 0.15),
    0 12px 48px -4px hsl(var(--primary) / 0.16),
    0 4px 12px -2px hsl(0 0% 0% / 0.2);
}

.mockup-glow {
  position: absolute;
  inset: 0;
  z-index: 1;
  pointer-events: none;
  background: linear-gradient(135deg, hsl(var(--primary) / 0.04) 0%, transparent 50%, hsl(var(--primary) / 0.02) 100%);
}

.hero-mockup img {
  width: 100%;
  display: block;
  position: relative;
  z-index: 0;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
