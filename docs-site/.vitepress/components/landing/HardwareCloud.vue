<template>
  <div class="hardware-cloud">
    <span class="cloud-label">Runs on</span>
    <div class="cloud-items">
      <span class="cloud-item">Apple Silicon</span>
      <span class="cloud-sep"></span>
      <span class="cloud-item">NVIDIA CUDA</span>
      <span class="cloud-sep"></span>
      <span class="cloud-item">MCP Protocol</span>
      <span class="cloud-sep"></span>
      <span class="cloud-item">Python 3.11+</span>
    </div>
  </div>
</template>

<style scoped>
.hardware-cloud {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  padding: 2rem 0 1.5rem;
}

.cloud-label {
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: hsl(var(--muted-foreground) / 0.45);
}

.cloud-items {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 0.25rem 1.125rem;
}

.cloud-item {
  font-size: 0.8125rem;
  color: hsl(var(--muted-foreground) / 0.55);
  font-weight: 500;
  transition: color 0.3s ease;
  cursor: default;
}

.cloud-item:hover {
  color: hsl(var(--foreground) / 0.8);
}

.cloud-sep {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: hsl(var(--border) / 0.6);
  display: none;
}

@media (min-width: 640px) {
  .cloud-sep { display: inline-block; }
}
</style>
