// Tracking-camera auto-exposure (runs in the device worker).

/** Proportional exposure controller; returns a new exposure value or null when in range. */
export function autoExposureStep(mean: number, exposure: number, target = 105): number | null {
  if (mean <= 0) return null;
  const ratio = target / mean;
  if (Math.abs(ratio - 1) < 0.1) return null;
  const next = exposure * Math.min(1.6, Math.max(0.6, Math.pow(ratio, 0.6)));
  return Math.max(0x40, Math.min(0x3000, Math.round(next)));
}
