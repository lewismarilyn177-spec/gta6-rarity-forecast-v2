// GTA6 Trophy Rarity Forecast
// Model: log-log linear trend per trophy (ln rarity vs ln hours since release),
// pooled median slope with shrinkage, tanh-capped projection, logistic quantiles.
// Launch-night stale-read prior: if the case cutoff is <24h after release and every
// trophy's history shows at most one distinct non-null rarity value, the Exophase
// rarity table is stale (pool hasn't refreshed yet). The projected center is shifted
// down by 1.55 in log space and sigma is increased by 0.35 in quadrature with a
// 0.6 floor, accounting for dilution when the tracked-player pool expands.

interface Case {
  schema: string;
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: { id: number; name: string; description: string; tier: string; points: number; secret: boolean }[];
  history: { slot: string; at: string; players_tracked: number; trophies_earned: number; rates: (number | null)[] }[];
  notes?: string;
}

interface OutputTrophy {
  id: number;
  q: [number, number, number, number, number];
}

function parseISO(dateStr: string): Date {
  return new Date(dateStr);
}

function hoursSinceRelease(dateStr: string, releaseDate: Date): number {
  const d = parseISO(dateStr);
  const ms = d.getTime() - releaseDate.getTime();
  return Math.max(ms / 3_600_000, 1e-6);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function olsSlope(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const xMean = xs.reduce((a, b) => a + b, 0) / n;
  const yMean = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xMean;
    num += dx * (ys[i] - yMean);
    den += dx * dx;
  }
  return den === 0 ? 0 : num / den;
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

function processCase(c: Case): OutputTrophy[] {
  const releaseDate = new Date(c.game.release_date + "T00:00:00Z");
  const atHours = hoursSinceRelease(c.at, releaseDate);
  const targetHours = hoursSinceRelease(c.target, releaseDate);
  const horizonDays = c.horizon_minutes / 60 / 24;

  const T = c.trophies.length;
  const H = c.history.length;

  // Build distinct (x, y) series per trophy
  const series: { xs: number[]; ys: number[]; lastRarity: number; lastHours: number }[] = [];
  for (let ti = 0; ti < T; ti++) {
    const seen = new Set<number>();
    const xs: number[] = [];
    const ys: number[] = [];
    let lastRarity = 0;
    let lastHours = 0;
    for (let hi = 0; hi < H; hi++) {
      const r = c.history[hi].rates[ti];
      if (r === null) continue;
      const clamped = clamp(r, 0.01, 99.99);
      const key = Math.round(clamped * 100);
      if (seen.has(key)) continue;
      seen.add(key);
      const h = hoursSinceRelease(c.history[hi].at, releaseDate);
      xs.push(Math.log(h));
      ys.push(Math.log(clamped / 100));
      lastRarity = clamped;
      lastHours = h;
    }
    series.push({ xs, ys, lastRarity, lastHours });
  }

  // Step 2: fit slopes, compute pooled median
  const ownSlopes: number[] = [];
  const hasSlope: boolean[] = [];
  for (let ti = 0; ti < T; ti++) {
    const { xs, ys } = series[ti];
    if (xs.length >= 3) {
      const s = olsSlope(xs, ys);
      ownSlopes.push(s);
      hasSlope.push(true);
    } else {
      hasSlope.push(false);
    }
  }
  const pooledSlope = ownSlopes.length > 0 ? clamp(median(ownSlopes), 0, 2.5) : 0;
  const anySlope = ownSlopes.length > 0;

  // Step 3: shrink per trophy
  const blendedSlopes: number[] = [];
  let slopeIdx = 0;
  for (let ti = 0; ti < T; ti++) {
    if (!hasSlope[ti]) {
      blendedSlopes.push(0);
      continue;
    }
    const n = series[ti].xs.length;
    const w = Math.min(0.75, n / 6);
    const blended = clamp(w * pooledSlope + (1 - w) * ownSlopes[slopeIdx], 0, 2.6);
    blendedSlopes.push(blended);
    slopeIdx++;
  }

  // Detect stale launch read: at < 24h after release AND every trophy has <= 1 distinct non-null value
  const isStaleLaunchRead = atHours < 24 && series.every(s => s.xs.length <= 1);

  // Step 4 & 5: project and compute quantiles
  const CAP = 3.0;
  const Z = [-1.645, -0.674, 0, 0.674, 1.645];
  const baseSigma = 0.6 + 0.25 * Math.log1p(horizonDays);
  const launchNightRefresh = (atHours <= 24 && !anySlope) ? 0.4 : 0;
  let totalSigma = clamp(Math.sqrt(baseSigma * baseSigma + launchNightRefresh * launchNightRefresh), 0.45, 1.30);

  const results: OutputTrophy[] = [];
  for (let ti = 0; ti < T; ti++) {
    const { lastRarity, lastHours } = series[ti];
    const slope = blendedSlopes[ti];

    if (lastRarity >= 99.5) {
      results.push({ id: c.trophies[ti].id, q: [100, 100, 100, 100, 100] });
      continue;
    }

    const yLast = Math.log(lastRarity / 100);
    const dt = Math.log(targetHours) - Math.log(lastHours);
    let yTarget = yLast + CAP * Math.tanh(slope * dt / CAP);
    let sigma = totalSigma;

    if (isStaleLaunchRead) {
      yTarget -= 1.55;
      sigma = Math.sqrt(sigma * sigma + 0.35 * 0.35);
      sigma = Math.max(sigma, 0.6);
    }

    const medianR = Math.exp(yTarget) * 100;

    const qs = Z.map(z => {
      const yq = yTarget + z * sigma;
      return clamp(100 / (1 + Math.exp(-yq)), 0, 100);
    }) as [number, number, number, number, number];

    // Round and enforce non-decreasing
    const rounded = qs.map(round4);
    for (let i = 1; i < 5; i++) {
      if (rounded[i] < rounded[i - 1]) rounded[i] = rounded[i - 1];
    }
    results.push({ id: c.trophies[ti].id, q: rounded as [number, number, number, number, number] });
  }

  return results;
}

async function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";

  const fs = await import("fs");
  const path = await import("path");

  fs.mkdirSync(outputDir, { recursive: true });

  const files = fs.readdirSync(inputDir)
    .filter(f => f.endsWith(".in"))
    .sort();

  for (const file of files) {
    const content = fs.readFileSync(path.join(inputDir, file), "utf8");
    const c: Case = JSON.parse(content);
    const trophies = processCase(c);
    const out = { trophies };
    fs.writeFileSync(path.join(outputDir, file.replace(".in", ".out")), JSON.stringify(out));
  }
}

main().catch(e => { console.error(e); process.exit(1); });