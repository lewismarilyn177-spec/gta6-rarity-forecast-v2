// Refresh-reconstruction approach: collapse duplicate history reads into distinct refresh
// states, infer per-trophy earn hazards from player deltas, then project earned counts forward.

interface InputCase {
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

interface OutputRow { id: number; q: [number, number, number, number, number]; }

const EPS = 1e-6;
const LOGIT_CLAMP = [EPS, 1 - EPS] as const;

function toLogit(pct: number): number {
  const p = Math.min(Math.max(pct / 100, LOGIT_CLAMP[0]), LOGIT_CLAMP[1]);
  return Math.log(p / (1 - p));
}

function fromLogit(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function median(arr: number[]): number {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length & 1 ? s[mid] : (s[mid - 1] + s[mid]) * 0.5;
}

function weightedMedian(pairs: [number, number][]): number {
  if (!pairs.length) return 0;
  const s = [...pairs].sort((a, b) => a[0] - b[0]);
  const total = s.reduce((a, b) => a + b[1], 0);
  if (total <= 0) return median(s.map(([v]) => v));
  let acc = 0;
  for (const [v, w] of s) {
    acc += w;
    if (acc >= total * 0.5) return v;
  }
  return s[s.length - 1][0];
}

function weightedMAD(values: number[], weights: number[], center: number): number {
  const devs = values.map((v, i) => [Math.abs(v - center), weights[i]] as [number, number]);
  return weightedMedian(devs);
}

interface RefreshState {
  time: number;
  players: number;
  rates: number[];
}

function extractStates(kase: InputCase): RefreshState[] {
  const T = kase.trophies.length;
  const states: RefreshState[] = [];
  for (const h of kase.history) {
    if (h.rates.some(r => r === null)) continue;
    const t = Date.parse(h.at);
    const last = states[states.length - 1];
    if (last && last.players === h.players_tracked && last.rates.every((r, i) => r === h.rates[i])) continue;
    states.push({ time: t, players: h.players_tracked, rates: h.rates.slice(0, T) });
  }
  return states;
}

interface SaturationFit {
  Pinf: number;
  A: number;
  lam: number;
  t0: number;
}

function fitSaturation(points: { t: number; p: number }[]): SaturationFit | null {
  if (points.length < 3) return null;
  const t0 = points[0].t;
  const shifted = points.map(p => ({ dt: p.t - t0, p: p.p }));
  let best: SaturationFit & { sse: number } | null = null;

  for (let lam = 2e-6; lam <= 1e-2; lam *= 1.15) {
    let Su = 0, Sp = 0, Suu = 0, Sup = 0, n = 0;
    for (const { dt, p } of shifted) {
      const u = Math.exp(-lam * dt);
      Su += u; Sp += p; Suu += u * u; Sup += u * p; n++;
    }
    const denom = n * Suu - Su * Su;
    if (denom <= 1e-12) continue;
    const m = (n * Sup - Su * Sp) / denom;
    const c = (Sp - m * Su) / n;
    const A = -m;
    if (A <= 0) continue;
    let sse = 0;
    for (const { dt, p } of shifted) {
      const pred = c - A * Math.exp(-lam * dt);
      sse += (p - pred) ** 2;
    }
    if (!best || sse < best.sse) best = { Pinf: c, A, lam, t0, sse };
  }
  if (!best || best.Pinf <= 0 || best.A <= 0) return null;
  return best;
}

function playersAtTime(fit: SaturationFit, t: number): number {
  return Math.max(1, fit.Pinf - fit.A * Math.exp(-fit.lam * (t - fit.t0)));
}

function solveCase(kase: InputCase): OutputRow[] {
  const T = kase.trophies.length;
  const targetMs = Date.parse(kase.target);
  const states = extractStates(kase);

  if (states.length === 0) {
    const fallbackRates: number[] = [];
    for (let i = 0; i < T; i++) {
      let rate = 0;
      for (let h = kase.history.length - 1; h >= 0; h--) {
        const r = kase.history[h].rates[i];
        if (r !== null && r !== undefined) { rate = r; break; }
      }
      fallbackRates.push(rate);
    }
    return kase.trophies.map((tr, i) => buildWideFanRow(tr.id, fallbackRates[i] ?? 0));
  }

  const last = states[states.length - 1];
  const sawRefresh = states.length >= 2;

  if (!sawRefresh) {
    return kase.trophies.map((tr, i) => buildLaunchRow(tr.id, last.rates[i]));
  }

  const earned = states.map(s =>
    s.rates.map(r => Math.max(0, Math.round(r / 100 * s.players)))
  );
  for (let k = 1; k < states.length; k++) {
    for (let i = 0; i < T; i++) {
      if (earned[k][i] < earned[k - 1][i]) earned[k][i] = earned[k - 1][i];
    }
  }

  const hazardVals: number[][] = Array.from({ length: T }, () => []);
  const hazardWts: number[][] = Array.from({ length: T }, () => []);
  for (let k = 1; k < states.length; k++) {
    const dP = states[k].players - states[k - 1].players;
    if (dP <= 0) continue;
    for (let i = 0; i < T; i++) {
      const dE = earned[k][i] - earned[k - 1][i];
      hazardVals[i].push(dE / dP);
      hazardWts[i].push(dP);
    }
  }

  const poolPairs: [number, number][] = [];
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < hazardVals[i].length; j++) {
      poolPairs.push([hazardVals[i][j], hazardWts[i][j]]);
    }
  }
  const pooledHazard = weightedMedian(poolPairs);

  const shrunkHazard: number[] = [];
  const hazardSigma: number[] = [];
  for (let i = 0; i < T; i++) {
    const n = hazardVals[i].length;
    if (n === 0) {
      shrunkHazard.push(pooledHazard);
      hazardSigma.push(Math.max(0.01, pooledHazard * 0.8));
      continue;
    }
    const pairs = hazardVals[i].map((v, j) => [v, hazardWts[i][j]] as [number, number]);
    const wMed = weightedMedian(pairs);
    const shrink = n / (n + 3);
    const h = wMed * shrink + pooledHazard * (1 - shrink);
    const mad = weightedMAD(hazardVals[i], hazardWts[i], wMed);
    shrunkHazard.push(h);
    hazardSigma.push(clamp(1.4826 * mad, 0.01, 0.6));
  }

  const fit = fitSaturation(states.map(s => ({ t: s.time, p: s.players })));

  const intervals: number[] = [];
  for (let k = 1; k < states.length; k++) intervals.push(states[k].time - states[k - 1].time);
  const cadence = intervals.length ? median(intervals) : Math.max(600_000, targetMs - last.time);

  const futureTimes: number[] = [];
  let t = last.time;
  while (t + cadence < targetMs && futureTimes.length < 500) {
    t += cadence;
    futureTimes.push(t);
  }

  let targetPlayers: number;
  if (fit) {
    targetPlayers = playersAtTime(fit, targetMs);
  } else {
    const lastGrowth = states.length >= 2
      ? states[states.length - 1].players - states[states.length - 2].players
      : 0;
    targetPlayers = last.players + lastGrowth * futureTimes.length;
  }
  targetPlayers = Math.max(last.players, Math.round(targetPlayers));

  const lastEarned = earned[earned.length - 1];
  const centerEarned: number[] = [];
  for (let i = 0; i < T; i++) {
    let e = lastEarned[i];
    let pPrev = last.players;
    for (const rt of futureTimes) {
      const pNext = fit ? playersAtTime(fit, rt) : last.players + (targetPlayers - last.players) * (futureTimes.indexOf(rt) + 1) / Math.max(1, futureTimes.length);
      const dP = Math.max(0, Math.round(pNext) - pPrev);
      e += Math.max(0, Math.round(dP * shrunkHazard[i]));
      pPrev = Math.round(pNext);
    }
    centerEarned.push(e);
  }

  const rows: OutputRow[] = [];
  const QUANTILE_Z = [-1.645, -0.674, 0, 0.674, 1.645];
  for (let i = 0; i < T; i++) {
    const trophy = kase.trophies[i];
    const cRate = targetPlayers > 0 ? (centerEarned[i] / targetPlayers) * 100 : 0;
    if (last.rates[i] >= 99.995) {
      rows.push({ id: trophy.id, q: [100, 100, 100, 100, 100] });
      continue;
    }
    if (last.rates[i] <= 0.0005) {
      rows.push({ id: trophy.id, q: [0, 0, 0, 0, 0] });
      continue;
    }

    const fracNew = 1 - last.players / targetPlayers;
    const relSigma = clamp(hazardSigma[i] / Math.max(1e-4, shrunkHazard[i]), 0.05, 1.5);
    const logitCenter = toLogit(cRate);
    const sigLogit = clamp(
      relSigma * fracNew * Math.abs(logitCenter) * 0.5 + 0.05,
      0.06,
      2.0
    );

    const q = QUANTILE_Z.map(z => {
      const val = fromLogit(logitCenter + z * sigLogit) * 100;
      return round4(clamp(val, 0, 100));
    });
    for (let j = 1; j < 5; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    rows.push({ id: trophy.id, q: q as [number, number, number, number, number] });
  }
  return rows;
}

function buildWideFanRow(id: number, rate: number): OutputRow {
  if (rate >= 99.995) return { id, q: [100, 100, 100, 100, 100] };
  if (rate <= 0.0005) return { id, q: [0, 0, 0, 0, 0] };
  const QUANTILE_Z = [-1.645, -0.674, 0, 0.674, 1.645];
  const l = toLogit(rate);
  const q = QUANTILE_Z.map(z => round4(clamp(fromLogit(l + z * 1.5) * 100, 0, 100)));
  for (let j = 1; j < 5; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
  return { id, q: q as [number, number, number, number, number] };
}

function buildLaunchRow(id: number, rate: number): OutputRow {
  if (rate >= 99.995) return { id, q: [100, 100, 100, 100, 100] };
  if (rate <= 0.0005) return { id, q: [0, 0, 0, 0, 0] };
  const QUANTILE_Z = [-1.645, -0.674, 0, 0.674, 1.645];
  const l = toLogit(rate) - 0.75;
  const q = QUANTILE_Z.map(z => round4(clamp(fromLogit(l + z * 1.2) * 100, 0, 100)));
  for (let j = 1; j < 5; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
  return { id, q: q as [number, number, number, number, number] };
}

async function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR!;
  const outDir = process.env.MAKEFUN_OUTPUT_DIR!;
  const { mkdirSync, readdirSync, readFileSync, writeFileSync } = await import("node:fs");
  const { join } = await import("node:path");

  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir).filter(f => f.endsWith(".in")).sort();
  for (const f of files) {
    const kase: InputCase = JSON.parse(readFileSync(join(inDir, f), "utf8"));
    const out = JSON.stringify({ trophies: solveCase(kase) });
    writeFileSync(join(outDir, f.replace(/\.in$/, ".out")), out);
  }
}

main().catch(e => { console.error(e); process.exit(1); });