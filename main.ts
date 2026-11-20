// GTA VI Trophy Rarity Forecast — refresh-reconstruction dilution forecaster v2
//
// Approach (matches what the top of the leaderboard does):
//  - Exophase only refreshes counts on its own schedule: between refreshes every
//    rate and players_tracked is constant. A read therefore reveals the state as
//    of the last refresh before it.
//  - For each observed refresh state k we reconstruct integer earned counts:
//      earned_i ~= rate_i% * players_k  (rounded, monotone across refreshes)
//  - players_tracked(t) grows on a saturating launch curve; fit
//      p(t) = Pinf - (Pinf - p0) * exp(-lam * (t - t0))
//    by least squares over the observed refresh states, and project the players
//    pool at each future refresh before the target.
//  - Per-trophy hazard: fraction of NEW tracked players who earn trophy i per
//    refresh, estimated as a weighted median of dEarned/dPlayers across
//    consecutive refresh pairs, shrunk toward the pooled tier median.
//  - Center forecast: earned_i(target) = earned_last + sum over predicted
//    refreshes of dP * hazard_i; rate = earned / players(target).
//  - Quantiles: fan in logit space from the residual scatter of the hazard
//    estimates (per-trophy robust sigma, floored), q05..q95.
//  - If the history never saw a refresh (stale launch-night page), fall back to
//    the observed launch dilution if present in the data, else hold the last
//    rate with a launch-night spread.

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

interface OutputTrophy { id: number; q: [number, number, number, number, number]; }

const LOGIT_EPS = 1e-6;
function logit(p: number): number {
  const x = Math.min(Math.max(p, LOGIT_EPS), 1 - LOGIT_EPS);
  return Math.log(x / (1 - x));
}
function expit(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}
function clamp(v: number, lo: number, hi: number): number { return Math.max(lo, Math.min(hi, v)); }
function median(a: number[]): number {
  if (a.length === 0) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function weightedMedian(items: [number, number][]): number {
  // items: [value, weight]
  if (items.length === 0) return 0;
  const s = [...items].sort((a, b) => a[0] - b[0]);
  const tot = s.reduce((acc, it) => acc + it[1], 0);
  if (tot <= 0) return median(s.map((it) => it[0]));
  let acc = 0;
  for (const it of s) {
    acc += it[1];
    if (acc >= tot / 2) return it[0];
  }
  return s[s.length - 1][0];
}
function round4(v: number): number { return Math.round(v * 10000) / 10000; }

// --- players saturation fit: p(t) = Pinf - A * exp(-lam * (t - t0)) ---
// Grid search over lam, closed-form (Pinf, A) via linear least squares on
// u = exp(-lam * t): p = Pinf - A * u.
function fitPlayers(points: { t: number; p: number }[]): { Pinf: number; lam: number; t0: number; A: number } | null {
  if (points.length < 3) return null;
  const t0 = points[0].t;
  const pts = points.map((q) => ({ u0: q.t - t0, p: q.p }));
  let best: { lam: number; Pinf: number; A: number; sse: number } | null = null;
  for (let lam = 0.000002; lam <= 0.01; lam *= 1.15) {
    let su = 0, sp = 0, suu = 0, sup = 0, n = 0;
    for (const q of pts) {
      const u = Math.exp(-lam * q.u0);
      su += u; sp += q.p; suu += u * u; sup += u * q.p; n++;
    }
    const den = n * suu - su * su;
    if (den <= 1e-12) continue;
    // p = Pinf - A*u  ->  regress p on u: p = c + m*u, Pinf = c, A = -m
    const m = (n * sup - su * sp) / den;
    const c = (sp - m * su) / n;
    const A = -m;
    if (A < 0) continue; // needs players below Pinf
    let sse = 0;
    for (const q of pts) {
      const pred = c - A * Math.exp(-lam * q.u0);
      sse += (q.p - pred) * (q.p - pred);
    }
    if (!best || sse < best.sse) best = { lam, Pinf: c, A, sse };
  }
  if (!best || best.Pinf <= 0 || best.A <= 0) return null;
  return { Pinf: best.Pinf, lam: best.lam, t0, A: best.A };
}
function playersAt(fit: { Pinf: number; lam: number; t0: number; A: number }, t: number): number {
  return Math.max(1, fit.Pinf - fit.A * Math.exp(-fit.lam * (t - fit.t0)));
}

function processCase(c: Case): OutputTrophy[] {
  const T = c.trophies.length;
  const targetMs = Date.parse(c.target);
  const atMs = Date.parse(c.at);

  // Distinct refresh states, oldest first.
  const states: { t: number; p: number; rates: number[] }[] = [];
  for (const h of c.history) {
    if (h.rates.some((r) => r === null)) continue;
    const t = Date.parse(h.at);
    const last = states[states.length - 1];
    if (last && last.p === h.players_tracked && last.rates.every((r, i) => r === h.rates[i])) continue;
    states.push({ t, p: h.players_tracked, rates: h.rates.slice(0, T) });
  }

  // Fallback: completely stale history (no distinct state but one).
  if (states.length === 0) {
    const last = c.history[c.history.length - 1];
    const rates = (last?.rates ?? []).map((r) => (r === null ? 0 : r));
    return c.trophies.map((tr, ti) => makeRow(tr.id, [rates[ti] ?? 0], 0, 0.9, 1.645, true));
  }

  const lastState = states[states.length - 1];
  // If no refresh was ever observed (launch-night stale page), the target very
  // likely falls after Exophase's next big dilution refresh that we cannot see:
  // shift the center down in log space and use a wide launch-dilution fan.
  const sawRefresh = states.length >= 2;

  // Refresh times: a state change happens at some read; the refresh slot itself
  // is unknown, model refresh epochs as evenly spaced between observed changes.
  const refreshTs = states.map((s) => s.t);
  const intervals: number[] = [];
  for (let k = 1; k < refreshTs.length; k++) intervals.push(refreshTs[k] - refreshTs[k - 1]);
  const cadence = intervals.length > 0 ? median(intervals) : Math.max(600000, targetMs - atMs);

  // Integer earned-count reconstruction per state.
  const earned: number[][] = states.map((s) =>
    s.rates.map((r) => Math.max(0, Math.round((r / 100) * s.p)))
  );
  // enforce monotone earned across states
  for (let k = 1; k < states.length; k++) {
    for (let i = 0; i < T; i++) if (earned[k][i] < earned[k - 1][i]) earned[k][i] = earned[k - 1][i];
  }

  // Per-trophy hazard: dEarned / dPlayers across consecutive refresh pairs.
  const hazards: number[][] = [];
  const hazardWeights: number[][] = [];
  for (let i = 0; i < T; i++) hazards.push([]);
  for (let i = 0; i < T; i++) hazardWeights.push([]);
  for (let k = 1; k < states.length; k++) {
    const dP = states[k].p - states[k - 1].p;
    if (dP <= 0) continue;
    for (let i = 0; i < T; i++) {
      const dE = earned[k][i] - earned[k - 1][i];
      const hz = dE / dP;
      hazards[i].push(hz);
      hazardWeights[i].push(dP);
    }
  }
  // Pooled median hazard (weighted by dP over all pairs) for shrinkage.
  const poolItems: [number, number][] = [];
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < hazards[i].length; j++) poolItems.push([hazards[i][j], hazardWeights[i][j]]);
  }
  const pooledH = weightedMedian(poolItems);

  // Per-trophy weighted median hazard, shrunk toward pooled (n-weighted).
  const hazShrunk: number[] = [];
  const hazSigma: number[] = [];
  for (let i = 0; i < T; i++) {
    const items: [number, number][] = hazards[i].map((h, j) => [h, hazardWeights[i][j]]);
    const n = hazards[i].length;
    if (n === 0) {
      hazShrunk.push(pooledH);
      hazSigma.push(Math.max(0.02, pooledH * 0.8));
      continue;
    }
    const hMed = weightedMedian(items);
    const shrink = n / (n + 3); // modest shrinkage toward pool
    const h = hMed * shrink + pooledH * (1 - shrink);
    // robust sigma: weighted MAD around hMed, floored
    const devs = hazards[i].map((h2) => Math.abs(h2 - hMed));
    const mad = weightedMedian(devs.map((d, j) => [d, hazardWeights[i][j]]));
    hazSigma.push(clamp(1.4826 * mad, 0.01, 0.6));
    hazShrunk.push(h);
  }

  // Players curve fit over observed states.
  const fit = fitPlayers(states.map((s) => ({ t: s.t, p: s.p })));

  // Predict future refreshes up to target: continue cadence from last observed
  // refresh; each future refresh pulls players along the fitted curve.
  const futureRefreshes: number[] = [];
  let t = lastState.t;
  while (t + cadence < targetMs && futureRefreshes.length < 500) {
    t += cadence;
    futureRefreshes.push(t);
  }

  // Project earned counts and players.
  let pTarget: number;
  if (fit) {
    pTarget = playersAt(fit, targetMs);
  } else {
    // linear dP per cadence from last observed growth
    const growth = states.length >= 2 ? (states[states.length - 1].p - states[states.length - 2].p) : 0;
    pTarget = lastState.p + growth * futureRefreshes.length;
  }
  pTarget = Math.max(lastState.p, Math.round(pTarget));

  const earnedLast = earned[earned.length - 1];
  const centerEarned: number[] = [];
  for (let i = 0; i < T; i++) {
    let e = earnedLast[i];
    let pPrev = lastState.p;
    for (const rt of futureRefreshes) {
      const pK = fit ? playersAt(fit, rt) : lastState.p + (pTarget - lastState.p) * (futureRefreshes.indexOf(rt) + 1) / Math.max(1, futureRefreshes.length);
      const dP = Math.max(0, Math.round(pK) - pPrev);
      e += Math.max(0, Math.round(dP * hazShrunk[i]));
      pPrev = Math.round(pK);
    }
    centerEarned.push(e);
  }

  // Platinum (id 1) is 100 by definition and any rate already 100 stays 100.
  const rows: OutputTrophy[] = [];
  for (let i = 0; i < T; i++) {
    const trophy = c.trophies[i];
    let cRate = pTarget > 0 ? (centerEarned[i] / pTarget) * 100 : 0;
    if (lastState.rates[i] >= 99.995) cRate = 100;
    if (!sawRefresh) {
      // launch-night stale prior: unseen dilution ahead
      rows.push(makeRow(trophy.id, [lastState.rates[i]], 0, 1.2, 1.645, true, 0.75));
      continue;
    }

    // Logit-space fan: center logit +/- z * sigmaLogit, where sigmaLogit comes
    // from hazard scatter propagated through the remaining dilution.
    const remainingDilution = Math.max(1, pTarget / Math.max(1, lastState.p));
    // relative uncertainty in log-rate space: hazard sigma vs hazard, scaled by
    // the fraction of the pool still to come, floor for estimation noise.
    const fracNew = 1 - 1 / remainingDilution;
    const relSigma = clamp(hazSigma[i] / Math.max(1e-4, hazShrunk[i]), 0.05, 1.5);
    const sigLogit = clamp(
      relSigma * fracNew * Math.abs(logit(cRate / 100)) * 0.5 + 0.05,
      0.06,
      2.0
    );
    rows.push(makeRow(trophy.id, null, cRate, sigLogit, 1.645, false));
  }
  return rows;
}

function makeRow(id: number, staleRates: number[] | null, cRate: number, sigLogit: number, z: number, stale: boolean, downShift = 0): OutputTrophy {
  const levels = [0.05, 0.25, 0.5, 0.75, 0.95];
  const zs = [-z, -0.674, 0, 0.674, z];
  const r0 = stale && staleRates ? (staleRates[0] ?? 0) : cRate;
  if (r0 >= 99.995) return { id, q: [100, 100, 100, 100, 100] };
  if (r0 <= 0.0005) return { id, q: [0, 0, 0, 0, 0] };
  const q = levels.map((_, j) => {
    let rate: number;
    if (stale && staleRates) {
      const r = staleRates[0] ?? 0;
      const l = logit(r / 100) - downShift;
      rate = expit(l + zs[j] * sigLogit) * 100;
    } else {
      rate = expit(logit(cRate / 100) + zs[j] * sigLogit) * 100;
    }
    return rate;
  });
  // enforce monotone + bounds
  for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
  for (let j = 0; j < q.length; j++) q[j] = round4(clamp(q[j], 0, 100));
  return { id, q: q as [number, number, number, number, number] };
}

async function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  const fs = await import("node:fs");
  const path = await import("node:path");
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs.readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  for (const f of files) {
    const c: Case = JSON.parse(fs.readFileSync(path.join(inDir, f), "utf8"));
    const rows = processCase(c);
    const out = JSON.stringify({ trophies: rows });
    fs.writeFileSync(path.join(outDir, f.replace(/\.in$/, ".out")), out);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
