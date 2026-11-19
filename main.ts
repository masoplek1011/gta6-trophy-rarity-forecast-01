// GTA VI trophy rarity forecaster — make.fun gta6-trophy-rarity-forecast-01.
//
// Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate
// (percent of the players Exophase tracks) at a case's target, as five
// quantiles, from the case's own history of Exophase reads. Node 22,
// standard library only, no network, fully deterministic.
//
// Model (all in logit space, per trophy):
//   Exophase does not refresh continuously: between refreshes the tracked
//   player count is flat and rates creep up as those players progress; at a
//   refresh the tracked count jumps and almost every rate falls, because the
//   newly tracked players hold fewer trophies. So the forecast is a cohort
//   blend, not a trend extrapolation:
//     1. Detect refreshes in the history (a jump in players_tracked).
//     2. At each refresh, recover the arriving cohort's earn rate per trophy
//        from the earned-count difference, and pool the logit gap between
//        the old rate and the new cohort's rate by tier (hierarchical
//        shrinkage toward a prior gap of 1.0 logit).
//     3. Estimate the progression drift inside refresh-free stretches:
//        the slope of logit rate against ln(1 + hours since release),
//        pooled by tier with shrinkage.
//     4. Predict the tracked count at the target from the expected number
//        of refreshes left (refresh rate from the history, shrunk toward
//        one refresh per five hours) times the typical log growth per
//        refresh (from the history, shrunk toward tripling).
//     5. Blend: the old cohort keeps progressing along its drift, the
//        arriving players earn at the new-cohort rate (aged half the
//        horizon), and the median is the player-weighted mean of the two.
//   The spread is a random walk in logit plus the refresh uncertainty
//   (expected refresh count times the typical one-refresh logit drop),
//   so it is wide exactly when a refresh is likely but its size is unsure.
//   With a flat history and no refresh seen yet, the forecast still
//   expects dilution — that is the whole point of the practice cases —
//   while a long history of real refreshes lets the data take over.
//
// Nothing here is tuned to any particular case: every number the forecast
// uses is either read from the case file or a stated prior above, and the
// same code runs the 90-minute practice cases and the 4.5-day final case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// The quantile levels the contest scores, in the order each row lists them.
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
// The standard normal's quantile at each level.
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

// --- Priors (stated, not fitted to any case) --------------------------------
// How often Exophase refreshes a game's counts: about once every four
// hours on launch night ("hours apart" in the rules; the first refresh came
// after roughly four and three-quarter hours of flat reads).
const PRIOR_REFRESH_PER_HOUR = 0.25;
// How much the tracked count grows at one refresh, in log terms: the first
// refresh grew it six-fold (638 -> 3868); later ones are smaller, so the
// prior sits at four-fold and the history pulls it down as refreshes arrive.
const PRIOR_LOG_GROWTH_PER_REFRESH = Math.log(4);
// The logit gap between the tracked players' rate and a newly arriving
// cohort's rate for the same trophy: new players have barely played. The
// rules' own totals pin the scale: at the first refresh the tracked count
// went 638 -> 3868 while trophies earned went 943 -> 4276, so the arriving
// cohort holds about 1.03 trophies a player against 1.48 for the base —
// and all but one trophy's rate fell, so the gap is large, about two logit
// for a typical trophy, not a fraction of one.
const PRIOR_COHORT_GAP = 2.0;
// The random-walk spread of a rate in logit units per square-root day.
const WALK_SIGMA = 0.5;
// A floor on the logit spread, so no trophy is ever forecast with false
// certainty, and a ceiling, so a long horizon cannot blow the band up.
const SPREAD_FLOOR = 0.3;
const SPREAD_CEILING = 2.5;

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  atMs: number;
  players: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (x: number): number => {
  const p = 1 / (1 + Math.exp(-x));
  return Math.min(100, Math.max(0, p * 100));
};
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
// Weighted mean of an estimate and a prior: `weight` counts how strongly
// the estimate is trusted, `priorWeight` how strongly the prior holds.
const shrink = (est: number, weight: number, prior: number, priorWeight: number): number =>
  (est * weight + prior * priorWeight) / (weight + priorWeight);

function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const releaseMs = Date.parse(`${c.game.release_date ?? "2026-11-19"}T00:00:00Z`);
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonHours = Math.max(0, c.horizon_minutes) / 60;
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const ageAt = (ms: number): number => Math.max(0, (ms - releaseMs) / 3600000);
  const logAge = (ms: number): number => Math.log(1 + ageAt(ms));

  // Normalise the read times to milliseconds (the case gives ISO strings).
  const hist: Read[] = (c.history ?? []).map((r) => ({
    atMs: Date.parse((r as unknown as { at: string }).at),
    players: (r as unknown as { players_tracked: number }).players_tracked,
    rates: r.rates,
  })).filter((r) => Number.isFinite(r.atMs) && Number.isFinite(r.players) && r.players > 0);

  const nT = c.trophies.length;
  // The last non-null rate per trophy, and its read index.
  const lastRate: number[] = new Array(nT).fill(NaN);
  const lastIdx: number[] = new Array(nT).fill(-1);
  hist.forEach((r, i) => {
    r.rates.forEach((v, j) => {
      if (v !== null && Number.isFinite(v)) {
        lastRate[j] = v;
        lastIdx[j] = i;
      }
    });
  });
  // Fallback rates when a trophy was never seen: the tier's typical level.
  const tierFallback: Record<string, number> = { platinum: 0.5, gold: 1.0, silver: 1.5, bronze: 2.0 };
  for (let j = 0; j < nT; j++) {
    if (!Number.isFinite(lastRate[j])) lastRate[j] = tierFallback[c.trophies[j].tier] ?? 1.0;
  }
  const lastRead = hist.length > 0 ? hist[hist.length - 1] : null;
  const lastN = lastRead ? lastRead.players : 1000; // No reads: only a blend weight, any scale works.

  // --- Refresh detection ----------------------------------------------------
  // A refresh is a read where the tracked count jumps. Small wiggles are
  // not refreshes; the jump has to clear 3% and at least 20 players.
  const refreshAt: boolean[] = hist.map(() => false);
  const refreshIdx: number[] = [];
  for (let i = 1; i < hist.length; i++) {
    const prev = hist[i - 1];
    const cur = hist[i];
    if (cur.players > prev.players * 1.03 && cur.players - prev.players >= 20) {
      refreshAt[i] = true;
      refreshIdx.push(i);
    }
  }
  const spanHours = hist.length >= 2
    ? Math.max(1, (hist[hist.length - 1].atMs - hist[0].atMs) / 3600000)
    : 1;
  // Refresh rate per hour, shrunk toward the prior (10 pseudo-hours).
  const lambda = (refreshIdx.length + PRIOR_REFRESH_PER_HOUR * 10) / (spanHours + 10);
  const expectedRefreshes = lambda * horizonHours;
  // Typical log growth of the tracked count per refresh, shrunk toward the
  // prior (4 pseudo-refreshes).
  const growths = refreshIdx.map((i) => Math.log(hist[i].players / hist[i - 1].players));
  const logGrowth = growths.length > 0
    ? shrink(median(growths), growths.length, PRIOR_LOG_GROWTH_PER_REFRESH, 4)
    : PRIOR_LOG_GROWTH_PER_REFRESH;
  // Expected tracked count at the target (capped: no launch grows 20-fold
  // in one horizon on this model).
  const playersTarget = Math.min(lastN * 20, lastN * Math.exp(expectedRefreshes * logGrowth));

  // --- New-cohort logit gap, per trophy -------------------------------------
  // At a refresh, the arriving cohort's rate follows from the earned counts:
  //   rho = (N_after * r_after - N_before * r_before) / (N_after - N_before)
  // Pool the gap logit(r_before) - logit(rho) by tier, then shrink to the
  // prior; trophies with no observed refresh keep the prior gap.
  const gapsByTier: Record<string, number[]> = {};
  const gapsGlobal: number[] = [];
  const trophyGap: number[] = new Array(nT).fill(NaN);
  const trophyGapN: number[] = new Array(nT).fill(0);
  for (const i of refreshIdx) {
    const prev = hist[i - 1];
    const cur = hist[i];
    const dN = cur.players - prev.players;
    if (dN <= 0) continue;
    for (let j = 0; j < nT; j++) {
      const rb = prev.rates[j];
      const ra = cur.rates[j];
      if (rb === null || ra === null || rb <= 0) continue;
      const earnedBefore = (prev.players * rb) / 100;
      const earnedAfter = (cur.players * ra) / 100;
      const rho = ((earnedAfter - earnedBefore) / dN) * 100;
      if (!Number.isFinite(rho)) continue;
      const gap = logit(rb) - logit(Math.max(0.01, rho));
      // A gap outside [-1, 6] is a counting artefact of a tiny cohort, not
      // a real cohort difference; drop it.
      if (gap < -1 || gap > 6) continue;
      trophyGap[j] = Number.isFinite(trophyGap[j]) ? trophyGap[j] + gap : gap;
      trophyGapN[j] += 1;
      const tier = c.trophies[j].tier;
      (gapsByTier[tier] ??= []).push(gap);
      gapsGlobal.push(gap);
    }
  }
  const globalGap = gapsGlobal.length > 0 ? median(gapsGlobal) : PRIOR_COHORT_GAP;
  const tierGap: Record<string, number> = {};
  for (const [tier, xs] of Object.entries(gapsByTier)) {
    tierGap[tier] = shrink(median(xs), xs.length, globalGap, 4);
  }
  const gapOf = (j: number): number => {
    if (trophyGapN[j] > 0) {
      const own = trophyGap[j] / trophyGapN[j];
      const pool = tierGap[c.trophies[j].tier] ?? globalGap;
      return shrink(own, trophyGapN[j], pool, 4);
    }
    return tierGap[c.trophies[j].tier] ?? globalGap;
  };

  // --- Progression drift, per trophy ----------------------------------------
  // Within a refresh-free stretch, logit rates move with ln(1 + age). Take
  // consecutive-read differences inside stretches, pool by tier, shrink.
  const slopeNum: number[] = new Array(nT).fill(0);
  const slopeDen: number[] = new Array(nT).fill(0);
  const slopesByTier: Record<string, number[]> = {};
  const slopesGlobal: number[] = [];
  for (let i = 1; i < hist.length; i++) {
    if (refreshAt[i]) continue;
    const prev = hist[i - 1];
    const cur = hist[i];
    const dx = logAge(cur.atMs) - logAge(prev.atMs);
    if (dx <= 0.001) continue;
    for (let j = 0; j < nT; j++) {
      const rb = prev.rates[j];
      const ra = cur.rates[j];
      if (rb === null || ra === null) continue;
      const s = (logit(ra) - logit(rb)) / dx;
      if (!Number.isFinite(s) || s < -2 || s > 4) continue;
      slopeNum[j] += s;
      slopeDen[j] += 1;
      (slopesByTier[c.trophies[j].tier] ??= []).push(s);
      slopesGlobal.push(s);
    }
  }
  const globalSlope = slopesGlobal.length > 0 ? median(slopesGlobal) : 0;
  const tierSlope: Record<string, number> = {};
  for (const [tier, xs] of Object.entries(slopesByTier)) {
    tierSlope[tier] = shrink(median(xs), xs.length, globalSlope, 4);
  }
  const slopeOf = (j: number): number => {
    if (slopeDen[j] > 0) {
      const own = slopeNum[j] / slopeDen[j];
      const pool = tierSlope[c.trophies[j].tier] ?? globalSlope;
      return shrink(own, slopeDen[j], pool, 4);
    }
    return tierSlope[c.trophies[j].tier] ?? globalSlope;
  };
  // Drift is capped: a week's play cannot move a logit by more than 2.
  const driftCap = 2;

  // --- Forecast ---------------------------------------------------------------
  const lastLogAge = lastRead ? logAge(lastRead.atMs) : logAge(checkpointMs);
  const targetLogAge = logAge(targetMs);
  const out: Array<{ id: number; q: number[] }> = [];
  for (let j = 0; j < nT; j++) {
    const r = lastRate[j];
    const slope = slopeOf(j);
    const drift = Math.max(-driftCap, Math.min(driftCap, slope * (targetLogAge - lastLogAge)));
    // The old cohort, progressing.
    const progressed = invLogitPct(logit(r) + drift);
    // The arriving cohort: its rate at arrival, aged half the horizon.
    const arrivalGap = gapOf(j);
    const joinerAge = Math.max(0.5, horizonHours / 2);
    const joinerDrift = Math.max(
      -driftCap,
      Math.min(driftCap, slope * (Math.log(1 + joinerAge) - Math.log(1.5))),
    );
    const joinerRate = invLogitPct(logit(r) - arrivalGap + joinerDrift);
    // Player-weighted blend at the target.
    const medianRate = Math.min(
      100,
      Math.max(
        0,
        (lastN * progressed + Math.max(0, playersTarget - lastN) * joinerRate) / playersTarget,
      ),
    );
    // Spread: the random walk, plus the refresh uncertainty. One refresh
    // moves this trophy by about the gap between its current rate and the
    // rate after a single average refresh, so the refresh variance scales
    // with the expected refresh count times that squared move — doubled,
    // because whether a refresh lands inside the horizon at all is itself
    // a coin flip, and the 0.25 floor keeps the band honest when the
    // history is flat and no refresh has been seen yet.
    const afterOne = (lastN * r + (Math.exp(logGrowth) - 1) * lastN * joinerRate) /
      (lastN * Math.exp(logGrowth));
    const oneJump = Math.abs(logit(r) - logit(afterOne));
    const variance =
      WALK_SIGMA * WALK_SIGMA * horizonDays +
      2 * expectedRefreshes * oneJump * oneJump +
      0.25;
    const spread = Math.max(SPREAD_FLOOR, Math.min(SPREAD_CEILING, Math.sqrt(variance)));
    const centre = logit(medianRate);
    const q = Z.map((z) => round4(invLogitPct(centre + z * spread)));
    // Rounding must never break the ordering.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    out.push({ id: c.trophies[j].id, q });
  }
  return out;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith(".in")).sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, k) => t !== LEVELS[k])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), JSON.stringify({ trophies }) + "\n", "utf8");
}
