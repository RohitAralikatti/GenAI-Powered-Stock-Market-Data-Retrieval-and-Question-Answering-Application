// Maps the high-level UI inputs (scenario, risk, horizon) onto the subset of
// firm-characteristic features the model is most sensitive to. Any feature
// not produced here is defaulted to 0.0 by the backend.

const PRESETS = {
  conservative: {
    mvel1: 0.35,
    beta: 0.6,
    chmom: -0.02,
    dolvol: 0.25,
    idiovol: 0.08,
    indmom: 0.0,
    mom1m: 0.0,
    mom6m: 0.0,
    mom12m: 0.02,
  },
  neutral: {
    mvel1: 0.12,
    beta: 1.0,
    chmom: 0.0,
    dolvol: 0.3,
    idiovol: 0.15,
    indmom: 0.0,
    mom1m: 0.0,
    mom6m: 0.05,
    mom12m: 0.08,
  },
  aggressive: {
    mvel1: -0.1,
    beta: 1.5,
    chmom: 0.05,
    dolvol: 0.4,
    idiovol: 0.3,
    indmom: 0.05,
    mom1m: 0.03,
    mom6m: 0.12,
    mom12m: 0.2,
  },
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

export function buildFeatures(scenario, risk, horizon) {
  const base = { ...(PRESETS[scenario] ?? PRESETS.neutral) };

  // risk in [0, 1] -> tilt in [-1, 1], 0.5 leaves the scenario baseline unchanged
  const riskTilt = (risk - 0.5) * 2;
  const riskMultiplier = 1 + riskTilt * 0.4;

  const beta = base.beta * riskMultiplier;
  const idiovol = base.idiovol * riskMultiplier;
  const dolvol = base.dolvol * riskMultiplier;
  const chmom = base.chmom * riskMultiplier;
  const indmom = base.indmom * riskMultiplier;

  // horizon in months -> weight momentum windows relative to a 12-month neutral point
  const horizonRatio = clamp(horizon / 12, 0.25, 4);
  const mom1mWeight = clamp(1 / horizonRatio, 0.25, 4);
  const mom12mWeight = clamp(horizonRatio, 0.25, 4);

  return {
    mvel1: base.mvel1,
    beta,
    betasq: beta * beta,
    chmom,
    dolvol,
    idiovol,
    indmom,
    mom1m: base.mom1m * mom1mWeight,
    mom6m: base.mom6m,
    mom12m: base.mom12m * mom12mWeight,
  };
}
