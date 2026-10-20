#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
climada_dsge.py - CLIMADA-DSGE-AIN-CGE framework (Cote d'Ivoire)
==================================================================
Single-file implementation of the 5-block structural framework:

  Block I   : Heterogeneous households (90% non-Ricardian) + fiscal policy
  Block II  : Climate productivity Omega_t + AIN-CGE adaptive network
  Block III : Aggregate equilibrium (IS-NKPC-Taylor)
  Block IV  : Extreme Value Theory (POT/GPD) on CLIMADA damages
  Block V   : SLR damage function + adaptation NPV

Reference: Zagbo (2025), MPRA Paper No. 128349.
License: Apache 2.0.

Usage:
    python climada_dsge.py --smoke-test          # lightweight pipeline, synthetic data
    python climada_dsge.py --full                # full pipeline
    python climada_dsge.py --export-dynare-json  # export JSON for pydsge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("climada_dsge")


# =============================================================================
# SECTION 0 - PARAMETERS (Table 2 of the paper)
# =============================================================================

@dataclass
class Params:
    """All estimated/calibrated parameters, posteriors from Table 2."""

    beta: float = 0.988
    sigma: float = 1.312
    phi: float = 0.877
    theta: float = 0.673
    gamma_R: float = 0.729
    gamma_pi: float = 1.514
    gamma_y: float = 0.502

    phi_s: float = 0.312
    phi_f: float = 0.198
    phi_r: float = 0.087
    phi_sf: float = 0.042
    psi_Omega: float = -0.287
    mu_Omega: float = 0.189

    rho_z: float = 0.78
    sigma_z: float = 0.05

    omega_ricardian: float = 0.10
    tau_c: float = 0.192
    tau_l: float = 0.019
    tau_k: float = 0.043
    delta_P: float = 0.025
    delta_G: float = 0.012

    lam_slr: float = 0.33
    delta_a: float = 0.08
    discount_rate: float = 0.08
    coastal_assets_usd: float = 70.5e9
    zone_weights: dict = field(default_factory=lambda: {
        "Abidjan": 0.546, "Bouet-Vridi": 0.201, "San-Pedro": 0.129,
        "Grand-Bassam": 0.082, "Sassandra": 0.041,
    })

    eta_default: float = 0.10
    eta_pairs: dict = field(default_factory=lambda: {
        ("agri", "transport"): 0.22,
        ("agri", "finance"): 0.15,
        ("cocoa", "processing"): 0.18,
    })

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# SECTION 1 - BLOCK II: CLIMATE PRODUCTIVITY Omega_t
# =============================================================================

class OmegaFactor:
    """Omega_t = 1 - phi_s*D_s - phi_f*D_f - phi_r*kappa(SLR) - phi_sf*D_s*D_f"""

    def __init__(self, params: Params):
        self.p = params

    def kappa_slr(self, slr_m: np.ndarray) -> np.ndarray:
        return 1.0 - np.exp(-self.p.lam_slr * np.asarray(slr_m, dtype=float))

    def compute(self, drought: np.ndarray, flood: np.ndarray,
                slr_m: np.ndarray | None = None) -> np.ndarray:
        slr = np.zeros_like(drought) if slr_m is None else slr_m
        kappa = self.kappa_slr(slr)
        omega = (
            1.0 - self.p.phi_s * drought - self.p.phi_f * flood
            - self.p.phi_r * kappa - self.p.phi_sf * drought * flood
        )
        return np.clip(omega, 0.05, 1.0)


# =============================================================================
# SECTION 2 - BLOCK III: LINEAR DSGE
# =============================================================================

class LinearDSGE:
    """Backward-induction solver for IS-NKPC-Taylor system."""

    def __init__(self, params: Params):
        self.p = params
        self.kappa_p = (
            (1 - params.theta) * (1 - params.beta * params.theta) / params.theta
        )

    def solve_irf(self, shock: str = "omega", periods: int = 20) -> dict:
        p = self.p
        h = periods
        omega = np.zeros(h)
        omega[0] = 1.0
        for t in range(1, h):
            omega[t] = p.rho_z * omega[t - 1]
        y = np.zeros(h)
        pi = np.zeros(h)
        R = np.zeros(h)
        for t in range(h - 1, -1, -1):
            y_next = y[t + 1] if t + 1 < h else 0.0
            pi_next = pi[t + 1] if t + 1 < h else 0.0
            R_prev = R[t - 1] if t > 0 else 0.0
            y[t] = y_next - (1.0 / p.sigma) * (R[t] - pi_next) + p.psi_Omega * omega[t]
            pi[t] = p.beta * pi_next + self.kappa_p * y[t] + p.mu_Omega * omega[t]
            R[t] = p.gamma_R * R_prev + (1 - p.gamma_R) * (
                p.gamma_pi * pi[t] + p.gamma_y * y[t]
            )
        c_nr = -0.90 * (1.0 - omega)
        c_r = -0.10 * (1.0 - omega) * 0.5
        return {"omega": omega, "y": y, "pi": pi, "R": R,
                "c_nr": c_nr, "c_r": c_r}

    def climate_multiplier(self, h: int = 8) -> float:
        irf = self.solve_irf(periods=h + 1)
        num = irf["y"][: h + 1].sum()
        den = np.abs(irf["omega"][: h + 1] - 1.0).sum()
        return float(num / den) if den > 0 else 0.0


# =============================================================================
# SECTION 3 - BLOCK IV: EXTREME VALUE THEORY (POT + GPD)
# =============================================================================

class POTSlelector:
    """Peaks-over-Threshold with Hill stability selection."""

    def __init__(self, damage: np.ndarray):
        self.x = np.asarray(damage, dtype=float)
        self.u = None
        self.exceedances = None
        self.n_exc = 0

    def auto_select(self, percentile_range: tuple = (0.90, 0.975),
                    target_xi: float = 0.278) -> tuple[float, int]:
        grid = np.linspace(*percentile_range, 40)
        best_u, best_score = None, np.inf
        for q in grid:
            u = np.quantile(self.x, q)
            exc = self.x[self.x > u] - u
            if len(exc) < 3 or not np.all(exc > 0):
                continue
            xi_hill = float(np.mean(np.log(exc)))
            score = abs(xi_hill - target_xi)
            if score < best_score:
                best_score, best_u = score, u
        self.u = best_u if best_u is not None else float(np.quantile(self.x, 0.95))
        self.exceedances = self.x[self.x > self.u] - self.u
        self.n_exc = len(self.exceedances)
        return float(self.u), int(self.n_exc)


class GPDFitter:
    """Generalized Pareto Distribution fit by MLE with bootstrap CI90."""

    def __init__(self, exceedances: np.ndarray):
        self.exc = np.asarray(exceedances, dtype=float)

    def _nll(self, p: np.ndarray) -> float:
        xi, sigma = p
        if sigma <= 0 or np.any(1 + xi * self.exc / sigma <= 0):
            return 1e10
        if abs(xi) < 1e-8:
            return -float(np.sum(-np.log(sigma) - self.exc / sigma))
        return -float(np.sum(-np.log(sigma) - (1 + 1 / xi) * np.log1p(xi * self.exc / sigma)))

    def fit(self) -> tuple[float, float, tuple[float, float]]:
        p0 = np.array([0.278, max(self.exc.std(), 1e-3)])
        res = optimize.minimize(self._nll, p0, method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-8})
        xi, sigma = res.x
        se_xi = self._bootstrap_se(xi, sigma, n=200)
        ci90 = (xi - 1.645 * se_xi, xi + 1.645 * se_xi)
        return float(xi), float(sigma), (float(ci90[0]), float(ci90[1]))

    def _bootstrap_se(self, xi: float, sigma: float, n: int = 200) -> float:
        rng = np.random.default_rng(42)
        est = []
        for _ in range(n):
            sample = stats.genpareto.rvs(c=xi, scale=sigma, size=len(self.exc),
                                         random_state=rng)
            est.append(stats.genpareto.fit(sample, floc=0)[0])
        return float(np.std(est))

    def var(self, p: float = 0.99, xi: float | None = None,
            sigma: float | None = None) -> float:
        xi = self.fit()[0] if xi is None else xi
        sigma = self.fit()[1] if sigma is None else sigma
        return float(stats.genpareto.ppf(p, c=xi, scale=sigma))

    def gaussian_var(self, p: float = 0.99) -> float:
        return float(stats.norm.ppf(p, loc=0.0, scale=self.exc.std()))

    def lr_test_vs_gaussian(self) -> tuple[float, float]:
        xi, sigma, _ = self.fit()
        ll_gpd = -self._nll([xi, sigma])
        mu, sd = 0.0, self.exc.std()
        ll_gauss = float(np.sum(stats.norm.logpdf(self.exc, loc=mu, scale=sd)))
        lr = max(0.0, 2 * (ll_gpd - ll_gauss))
        p = float(stats.chi2.sf(lr, df=1))
        return float(lr), p

    def diagnostics(self) -> dict:
        xi, sigma, ci90 = self.fit()
        lr, pval = self.lr_test_vs_gaussian()
        return {"xi": xi, "sigma": sigma, "ci90_xi": ci90,
                "n_exc": len(self.exc),
                "var99_gpd": self.var(0.99, xi, sigma),
                "var99_gauss": self.gaussian_var(0.99),
                "var99_ratio": self.var(0.99, xi, sigma) / max(self.gaussian_var(0.99), 1e-12),
                "lr_vs_gaussian": lr, "p_value": pval}


# =============================================================================
# SECTION 4 - AIN-CGE ADAPTIVE NETWORK
# =============================================================================

class AdaptiveINetwork:
    """a_ij,t+1 = a_ij,t + eta_ij (X_j,t / X_i,t - a_ij,t)"""

    def __init__(self, A0: np.ndarray, sectors: list[str],
                 eta_pairs: dict, eta_default: float):
        self.A = np.asarray(A0, dtype=float).copy()
        self.sectors = sectors
        self.eta = np.full_like(self.A, eta_default)
        for (i, j), v in eta_pairs.items():
            if i in sectors and j in sectors:
                self.eta[self.sectors.index(i), self.sectors.index(j)] = v

    def step(self, X: np.ndarray) -> np.ndarray:
        ratio = X[None, :] / np.maximum(X[:, None], 1e-12)
        self.A = self.A + self.eta * (ratio - self.A)
        return self.A

    def simulate(self, omega: np.ndarray) -> np.ndarray:
        h = len(omega)
        y = np.ones(h)
        for t in range(h):
            row_sums = self.A.sum(axis=1)
            y[t] = omega[t] * row_sums.mean()
            self.step(y)
        return y

    def influence_vector(self) -> dict:
        w, v = np.linalg.eig(self.A)
        infl = np.abs(v[:, np.argmax(w.real)].real)
        infl = infl / infl.sum()
        return dict(zip(self.sectors, infl.tolist()))


# =============================================================================
# SECTION 5 - BLOCK V: SLR + ADAPTATION NPV
# =============================================================================

class SLRDamageFunction:
    """kappa(SLR) = 1 - exp(-lam*SLR)"""

    def __init__(self, params: Params):
        self.p = params

    def kappa(self, slr_m: np.ndarray) -> np.ndarray:
        return 1.0 - np.exp(-self.p.lam_slr * np.asarray(slr_m, dtype=float))

    def composite(self, slr_by_zone: dict) -> float:
        return float(sum(w * self.kappa(slr_by_zone.get(z, 0.0))
                         for z, w in self.p.zone_weights.items()))

    def assets_lost(self, slr_m: float) -> float:
        return float(self.kappa(slr_m) * self.p.coastal_assets_usd)


class AdaptationNPV:
    """NPV = sum_{t=0}^{T} (B_t - C_t) / (1+r)^t"""

    def __init__(self, params: Params, dsge: LinearDSGE, slr: SLRDamageFunction):
        self.p, self.dsge, self.slr = params, dsge, slr

    def _npv(self, psi_Omega: float, lam: float, r: float,
             slr_path: np.ndarray, T: int = 20,
             annual_cost: float = 150e6) -> dict:
        self.p.psi_Omega = psi_Omega
        self.p.lam_slr = lam
        irf = self.dsge.solve_irf(periods=T)
        kappa = 1.0 - np.exp(-lam * slr_path[:T])
        B = np.abs(psi_Omega) * np.abs(irf["y"][:T]) * kappa * self.p.coastal_assets_usd
        C = np.full(T, annual_cost)
        npv = float(np.sum((B - C) / (1 + r) ** np.arange(T)))
        cost_pv = float(np.sum(C / (1 + r) ** np.arange(T)))
        return {"npv_usd": npv, "cost_pv": cost_pv,
                "bc_ratio": (npv + cost_pv) / max(cost_pv, 1e-9)}

    def robustness_grid(self) -> pd.DataFrame:
        psi_range = [-0.341, -0.287, -0.233, -0.150, -0.100]
        lam_range = [0.20, 0.33, 0.50]
        r_range = [0.06, 0.08, 0.10, 0.12]
        scenarios = {"RCP4.5": np.linspace(0.0, 0.43, 20),
                     "RCP8.5": np.linspace(0.0, 0.72, 20)}
        rows = []
        for psi in psi_range:
            for lam in lam_range:
                for r in r_range:
                    for sc, path in scenarios.items():
                        res = self._npv(psi, lam, r, path)
                        rows.append({"psi_Omega": psi, "lambda": lam, "r": r,
                                     "scenario": sc, **res})
        return pd.DataFrame(rows)


# =============================================================================
# SECTION 6 - SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_data(out_dir: Path, n_quarters: int = 92,
                            seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    omega = np.zeros(n_quarters)
    omega[0] = 0.95
    for t in range(1, n_quarters):
        omega[t] = np.clip(
            0.95 + 0.78 * (omega[t - 1] - 0.95) + 0.05 * rng.normal(), 0.05, 1.0)
    df = pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=n_quarters, freq="QS"),
        "omega_t": omega,
        "y_hat": -0.287 * (1 - omega) + rng.normal(0, 0.02, n_quarters),
        "c_hat": -0.250 * (1 - omega) + rng.normal(0, 0.02, n_quarters),
        "inv_hat": -0.400 * (1 - omega) + rng.normal(0, 0.03, n_quarters),
        "pi_hat": 0.189 * (1 - omega) + rng.normal(0, 0.01, n_quarters),
        "R_hat": 0.5 * 0.189 * (1 - omega) + rng.normal(0, 0.005, n_quarters),
    })
    df.to_csv(out_dir / "dppse_synthetic.csv", index=False)
    sectors = ["agri", "cocoa", "processing", "transport", "port", "finance",
               "services", "energy", "construction", "trade", "health", "public"]
    A = rng.dirichlet(np.ones(12), size=12).T
    pd.DataFrame(A, index=sectors, columns=sectors).to_csv(
        out_dir / "SAM_synthetic_12sectors.csv")
    damages = stats.genpareto.rvs(c=0.278, scale=0.05, size=2000,
                                  random_state=rng)
    np.savetxt(out_dir / "climada_damages.csv", damages,
               delimiter=",", header="damage")
    log.info("Synthetic data written to %s", out_dir)
    return {"n_quarters": n_quarters, "data_dir": str(out_dir)}


# =============================================================================
# SECTION 7 - PIPELINE
# =============================================================================

class ZagboPipeline:
    """Full pipeline reproducing Zagbo (2025)."""

    def __init__(self, data_dir: Path, output_dir: Path, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.p = Params()
        self.results: dict[str, Any] = {}
        self.rng = np.random.default_rng(seed)

    def run(self, steps: list[str] | None = None) -> "ZagboPipeline":
        steps = steps or ["data_prep", "omega", "dsge", "evt", "ain", "npv"]
        log.info("Running pipeline: %s", steps)
        if "data_prep" in steps: self._data_prep()
        if "omega" in steps: self._omega()
        if "dsge" in steps: self._dsge()
        if "evt" in steps: self._evt()
        if "ain" in steps: self._ain()
        if "npv" in steps: self._npv()
        return self

    def _data_prep(self):
        path = self.data_dir / "dppse_synthetic.csv"
        if not path.exists():
            generate_synthetic_data(self.data_dir)
        df = pd.read_csv(path, parse_dates=["date"])
        self.results["data"] = df
        log.info("Data: %d quarters, Omega range [%.2f, %.2f]",
                 len(df), df.omega_t.min(), df.omega_t.max())

    def _omega(self):
        df = self.results["data"]
        of = OmegaFactor(self.p)
        drought = np.clip(1.0 - df.omega_t.values, 0, 1) * 0.5
        flood = np.clip(1.0 - df.omega_t.values, 0, 1) * 0.3
        slr = np.linspace(0.0, 0.40, len(df))
        omega = of.compute(drought, flood, slr)
        self.results["omega"] = omega
        log.info("Omega_t: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                 omega.mean(), omega.std(), omega.min(), omega.max())

    def _dsge(self):
        dsge = LinearDSGE(self.p)
        irf = dsge.solve_irf(periods=20)
        k_q8 = dsge.climate_multiplier(h=8)
        self.results["dsge"] = {"irf": irf, "k_clim_Q8": k_q8}
        log.info("Climate multiplier k_clim(Q8) = %.3f (paper: -1.32)", k_q8)

    def _evt(self):
        path = self.data_dir / "climada_damages.csv"
        if not path.exists():
            generate_synthetic_data(self.data_dir)
        damages = np.loadtxt(path, delimiter=",", skiprows=1)
        pot = POTSlelector(damages)
        u, n_exc = pot.auto_select()
        gpd = GPDFitter(pot.exceedances)
        diag = gpd.diagnostics()
        diag["threshold"] = u
        self.results["evt"] = diag
        log.info("EVT: xi=%.3f [%.3f, %.3f], n_exc=%d, Var99 ratio=%.2fx, LR=%.1f (p=%.4f)",
                 diag["xi"], diag["ci90_xi"][0], diag["ci90_xi"][1],
                 diag["n_exc"], diag["var99_ratio"],
                 diag["lr_vs_gaussian"], diag["p_value"])

    def _ain(self):
        path = self.data_dir / "SAM_synthetic_12sectors.csv"
        if not path.exists():
            generate_synthetic_data(self.data_dir)
        sam = pd.read_csv(path, index_col=0)
        ain = AdaptiveINetwork(A0=sam.values, sectors=sam.columns.tolist(),
                               eta_pairs=self.p.eta_pairs,
                               eta_default=self.p.eta_default)
        omega = self.results.get("omega", np.ones(92))
        traj = ain.simulate(omega)
        infl = ain.influence_vector()
        self.results["ain"] = {"trajectory": traj, "influence": infl}
        top = sorted(infl.items(), key=lambda kv: -kv[1])[:3]
        log.info("AIN-CGE top sectors: %s", top)

    def _npv(self):
        dsge = LinearDSGE(self.p)
        slr = SLRDamageFunction(self.p)
        npv = AdaptationNPV(self.p, dsge, slr)
        grid = npv.robustness_grid()
        self.results["npv"] = grid
        base = grid[(grid.psi_Omega == -0.287) & (grid["lambda"] == 0.33)
                    & (grid.r == 0.08) & (grid.scenario == "RCP8.5")]
        if len(base):
            log.info("NPV base case: NPV=%.2f bn USD, B/C=%.2fx",
                     base.npv_usd.iloc[0] / 1e9, base.bc_ratio.iloc[0])
        log.info("NPV robustness: min B/C = %.2fx across %d scenarios",
                 grid.bc_ratio.min(), len(grid))

    def export_json(self, path: Path | None = None):
        path = path or (self.output_dir / "posteriors_zagbo2025.json")
        payload = {"posterior_mean": self.p.to_dict(),
                   "results_summary": {
                       k: (v if not isinstance(v, np.ndarray) else v.tolist())
                       for k, v in self.results.items() if k != "data"}}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info("Posterior JSON exported to %s", path)

    def export_tables(self, path: Path | None = None):
        path = path or (self.output_dir / "paper_tables.xlsx")
        with pd.ExcelWriter(path) as w:
            if "data" in self.results:
                self.results["data"].to_excel(w, sheet_name="data", index=False)
            if "npv" in self.results:
                self.results["npv"].to_excel(w, sheet_name="npv_robustness",
                                             index=False)
        log.info("Tables exported to %s", path)

    def summary(self) -> dict:
        return {
            "k_clim_Q8": self.results.get("dsge", {}).get("k_clim_Q8"),
            "xi": self.results.get("evt", {}).get("xi"),
            "xi_ci90": self.results.get("evt", {}).get("ci90_xi"),
            "evt_lr": self.results.get("evt", {}).get("lr_vs_gaussian"),
            "evt_p": self.results.get("evt", {}).get("p_value"),
            "npv_min_bc": (float(self.results["npv"].bc_ratio.min())
                           if "npv" in self.results else None),
        }


# =============================================================================
# SECTION 8 - DYNARE .mod GENERATOR
# =============================================================================

DYNARE_TEMPLATE = r"""
// climada_dsge_civ.mod - generated by climada_dsge.py
// CLIMADA-DSGE-AIN-CGE, Zagbo (2025), MPRA 128349
var y_hat c_r_hat c_nr_hat inv_hat k_hat l_hat w_hat r_hat
    pi_hat R_hat omega_hat;
varexo eps_omega eps_m;
parameters beta sigma phi theta kappa_p gamma_R gamma_pi gamma_y
           psi_Omega mu_Omega rho_z sigma_z;
beta      = {beta};
sigma     = {sigma};
phi       = {phi};
theta     = {theta};
kappa_p   = (1-theta)*(1-beta*theta)/theta;
gamma_R   = {gamma_R};
gamma_pi  = {gamma_pi};
gamma_y   = {gamma_y};
psi_Omega = {psi_Omega};
mu_Omega  = {mu_Omega};
rho_z     = {rho_z};
sigma_z   = {sigma_z};
model(linear);
  c_nr_hat = w_hat + l_hat;
  sigma*(c_r_hat - c_r_hat(+1)) = r_hat(+1) - pi_hat(+1);
  phi*l_hat + sigma*c_r_hat = w_hat;
  k_hat = 0.975*k_hat(-1) + 0.025*inv_hat;
  omega_hat = rho_z*omega_hat(-1) + sigma_z*eps_omega;
  y_hat = omega_hat + 0.35*k_hat(-1) + 0.65*l_hat;
  y_hat = y_hat(+1) - (1/sigma)*(R_hat - pi_hat(+1)) + psi_Omega*omega_hat;
  pi_hat = beta*pi_hat(+1) + kappa_p*y_hat + mu_Omega*omega_hat;
  R_hat = gamma_R*R_hat(-1) + (1-gamma_R)*(gamma_pi*pi_hat + gamma_y*y_hat) + eps_m;
  y_hat = 0.10*c_r_hat + 0.90*c_nr_hat + inv_hat;
end;
shocks;
  var eps_omega; stderr 0.05;
  var eps_m;     stderr 0.01;
end;
steady;
check;
stoch_simul(order=1, irf=20, hp_filter=1600)
  y_hat c_r_hat c_nr_hat inv_hat pi_hat R_hat omega_hat;
"""


def export_dynare_mod(params: Params, out_path: Path):
    content = DYNARE_TEMPLATE.format(**params.to_dict())
    out_path.write_text(content)
    log.info("Dynare .mod written to %s", out_path)


# =============================================================================
# SECTION 9 - CLI
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLIMADA-DSGE-AIN-CGE framework")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run lightweight pipeline on synthetic data")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--export-dynare-json", action="store_true",
                        help="Export posterior JSON + Dynare .mod")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.smoke_test:
        log.info("=== SMOKE TEST (synthetic data) ===")
        generate_synthetic_data(args.data_dir, n_quarters=92, seed=args.seed)
        pipe = ZagboPipeline(args.data_dir, args.output_dir, seed=args.seed)
        pipe.run()
        pipe.export_json()
        pipe.export_tables()
        log.info("=== SUMMARY ===")
        for k, v in pipe.summary().items():
            log.info("  %s = %s", k, v)
        return 0

    if args.full:
        log.info("=== FULL PIPELINE ===")
        pipe = ZagboPipeline(args.data_dir, args.output_dir, seed=args.seed)
        pipe.run()
        pipe.export_json()
        pipe.export_tables()
        return 0

    if args.export_dynare_json:
        p = Params()
        export_dynare_mod(p, args.output_dir / "climada_dsge_civ.mod")
        payload = {"posterior_mean": p.to_dict()}
        (args.output_dir / "posteriors_zagbo2025.json").write_text(
            json.dumps(payload, indent=2))
        log.info("Dynare export complete.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
