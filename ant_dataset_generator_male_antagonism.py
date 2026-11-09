"""
Ant polymorphism simulation WITH MALE ANTAGONISM -> ML training dataset

Extension of ant_dataset_generator.py (which is left untouched and still runs the
original model). Every row of the CSV is ONE parameter combination (Latin hypercube
design) with summary statistics of the final allele frequency p(A) over `reps`
replicate simulations.

What is new compared with the original model
--------------------------------------------
* Males are haploid (hemizygous): phenotype zm + sm * allele (no dominance).
  zm and sm are tied to zw and sw, like zq and sq. Male antagonism therefore comes
  only from the male optimum `om` (1 = queen optimum, 0 = worker optimum).
* Males are produced explicitly. Queen q of a deme produces
  Poisson(avg_m_queen * fec_q) sons, so a deme's male number is multiplied by its
  number of (fecund) queens. Each son carries one of its mother's two alleles.
  avg_m_queen is the number of sons of the fittest possible queen (fec = 1).
* Male survival: avg_m_surv * (1 - male_penalty * (z_male - om)^2), clipped to [0, 1].
  avg_m_surv is the survival of a male sitting exactly on the optimum; male_penalty
  (fixed setting, 0-1, default 0.5) softens how hard the distance to the optimum bites.
* Dispersal is sex-specific:
    mq : probability that a new queen is a migrant (born in another deme, mother chosen
         in proportion to deme fecundity, as in the original model). A migrant queen
         also settles in the vacant slot she fills (empty demes can only be refilled
         by migrant queens).
    mm : probability that a surviving male is a migrant (joins the mating flight).
* Mating (each male mates with exactly ONE queen; a queen can have several mates):
    - local males (not migrating) mate only with the LOCAL virgin queens of their own
      deme, chosen uniformly; a migrant queen never mates with them.
    - migrant males form ONE global mating-flight pool shared by ALL virgin queens:
      migrant queens (which only mate in the flight) and local queens of every deme.
      Each migrant male picks one of them uniformly at random.
  Hence full local mating needs a male and a queen of the same deme; full migration
  mating is any migrant male x any migrant queen; local queen x migrant male is also
  possible, migrant queen x local male is not.
* A queen's sperm pool is the fraction of A among her actual mates (variable number,
  `Nm` no longer exists). A virgin queen that receives no male dies; her slot stays
  empty and is refilled in the next generation. Males without an eligible queen are lost.
* Empty slots are tracked with an occupancy mask, so demes can be partly or totally
  empty (and stay empty if no male ever reaches them, e.g. mm = 0).

Targets (one row per parameter set)
-----------------------------------
n_reps            replicates with a living population (basis of the allele statistics)
mean_pA, sd_pA, frac_lost, frac_fixed, frac_polymorphic   over those replicates
frac_deme_alive   mean over ALL replicates of the fraction of demes holding at least
                  one queen when the replicate stopped (0 if the population died out)
If every replicate went extinct, the allele statistics are NaN.

By default a replicate stops as soon as the allele is lost (absorbing state, as in
the original code); frac_deme_alive is then measured at that moment. Use
--no-early-stop to always run the full T generations (slower, but frac_deme_alive is
then always measured at generation T).

Command line (example):

    python ant_dataset_male.py --n-samples 2000 --reps 20 --workers 4 --out ant_male_dataset.csv

The run can be interrupted at any time (kernel stop) and resumed by calling the same
command again: finished rows are already on disk.
"""

from __future__ import annotations
import argparse
import csv
import json
import multiprocessing
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# name: (low, high, is_integer).  Edit these ranges to change the explored space.
# A parameter with low == high is simply held fixed.
PARAM_SPACE: dict[str, tuple[float, float, bool]] = {
    "s":          (0.0, 1.0, False),    # queen survival probability
    "e":          (0.0, 1.0, False),    # deme extinction probability
    "mq":         (0.0, 1.0, False),    # queen dispersal (probability a new queen is a migrant)
    "mm":         (0.0, 1.0, False),    # male dispersal (probability a male joins the mating flight)
    "zw":         (0.0, 1.0, False),    # baseline phenotype (workers, tied to queens and males)
    "dw":         (0.0, 1.0, False),    # heterozygote weight, workers
    "dq":         (0.0, 1.0, False),    # dominance, queens
    "sw":         (-0.1, 0.1, False),   # phenotype scaling (mutation effect; tied to sq, sm)
    "Nq":         (1, 10, True),        # queens per deme
    "om":         (0.0, 1.0, False),    # male optimum (1 = queen optimum, 0 = worker optimum)
    "avg_m_surv": (0.0, 1.0, False),    # male survival at the optimum phenotype
    "avg_m_queen": (100.0, 200.0, False),  # sons of the fittest queen (fec = 1); real queens get avg_m_queen * fec
}

# Parameters forced to equal another sampled one (zq == zm == zw and sq == sm == sw).
# Empty this dict to sample them independently (then also add them to PARAM_SPACE).
TIED: dict[str, str] = {"zq": "zw", "zm": "zw", "sq": "sw", "sm": "sw"}

# Fixed simulation settings (not part of the ML feature space).
# male_penalty (0-1) scales the quadratic survival loss of males away from their optimum:
# 1 = full penalty, 0 = males never pay for being off-optimum (no male selection).
FIXED_DEFAULTS = {"Nd": 200, "T": 1000, "ic": 20, "reps": 100, "male_penalty": 1.0}

FEATURE_COLUMNS = ["s", "e", "mq", "mm", "zw", "zq", "zm", "dw", "dq", "sw", "sq", "sm",
                   "Nq", "om", "avg_m_surv", "avg_m_queen"]
TARGET_COLUMNS = ["n_reps", "mean_pA", "sd_pA", "frac_lost", "frac_fixed",
                  "frac_polymorphic", "frac_deme_alive"]
CSV_COLUMNS = ["sample_id"] + FEATURE_COLUMNS + TARGET_COLUMNS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_params(p: dict) -> tuple[bool, str]:
    """Check one full parameter set before simulating it."""
    for k in ("Nd", "Nq", "reps", "T"):
        if not (isinstance(p.get(k), (int, np.integer)) and p[k] >= 1):
            return False, f"{k} must be an integer >= 1 (got {p.get(k)!r})"
    if not (isinstance(p.get("ic"), (int, np.integer)) and 0 <= p["ic"] <= p["Nd"]):
        return False, f"ic must be an integer in [0, Nd] (got {p.get('ic')!r})"
    for k in ("s", "e", "mq", "mm", "dq", "dw", "avg_m_surv", "male_penalty"):
        v = p.get(k)
        if not (isinstance(v, (int, float, np.floating)) and 0.0 <= v <= 1.0):
            return False, f"{k} must be in [0, 1] (got {v!r})"
    for k in ("zw", "zq", "zm", "sw", "sq", "sm", "om"):
        if not np.isfinite(p.get(k, np.nan)):
            return False, f"{k} must be a finite number (got {p.get(k)!r})"
    v = p.get("avg_m_queen")
    if not (isinstance(v, (int, float, np.floating)) and np.isfinite(v) and v >= 0.0):
        return False, f"avg_m_queen must be a finite number >= 0 (got {v!r})"
    return True, ""


def validate_space(space: dict, tied: dict) -> tuple[bool, str]:
    """Check the sampling ranges and the tie map."""
    for name, (lo, hi, _is_int) in space.items():
        if name not in FEATURE_COLUMNS:
            return False, f"unknown parameter '{name}' in PARAM_SPACE"
        if not lo <= hi:
            return False, f"{name}: low ({lo}) must be <= high ({hi})"
    for child, parent in tied.items():
        if child not in FEATURE_COLUMNS:
            return False, f"unknown tied parameter '{child}'"
        if parent not in space:
            return False, f"'{child}' is tied to '{parent}', which is not in PARAM_SPACE"
    missing = [c for c in FEATURE_COLUMNS if c not in space and c not in tied]
    if missing:
        return False, f"parameters without a range or a tie: {missing}"
    return True, ""


# ---------------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------------

def build_design(n_samples: int, seed: int, space: dict, tied: dict) -> pd.DataFrame:
    """Latin hypercube sample of the parameter space (one row per sample_id)."""
    rng = np.random.default_rng(seed)
    names = list(space)
    n_dim = len(names)

    # Each column is a random permutation of n strata, jittered inside its stratum.
    strata = rng.permuted(np.tile(np.arange(n_samples), (n_dim, 1)), axis=1).T
    u = (strata + rng.random((n_samples, n_dim))) / n_samples

    data = {"sample_id": np.arange(n_samples)}
    for j, name in enumerate(names):
        lo, hi, is_int = space[name]
        if is_int:
            vals = np.minimum(np.floor(lo + u[:, j] * (hi - lo + 1)), hi).astype(int)
        else:
            vals = lo + u[:, j] * (hi - lo)
        data[name] = vals
    for child, parent in tied.items():
        data[child] = data[parent].copy()

    return pd.DataFrame(data)[["sample_id"] + FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def fecundity(phenotype_q, phenotype_w):
    """Eusocial fecundity, vectorised: max(0, (1 - queen) * worker)."""
    return np.maximum(0.0, (1.0 - phenotype_q) * phenotype_w)


def male_survival(z: float, om: float, avg_m_surv: float, male_penalty: float = 1.0) -> float:
    """Survival probability of a male with phenotype z: avg_m_surv at the optimum om,
    minus a quadratic loss softened by male_penalty (0 = no loss, 1 = full loss)."""
    return float(np.clip(avg_m_surv * (1.0 - male_penalty * (z - om) ** 2), 0.0, 1.0))


def simulate_replicate(Nd, Nq, s, e, mq, mm, T, ic, zw, zq, zm, sq, sw, sm, dq, dw,
                       om, avg_m_surv, avg_m_queen, male_penalty, rng: np.random.Generator,
                       stop_on_loss: bool = True) -> tuple[float, float]:
    """
    One replicate. Returns (final mean p(A) over occupied queen slots, fraction of
    demes holding at least one queen). p(A) is NaN if the whole population died out.

    State:  alleles[d, q, 0/1]  maternal / paternal allele (0 or 1) of queen q in deme d
            sperm[d, q]         p(A) of the sperm stored by that queen
            occ[d, q]           True if the slot holds a living, mated queen
    """
    alleles = np.zeros((Nd, Nq, 2))
    sperm = np.zeros((Nd, Nq))
    occ = np.ones((Nd, Nq), dtype=bool)
    worker_pheno = np.full(Nd, zw, dtype=float)
    if ic > 0:
        alleles[:ic, 0, 0] = 1.0  # seed the A allele in queen 1 of the first `ic` demes

    # Hemizygous males: survival depends only on the allele carried (constant in time).
    surv_a = male_survival(zm, om, avg_m_surv, male_penalty)
    surv_A = male_survival(zm + sm, om, avg_m_surv, male_penalty)

    p_A = float(alleles.mean())

    for _ in range(T):
        # ---- 1. Extinction -------------------------------------------------
        occ &= ~(rng.random(Nd) < e)[:, None]
        alive = occ.any(axis=1)                      # demes holding at least one queen

        # ---- 2. Phenotypes and fecundities (all demes at once) -------------
        p_queen = alleles.mean(axis=2)               # queen p(A) (0 in empty slots)
        w_AA = p_queen * sperm
        w_Aa = (1 - p_queen) * sperm + (1 - sperm) * p_queen
        q_pheno = zq + sq * (2 * dq * p_queen + (1 - 2 * dq) * alleles[:, :, 0] * alleles[:, :, 1])

        # Queen weights use the worker phenotype carried over from last generation
        # (-1 in empty demes, giving zero fecundity and hence uniform weights).
        f_first = fecundity(q_pheno, worker_pheno[:, None]) * occ
        f_sum = f_first.sum(axis=1, keepdims=True)
        uniform_occ = occ / np.maximum(occ.sum(axis=1, keepdims=True), 1)
        weights = np.where(f_sum > 0, f_first / np.where(f_sum > 0, f_sum, 1.0), uniform_occ)

        p_AA = (w_AA * weights).sum(axis=1)
        p_Aa = (w_Aa * weights).sum(axis=1)
        worker_pheno = np.where(alive, zw + p_AA * sw + p_Aa * (dw * sw), -1.0)

        fec = fecundity(q_pheno, worker_pheno[:, None]) * occ      # (Nd, Nq), 0 if empty
        deme_fec = fec.sum(axis=1)

        # ---- 3. Males: production (scaled by queen number), survival -------
        sons = rng.poisson(avg_m_queen * fec)                      # sons per queen
        sons_A = rng.binomial(sons, p_queen)                       # each son gets one maternal allele
        m_A = sons_A.sum(axis=1)                                   # per deme
        m_a = sons.sum(axis=1) - m_A
        m_A = rng.binomial(m_A, surv_A)
        m_a = rng.binomial(m_a, surv_a)

        # ---- 4. Male dispersal: migrants leave for the global mating-flight pool
        # m_A / m_a (per deme) keep only the LOCAL males, who mate in their own deme.
        if mm > 0.0:
            fly_A = rng.binomial(m_A, mm)
            fly_a = rng.binomial(m_a, mm)
            m_A, m_a = m_A - fly_A, m_a - fly_a
            pool_A, pool_a = int(fly_A.sum()), int(fly_a.sum())    # migrant males, all demes
        else:
            pool_A = pool_a = 0

        # ---- 5. Queen turnover: which slots are vacant ----------------------
        survive = occ & (rng.random((Nd, Nq)) < s)
        vacant = ~survive
        vi, vk = np.nonzero(vacant)                                # vacant slot coordinates
        n_v = vi.size

        # ---- 6. Virgin daughter queens for every vacant slot ----------------
        # Mother's deme: local (deme still alive) or, with prob. mq, from another deme
        # drawn in proportion to deme fecundity. Slots whose deme is empty must use an
        # immigrant mother. If no mother is available the slot stays empty.
        w_dem = np.where(alive, deme_fec, 0.0)
        if w_dem.sum() <= 0.0:
            w_dem = alive.astype(float)                            # all fecundities 0: uniform
        cum = np.cumsum(w_dem)
        w_own = w_dem[vi]
        other_ok = (cum[-1] - w_own) > 1e-12                       # another deme can supply a mother
        local_ok = alive[vi]
        go_imm = other_ok & ((rng.random(n_v) < mq) | ~local_ok)
        made = go_imm | local_ok                                   # daughter can be produced

        src = vi.copy()                                            # local by default
        im = np.flatnonzero(go_imm)
        if im.size:                                                # weighted draw excluding own deme
            u = rng.random(im.size) * (cum[-1] - w_own[im])
            u = np.where(u >= cum[vi[im]] - w_own[im], u + w_own[im], u)
            src[im] = np.minimum(np.searchsorted(cum, u, side="right"), Nd - 1)

        d_idx = np.flatnonzero(made)                               # daughters actually produced
        vi_d, vk_d, src_d = vi[d_idx], vk[d_idx], src[d_idx]
        migrant_q = go_imm[d_idx]                                  # True: migrant queen (mates in the flight only)

        # Mother queen inside the source deme: fecundity-weighted (uniform if all 0).
        src_w = np.where((deme_fec > 0)[:, None], fec, occ.astype(float))
        row = src_w.sum(axis=1, keepdims=True)
        cdf_q = np.cumsum(src_w / np.where(row > 0, row, 1.0), axis=1)
        q_src = np.minimum((rng.random(d_idx.size)[:, None] >= cdf_q[src_d]).sum(axis=1), Nq - 1)

        # Daughter genotype: one maternal allele, one paternal allele from mother's sperm.
        mat = alleles[src_d, q_src, rng.integers(0, 2, d_idx.size)]
        pat = (rng.random(d_idx.size) < sperm[src_d, q_src]).astype(float)

        # ---- 7. Mating: each male picks ONE virgin queen --------------------
        # (a) LOCAL males mate only with the LOCAL virgin queens of their own deme.
        local_virgin = np.zeros((Nd, Nq), dtype=bool)
        local_virgin[vi_d[~migrant_q], vk_d[~migrant_q]] = True
        n_local_virgin = local_virgin.sum(axis=1)

        def allocate(n_males: np.ndarray) -> np.ndarray:
            """Spread n_males[d] males uniformly over the local virgin slots of deme d
            (sequential binomials = multinomial with equal probabilities)."""
            out = np.zeros((Nd, Nq), dtype=np.int64)
            remaining = n_males.astype(np.int64)
            left = n_local_virgin.astype(np.int64)
            for k in range(Nq):
                is_v = local_virgin[:, k]
                p = np.where(is_v & (left > 0), 1.0 / np.maximum(left, 1), 0.0)
                take = rng.binomial(remaining, p)
                out[:, k] = take
                remaining = remaining - take
                left = left - is_v
            return out                                            # males in demes without local virgins are lost

        local_A = allocate(m_A)
        local_n = local_A + allocate(m_a)
        a_d = local_A[vi_d, vk_d]                                  # per daughter: local A mates
        n_d = local_n[vi_d, vk_d]                                  # per daughter: local mates

        # (b) MIGRANT males: one global pool shared by ALL virgin queens (migrant queens
        # and the local queens of every deme); each male picks one uniformly at random.
        n_virgin = d_idx.size
        if n_virgin and (pool_A + pool_a) > 0:
            equal = np.full(n_virgin, 1.0 / n_virgin)
            fly_mates_A = rng.multinomial(pool_A, equal)
            fly_mates_a = rng.multinomial(pool_a, equal)
            a_d = a_d + fly_mates_A
            n_d = n_d + fly_mates_A + fly_mates_a

        # ---- 8. Write the new generation ------------------------------------
        mated = n_d > 0                                            # unmated queens die

        alleles[vacant] = 0.0
        sperm[vacant] = 0.0
        occ = survive                                              # surviving old queens keep their slot
        di, dk = vi_d[mated], vk_d[mated]
        alleles[di, dk, 0] = mat[mated]
        alleles[di, dk, 1] = pat[mated]
        sperm[di, dk] = a_d[mated] / n_d[mated]
        occ[di, dk] = True

        if not occ.any():                                          # population extinct
            return float("nan"), 0.0
        p_A = float(alleles[occ].mean())
        if stop_on_loss and p_A == 0.0:                            # allele lost: absorbing state
            break

    return p_A, float(occ.any(axis=1).mean())


def final_frequencies(params: dict, rng: np.random.Generator,
                      stop_on_loss: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(final p(A), fraction of living demes) of `params['reps']` independent replicates."""
    keys = ("Nd", "Nq", "s", "e", "mq", "mm", "T", "ic", "zw", "zq", "zm", "sq", "sw", "sm",
            "dq", "dw", "om", "avg_m_surv", "avg_m_queen", "male_penalty")
    kwargs = {k: params[k] for k in keys}
    out = [simulate_replicate(**kwargs, rng=rng, stop_on_loss=stop_on_loss)
           for _ in range(params["reps"])]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def summarise(finals: np.ndarray, alive_frac: np.ndarray) -> dict:
    """Targets for one parameter set (allele statistics use replicates that did not go extinct)."""
    valid = finals[~np.isnan(finals)]
    out = {"n_reps": int(valid.size), "frac_deme_alive": float(alive_frac.mean())}
    if valid.size == 0:
        out.update(mean_pA=np.nan, sd_pA=np.nan, frac_lost=np.nan,
                   frac_fixed=np.nan, frac_polymorphic=np.nan)
        return out
    out.update(
        mean_pA=float(valid.mean()),
        sd_pA=float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        frac_lost=float(np.mean(valid == 0.0)),
        frac_fixed=float(np.mean(valid == 1.0)),
        frac_polymorphic=float(np.mean((valid > 0.0) & (valid < 1.0))),
    )
    return out


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _run_sample(task: tuple) -> tuple:
    """Pool worker. Returns ('ok', sample_id, row) or ('error', sample_id, message)."""
    sample_id, params, base_seed, stop_on_loss = task
    try:
        ok, msg = validate_params(params)
        if not ok:
            return "error", sample_id, msg
        rng = np.random.default_rng(np.random.SeedSequence(entropy=base_seed, spawn_key=(sample_id,)))
        row = {"sample_id": sample_id, **{c: params[c] for c in FEATURE_COLUMNS}}
        row.update(summarise(*final_frequencies(params, rng, stop_on_loss)))
        return "ok", sample_id, row
    except Exception as exc:  # keep the batch alive, report the bad sample instead
        return "error", sample_id, f"{type(exc).__name__}: {exc}"


def _to_python(value):
    return value.item() if isinstance(value, np.generic) else value


def generate_dataset(n_samples: int, out_path: str | None = None, seed: int = 0,
                     Nd: int = FIXED_DEFAULTS["Nd"], T: int = FIXED_DEFAULTS["T"],
                     ic: int = FIXED_DEFAULTS["ic"], reps: int = FIXED_DEFAULTS["reps"],
                     workers: int | None = None, resume: bool = True,
                     space: dict | None = None, tied: dict | None = None,
                     stop_on_loss: bool = True,
                     male_penalty: float = FIXED_DEFAULTS["male_penalty"]) -> tuple[bool, str]:
    """
    Build the ML dataset. Returns (ok, message); never raises on bad input.

    n_samples    : number of parameter combinations (rows), e.g. 3000
    out_path     : CSV location. Default: ant_male_dataset_<n_samples>.csv in the current folder
    reps         : replicate simulations per combination (more reps = less noisy targets)
    workers      : processes to use (default: all CPU cores; 1 = no multiprocessing)
    resume       : continue an interrupted run with identical settings
    stop_on_loss : stop a replicate once the allele is lost (faster; frac_deme_alive is then
                   measured at that moment). False = always run T generations.
    male_penalty : 0-1, softens the male survival loss away from the optimum (see docstring)
    """
    space = PARAM_SPACE if space is None else space
    tied = TIED if tied is None else tied
    if not (isinstance(n_samples, int) and n_samples >= 1):
        return False, "n_samples must be an integer >= 1"
    out = Path(out_path or f"ant_male_dataset_{n_samples}.csv").expanduser().resolve()
    meta_path = out.with_suffix(out.suffix + ".meta.json")

    ok, msg = validate_space(space, tied)
    if not ok:
        return False, f"Invalid parameter space: {msg}"
    fixed = {"Nd": Nd, "T": T, "ic": ic, "reps": reps, "male_penalty": male_penalty}
    ok, msg = validate_params({**{c: 0.5 for c in FEATURE_COLUMNS}, "Nq": 1, **fixed})
    if not ok:
        return False, f"Invalid fixed settings: {msg}"

    config = {"n_samples": n_samples, "seed": seed, **fixed, "stop_on_loss": stop_on_loss,
              "space": {k: list(v) for k, v in space.items()}, "tied": tied}

    # --- resume / overwrite handling ---
    done_ids: set[int] = set()
    if out.exists() and out.stat().st_size > 0:
        if not resume:
            out.unlink()
            meta_path.unlink(missing_ok=True)
        else:
            old = json.loads(meta_path.read_text()) if meta_path.exists() else None
            if old != config:
                return False, (f"{out} exists but was made with different settings. "
                               "Use a new file name, or resume=False to overwrite.")
            try:
                done_ids = set(pd.read_csv(out, usecols=["sample_id"])["sample_id"].tolist())
            except Exception as exc:
                return False, f"Could not read the existing CSV ({exc}); use resume=False to restart."
    meta_path.write_text(json.dumps(config, indent=2))

    design = build_design(n_samples, seed, space, tied)
    tasks = []
    for rec in design.to_dict("records"):
        sid = int(rec["sample_id"])
        if sid in done_ids:
            continue
        params = {k: _to_python(v) for k, v in rec.items() if k != "sample_id"}
        tasks.append((sid, {**params, **fixed}, seed, stop_on_loss))

    if not tasks:
        return True, f"Nothing to do: all {n_samples} rows already in {out}."

    workers = max(1, workers or os.cpu_count() or 1)
    # Worker processes must be able to find _run_sample. If this code was pasted into a
    # notebook cell (instead of imported), spawned workers cannot, so use 'fork' there.
    pool_ctx = multiprocessing.get_context()
    if workers > 1 and _run_sample.__module__ == "__main__" and _running_in_notebook():
        if "fork" in multiprocessing.get_all_start_methods():
            pool_ctx = multiprocessing.get_context("fork")
        else:
            workers = 1
            print("Note: running on 1 process because this code lives in a notebook cell. "
                  "Save it as ant_dataset_male.py next to the notebook and import it to use several workers.")
    print(f"Output file : {out}")
    print(f"Combinations: {n_samples} total, {len(done_ids)} already done, {len(tasks)} to run "
          f"({reps} replicates each, {workers} worker(s))")
    t_start = time.perf_counter()
    skipped: list[tuple[int, str]] = []
    write_header = not out.exists() or out.stat().st_size == 0

    try:
        with open(out, "a", newline="") as fh, \
                tqdm(total=n_samples, initial=len(done_ids), desc="Generating", unit="set",
                     dynamic_ncols=True) as bar:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()

            def handle(result: tuple) -> None:
                status, sid, payload = result
                if status == "ok":
                    writer.writerow(payload)
                    fh.flush()
                else:
                    skipped.append((sid, payload))
                    warnings.warn(f"sample {sid} skipped: {payload}")
                bar.update(1)
                rate = bar.format_dict.get("rate")
                post = {"skipped": len(skipped)}
                if rate:
                    eta = datetime.now() + timedelta(seconds=(bar.total - bar.n) / rate)
                    post["done_at"] = eta.strftime("%a %H:%M" if (eta.date() != datetime.now().date()) else "%H:%M")
                bar.set_postfix(post)

            if workers == 1:
                for task in tasks:
                    handle(_run_sample(task))
            else:
                with pool_ctx.Pool(workers) as pool:
                    for result in pool.imap_unordered(_run_sample, tasks, chunksize=1):
                        handle(result)
    except KeyboardInterrupt:
        return False, f"Interrupted. Finished rows are saved in {out}; run again to resume."

    n_rows = n_samples - len(skipped)
    elapsed = timedelta(seconds=int(time.perf_counter() - t_start))
    msg = f"Saved {n_rows} rows to {out} (this run took {elapsed})."
    if skipped:
        msg += f" {len(skipped)} sample(s) skipped (ids: {[s for s, _ in skipped][:10]}...)."
    return True, msg


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the ant-polymorphism ML dataset (male antagonism).")
    ap.add_argument("--n-samples", type=int, required=True, help="parameter combinations (rows)")
    ap.add_argument("--out", default=None, help="default: ant_male_dataset_<n>.csv in the current folder")
    ap.add_argument("--reps", type=int, default=FIXED_DEFAULTS["reps"])
    ap.add_argument("--T", type=int, default=FIXED_DEFAULTS["T"])
    ap.add_argument("--Nd", type=int, default=FIXED_DEFAULTS["Nd"])
    ap.add_argument("--ic", type=int, default=FIXED_DEFAULTS["ic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true", help="overwrite an existing CSV")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="always run T generations (do not stop when the allele is lost)")
    ap.add_argument("--male-penalty", type=float, default=FIXED_DEFAULTS["male_penalty"],
                    help="0-1: softens male survival loss away from the optimum (default 0.5)")
    a = ap.parse_args()

    ok, msg = generate_dataset(a.n_samples, a.out, a.seed, a.Nd, a.T, a.ic, a.reps,
                               a.workers, resume=not a.no_resume,
                               stop_on_loss=not a.no_early_stop,
                               male_penalty=a.male_penalty)
    print(msg)
    raise SystemExit(0 if ok else 1)


def _running_in_notebook() -> bool:
    """True inside Jupyter / VS Code notebooks, where the CLI must not start."""
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore[name-defined]
    except NameError:
        return False


if __name__ == "__main__" and not _running_in_notebook():
    main()
