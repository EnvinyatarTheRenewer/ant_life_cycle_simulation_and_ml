"""
Ant polymorphism simulation with male selection on the allele fraction. -> ML dataset

Back to the structure of the original ant_dataset_generator.py (Nm matings per new queen,
vacant slots and extinct demes refilled in the same generation, so no deme stays empty),
plus male antagonism. Every CSV row is ONE parameter combination (Latin hypercube design)
with summary statistics of the final allele frequency p(A) over `reps` replicates.

How males work (no explicit males, hence no male-number parameters)
--------------------------------------------------------------------
A new queen mates with Nm males. Each mate is drawn from one of two male pools:
    global pool : fecundity-weighted queens of all living demes          (prob. mm)
    local pool  : the queens of the new queen's own deme (unweighted)    (prob. 1 - mm)
(a deme that just went extinct has no local males: it uses the global pool).
A son carries the allele of a random one of his mother's two alleles, so the fraction of A
males in a pool is f = mean queen p(A), exactly as in the original code.

Male selection reweights that fraction before mating, with the relative survival of the two
male classes (a = ancestral, A = de novo allele):

    surv(z) = 1 - min(|om - z|, 1) ** male_exponent            (clipped to [0, 1])
    surv_a  = surv(zm)          surv_A = surv(zm + sm)
    f'      = f * surv_A / ( f * surv_A + (1 - f) * surv_a )

i.e. haploid selection with relative fitness surv_A / surv_a: the A fraction is multiplied by
the survival index and the pool is renormalised. If no male would survive (denominator 0),
f is left unchanged. Absolute male survival cancels out of a frequency, so no male survival
or male number parameter is needed. Each of the Nm mates is then A with probability
    p = mm * f'_global + (1 - mm) * f'_local          ->  sperm p(A) = Binomial(Nm, p) / Nm.

om is the male optimum (1 = queen optimum, 0 = worker optimum); zm, sm are tied to zw, sw
(like zq, sq), so male antagonism comes only from om. male_exponent (fixed setting, default 1)
sets how fast survival falls with the distance to the optimum (2 = quadratic, 0.5 = sqrt).

Queens: mq is the original `m` (migration weight): a vacant slot is filled by a migrant
queen with probability m*imm / (m*imm + (1-m)*local) from the mean fecundities, else by a
local one. Sperm origin is now independent of queen origin (it is set by mm).

Deliberate differences from ant_dataset_generator.py
----------------------------------------------------
* mm (male dispersal) is new; the original tied sperm origin to queen origin.
* A new local queen's paternal allele is read from the generation before the update. The
  original read the partly updated deme, and for an extinct deme it read the wiped deme
  (so such a queen could never get an A paternal allele and always got sperm 0): with
  mq = 0 and e > 0 the original therefore under-estimates A; here an extinct deme refills
  from its own pre-extinction queens, with sperm from the global male pool.
* If every deme goes extinct in the same generation, that generation is rolled back (the
  original restarted from the pre-extinction deme with sperm 0). It only matters for e ~ 1.
* Early stopping is exact: a replicate stops only when the allele is fixed (all alleles AND
  all stored sperm are A) or lost (no A allele AND no A sperm stored), states it could never
  leave, so the result equals running to T. The original stopped as soon as no queen
  carried A, even while A sperm was still stored (which can re-introduce A), so frac_lost
  is a little lower here (mean_pA is unchanged within noise).

Targets: n_reps, mean_pA, sd_pA, frac_lost, frac_fixed, frac_polymorphic (as in the original).

The run can be interrupted at any time and resumed by calling the same command again (using resume = True).
"""

### --- Import Useful libraries --- ###

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

### --- Configuration --- ###

# name: (low, high, is_integer).  Edit these ranges to change the explored space.
# A parameter with low == high is simply held fixed.
PARAM_SPACE: dict[str, tuple[float, float, bool]] = {
    "s":  (0.0, 1.0, False),    # Queen survival probability.
    "e":  (0.0, 1.0, False),    # Deme extinction probability.
    "mq": (0.0, 1.0, False),    # Queen migration weight (the original `m`).
    "mm": (0.0, 1.0, False),    # Male dispersal: probability a mate comes from the global pool.
    "zw": (0.0, 1.0, False),    # Baseline phenotype (workers; tied to queens and males).
    "dw": (0.0, 1.0, False),    # Dominance in workers.
    "dq": (0.0, 1.0, False),    # Dominance in queens.
    "sw": (-0.1, 0.1, False),   # Phenotype scaling (mutation effect; tied to sq, sm).
    "Nq": (1, 10, True),        # Queens per deme.
    "Nm": (1, 20, True),        # Matings per new queen.
    "om": (0.0, 1.0, False),    # Male optimum (1 = queen optimum, 0 = worker optimum).
}

# Parameters forced to equal another sampled one. Empty this dict to sample them
# independently (then also add them to PARAM_SPACE).
TIED: dict[str, str] = {"zq": "zw", "zm": "zw", "sq": "sw", "sm": "sw"}

# Fixed simulation settings (not part of the ML feature space).
# male_exponent: survival = 1 - distance ** male_exponent (1 = linear, 2 = quadratic).
FIXED_DEFAULTS = {"Nd": 100, "T": 1000, "ic": 20, "reps": 100, "male_exponent": 1.0}

FEATURE_COLUMNS = ["s", "e", "mq", "mm", "zw", "zq", "zm", "dw", "dq", "sw", "sq", "sm",
                   "Nq", "Nm", "om"]
TARGET_COLUMNS = ["n_reps", "mean_pA", "sd_pA", "frac_lost", "frac_fixed", "frac_polymorphic"]
CSV_COLUMNS = ["sample_id"] + FEATURE_COLUMNS + TARGET_COLUMNS

### ---  Validation of Parameters --- ###

def validate_params(p: dict) -> tuple[bool, str]:
    """Check one full parameter set before simulating it."""
    for k in ("Nd", "Nq", "Nm", "reps", "T"):
        if not (isinstance(p.get(k), (int, np.integer)) and p[k] >= 1):
            return False, f"{k} must be an integer >= 1 (got {p.get(k)!r})"
    if not (isinstance(p.get("ic"), (int, np.integer)) and 0 <= p["ic"] <= p["Nd"]):
        return False, f"ic must be an integer in [0, Nd] (got {p.get('ic')!r})"
    for k in ("s", "e", "mq", "mm", "dq", "dw"):
        v = p.get(k)
        if not (isinstance(v, (int, float, np.floating)) and 0.0 <= v <= 1.0):
            return False, f"{k} must be in [0, 1] (got {v!r})"
    for k in ("zw", "zq", "zm", "sw", "sq", "sm", "om"):
        if not np.isfinite(p.get(k, np.nan)):
            return False, f"{k} must be a finite number (got {p.get(k)!r})"
    v = p.get("male_exponent")
    if not (isinstance(v, (int, float, np.floating)) and np.isfinite(v) and v > 0.0):
        return False, f"male_exponent must be a finite number > 0 (got {v!r})"
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

### --- Experimental Design -- ###

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

### --- Simulation Core --- ###

def fecundity(phenotype_q, phenotype_w):
    """Eusocial fecundity, vectorised: max(0, (1 - queen) * worker)."""
    return np.maximum(0.0, (1.0 - phenotype_q) * phenotype_w)

def male_survival(z: float, om: float, male_exponent: float) -> float:
    """Male survival index: 1 - min(|om - z|, 1) ** exponent, in [0, 1]."""
    d = min(abs(om - z), 1.0)
    return max(0.0, 1.0 - d ** male_exponent)

def select_fraction(f, surv_A: float, surv_a: float):
    """Fraction of A males after selection: f*sA / (f*sA + (1-f)*sa); unchanged if no male survives."""
    f = np.asarray(f, dtype=float)
    den = f * surv_A + (1.0 - f) * surv_a
    return np.where(den > 0.0, f * surv_A / np.where(den > 0.0, den, 1.0), f)

def simulate_replicate(Nd, Nq, Nm, s, e, mq, mm, T, ic, zw, zq, zm, sq, sw, sm, dq, dw,
                       om, male_exponent, rng: np.random.Generator,
                       early_stop: bool = True) -> float:
    """
    One replicate: returns the final mean p(A) across all demes and queen slots.

    alleles[d, q, 0/1] maternal / paternal allele (0/1) of queen q in deme d
    sperm[d, q]        p(A) of the sperm stored by that queen
    Every slot always holds a queen: vacant slots are refilled in the same generation.
    """
    alleles = np.zeros((Nd, Nq, 2))
    sperm = np.zeros((Nd, Nq))
    worker_pheno = np.full(Nd, zw, dtype=float)
    if ic > 0:
        alleles[:ic, 0, 0] = 1.0  # seed the A allele in queen 1 of the first `ic` demes

    surv_a = male_survival(zm, om, male_exponent)
    surv_A = male_survival(zm + sm, om, male_exponent)
    p_A = float(alleles.mean())

    for _ in range(T):
        # ---- 1. Extinction -------------------------------------------------
        extinct = rng.random(Nd) < e
        alive = ~extinct
        if not alive.any():          # Every deme extinct at once: roll this generation back.
            continue

        # ---- 2. Phenotypes and fecundities (all demes at once) -------------
        p_queen = alleles.mean(axis=2)
        w_AA = p_queen * sperm
        w_Aa = (1 - p_queen) * sperm + (1 - sperm) * p_queen
        q_pheno = zq + sq * (2 * dq * p_queen + (1 - 2 * dq) * alleles[:, :, 0] * alleles[:, :, 1])

        # Queen weights use the worker phenotype carried over from last generation.
        f_first = fecundity(q_pheno, worker_pheno[:, None])
        f_sum = f_first.sum(axis=1, keepdims=True)
        weights = np.where(f_sum > 0, f_first / np.where(f_sum > 0, f_sum, 1.0), 1.0 / Nq)

        p_AA = (w_AA * weights).sum(axis=1)
        p_Aa = (w_Aa * weights).sum(axis=1)
        worker_pheno = np.where(alive, zw + p_AA * sw + p_Aa * (dw * sw), worker_pheno)

        fec = fecundity(q_pheno, worker_pheno[:, None])
        fec[extinct] = 0.0
        worker_pheno[extinct] = -1.0

        # ---- 3. Which slots are refilled -----------------------------------
        replace = extinct[:, None] | (rng.random((Nd, Nq)) > s)
        vi, vk = np.nonzero(replace)
        n = vi.size
        if n == 0:
            continue

        # ---- 4. Mother of each new queen: local, or migrant from another deme
        dmf = fec.mean(axis=1)                                    # Seme mean fecundity (0 if extinct).
        cnt_oth = alive.sum() - alive                             # Living demes other than d.
        imm_mean = np.where(cnt_oth > 0, (dmf.sum() - dmf) / np.maximum(cnt_oth, 1), 0.0)
        denom = mq * imm_mean + (1 - mq) * dmf
        p_imm = np.where(denom > 0, mq * imm_mean / np.where(denom > 0, denom, 1.0), 0.5)

        cum = np.cumsum(dmf)
        w_own = dmf[vi]
        can_imm = (cum[-1] - w_own) > 1e-12                       # Another deme can supply a mother.
        go_imm = can_imm & (rng.random(n) < p_imm[vi])

        src = vi.copy()                                           # Local mother by default.
        im = np.flatnonzero(go_imm)
        if im.size:                                               # Fecundity-weighted deme, own excluded.
            u = rng.random(im.size) * (cum[-1] - w_own[im])
            u = np.where(u >= cum[vi[im]] - w_own[im], u + w_own[im], u)
            src[im] = np.minimum(np.searchsorted(cum, u, side="right"), Nd - 1)

        # Mother queen inside the source deme: fecundity-weighted (uniform if all 0).
        src_w = np.where((fec.sum(axis=1) > 0)[:, None], fec, 1.0)
        cdf_q = np.cumsum(src_w / src_w.sum(axis=1, keepdims=True), axis=1)
        q_src = np.minimum((rng.random(n)[:, None] >= cdf_q[src]).sum(axis=1), Nq - 1)

        # Daughter genotype: one maternal allele, one paternal allele from the mother's sperm.
        mat = alleles[src, q_src, rng.integers(0, 2, n)]
        pat = (rng.random(n) < sperm[src, q_src]).astype(float)

        # ---- 5. Males: pool fractions, selection, mating ---------------------
        tot = fec.sum()
        f_glob = (fec * p_queen).sum() / tot if tot > 0 else p_queen[alive].mean()
        f_loc = np.where(alive, p_queen.mean(axis=1), f_glob)     # Extinct deme: no local males.
        p_mix = mm * select_fraction(f_glob, surv_A, surv_a) + (1 - mm) * select_fraction(f_loc, surv_A, surv_a)
        hits = rng.binomial(Nm, np.clip(p_mix, 0.0, 1.0)[vi])

        # ---- 6. Write the new queens ------------------------------------------
        alleles[vi, vk, 0] = mat
        alleles[vi, vk, 1] = pat
        sperm[vi, vk] = hits / Nm

        p_A = float(alleles.mean())
        if early_stop:                                            # Absorbing states.
            if p_A == 0.0 and not sperm.any():
                break
            if p_A == 1.0 and sperm.min() == 1.0:
                break

    return p_A

def final_frequencies(params: dict, rng: np.random.Generator, early_stop: bool = True) -> np.ndarray:
    """Final p(A) of `params['reps']` independent replicates."""
    keys = ("Nd", "Nq", "Nm", "s", "e", "mq", "mm", "T", "ic", "zw", "zq", "zm", "sq", "sw", "sm",
            "dq", "dw", "om", "male_exponent")
    kwargs = {k: params[k] for k in keys}
    return np.array([simulate_replicate(**kwargs, rng=rng, early_stop=early_stop)
                     for _ in range(params["reps"])])

def summarise(finals: np.ndarray) -> dict:
    """Targets for one parameter set."""
    return {
        "n_reps": int(finals.size),
        "mean_pA": float(finals.mean()),
        "sd_pA": float(finals.std(ddof=1)) if finals.size > 1 else 0.0,
        "frac_lost": float(np.mean(finals == 0.0)),
        "frac_fixed": float(np.mean(finals == 1.0)),
        "frac_polymorphic": float(np.mean((finals > 0.0) & (finals < 1.0))),
    }

### --- Workers --- ###

def _run_sample(task: tuple) -> tuple:
    """Pool worker. Returns ('ok', sample_id, row) or ('error', sample_id, message)."""
    sample_id, params, base_seed, early_stop = task
    try:
        ok, msg = validate_params(params)
        if not ok:
            return "error", sample_id, msg
        rng = np.random.default_rng(np.random.SeedSequence(entropy=base_seed, spawn_key=(sample_id,)))
        row = {"sample_id": sample_id, **{c: params[c] for c in FEATURE_COLUMNS}}
        row.update(summarise(final_frequencies(params, rng, early_stop)))
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
                     early_stop: bool = True,
                     male_exponent: float = FIXED_DEFAULTS["male_exponent"]) -> tuple[bool, str]:
    """
    Build the ML dataset. Returns (ok, message); never raises on bad input.

    n_samples     : number of parameter combinations (rows), e.g. 3000
    out_path      : CSV location. Default: ant_male_freq_dataset_<n_samples>.csv in the current folder
    reps          : replicate simulations per combination (more reps = less noisy targets)
    workers       : processes to use (default: all CPU cores; 1 = no multiprocessing)
    resume        : continue an interrupted run with identical settings
    early_stop    : stop a replicate once the allele is fixed or lost (absorbing states)
    male_exponent : male survival = 1 - distance ** male_exponent (see module docstring)
    """
    space = PARAM_SPACE if space is None else space
    tied = TIED if tied is None else tied
    if not (isinstance(n_samples, int) and n_samples >= 1):
        return False, "n_samples must be an integer >= 1"
    out = Path(out_path or f"ant_male_freq_dataset_{n_samples}.csv").expanduser().resolve()
    meta_path = out.with_suffix(out.suffix + ".meta.json")

    ok, msg = validate_space(space, tied)
    if not ok:
        return False, f"Invalid parameter space: {msg}"
    fixed = {"Nd": Nd, "T": T, "ic": ic, "reps": reps, "male_exponent": male_exponent}
    ok, msg = validate_params({**{c: 0.5 for c in FEATURE_COLUMNS}, "Nq": 1, "Nm": 1, **fixed})
    if not ok:
        return False, f"Invalid fixed settings: {msg}"

    config = {"n_samples": n_samples, "seed": seed, **fixed, "early_stop": early_stop,
              "space": {k: list(v) for k, v in space.items()}, "tied": tied}

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
        tasks.append((sid, {**params, **fixed}, seed, early_stop))

    if not tasks:
        return True, f"Nothing to do: all {n_samples} rows already in {out}."

    workers = max(1, workers or os.cpu_count() or 1)
    pool_ctx = multiprocessing.get_context()
    if workers > 1 and _run_sample.__module__ == "__main__" and _running_in_notebook():
        if "fork" in multiprocessing.get_all_start_methods():
            pool_ctx = multiprocessing.get_context("fork")
        else:
            workers = 1
            print("Note: running on 1 process because this code lives in a notebook cell. "
                  "Save it as ant_dataset_male_freq.py next to the notebook and import it to use several workers.")
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

### --- Command Lines --- ###

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the ant-polymorphism ML dataset (male selection on the allele fraction).")
    ap.add_argument("--n-samples", type=int, required=True, help="parameter combinations (rows)")
    ap.add_argument("--out", default=None, help="default: ant_male_freq_dataset_<n>.csv in the current folder")
    ap.add_argument("--reps", type=int, default=FIXED_DEFAULTS["reps"])
    ap.add_argument("--T", type=int, default=FIXED_DEFAULTS["T"])
    ap.add_argument("--Nd", type=int, default=FIXED_DEFAULTS["Nd"])
    ap.add_argument("--ic", type=int, default=FIXED_DEFAULTS["ic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true", help="overwrite an existing CSV")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="always run T generations (do not stop at fixation or loss)")
    ap.add_argument("--male-exponent", type=float, default=FIXED_DEFAULTS["male_exponent"],
                    help="male survival = 1 - distance ** exponent (default 1 = linear)")
    a = ap.parse_args()

    ok, msg = generate_dataset(a.n_samples, a.out, a.seed, a.Nd, a.T, a.ic, a.reps,
                               a.workers, resume=not a.no_resume,
                               early_stop=not a.no_early_stop, male_exponent=a.male_exponent)
    print(msg)
    raise SystemExit(0 if ok else 1)

def _running_in_notebook() -> bool:
    """True inside Jupyter / VS Code notebooks, where the CLI must not start."""
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False

if __name__ == "__main__" and not _running_in_notebook():
    main()