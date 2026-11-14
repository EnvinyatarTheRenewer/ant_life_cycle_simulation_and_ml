"""
Ant polymorphism simulation -> ML training dataset

Generates a CSV in which every row is one parameter combination (sampled with a
space-filling Latin hypercube design) together with summary statistics of the
final allele frequency p(A) over `reps` replicate simulations.

The run can be interrupted at any time (kernel stop) and resumed (using resume = True) 
by calling the same command again: finished rows are already on disk.
"""

### --- Useful Libraries --- ###

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
    "m":  (0.0, 1.0, False),    # Migration weight.
    "zw": (0.0, 1.0, False),    # Baseline worker phenotype (= queen).
    "dw": (0.0, 1.0, False),    # Dominance for workers.
    "dq": (0.0, 1.0, False),    # Dominance for queens.
    "sw": (-0.1, 0.1, False),   # Phenotype scaling (mutation effect).
    "Nq": (1, 10, True),        # Queens per deme.
    "Nm": (1, 20, True),        # Matings per new queen.
}

# Parameters forced to equal another sampled one zq == zw and sq == sw). 
# Empty this dict to sample them independently (then also add "zq" and 
# "sq" to PARAM_SPACE).
TIED: dict[str, str] = {"zq": "zw", "sq": "sw"}

# Fixed simulation settings (not part of the ML feature space).
FIXED_DEFAULTS = {"Nd": 100, "T": 1000, "ic": 20, "reps": 100}

FEATURE_COLUMNS = ["s", "e", "m", "zw", "zq", "dw", "dq", "sw", "sq", "Nq", "Nm"]
TARGET_COLUMNS = ["n_reps", "mean_pA", "sd_pA", "frac_lost", "frac_fixed", "frac_polymorphic"]
CSV_COLUMNS = ["sample_id"] + FEATURE_COLUMNS + TARGET_COLUMNS

### --- Parameter Validation --- ###

def validate_params(p: dict) -> tuple[bool, str]:
    """Check one full parameter set before simulating it."""
    for k in ("Nd", "Nq", "Nm", "reps", "T"):
        if not (isinstance(p.get(k), (int, np.integer)) and p[k] >= 1):
            return False, f"{k} must be an integer >= 1 (got {p.get(k)!r})"
    if not (isinstance(p.get("ic"), (int, np.integer)) and 0 <= p["ic"] <= p["Nd"]):
        return False, f"ic must be an integer in [0, Nd] (got {p.get('ic')!r})"
    for k in ("s", "e", "m", "dq", "dw"):
        v = p.get(k)
        if not (isinstance(v, (int, float, np.floating)) and 0.0 <= v <= 1.0):
            return False, f"{k} must be in [0, 1] (got {v!r})"
    for k in ("zw", "zq", "sw", "sq"):
        if not np.isfinite(p.get(k, np.nan)):
            return False, f"{k} must be a finite number (got {p.get(k)!r})"
    return True, ""

def validate_space(space: dict, tied: dict) -> tuple[bool, str]:
    """Check the sampling ranges and the tie map."""
    for name, (lo, hi, _is_int) in space.items():
        if name not in FEATURE_COLUMNS:
            return False, f"unknown parameter '{name}' in PARAM_SPACE"
        if not lo <= hi:
            return False, f"{name}: low ({lo}) must be <= high ({hi})"
    for child, parent in tied.items():
        if parent not in space:
            return False, f"'{child}' is tied to '{parent}', which is not in PARAM_SPACE"
    missing = [c for c in FEATURE_COLUMNS if c not in space and c not in tied]
    if missing:
        return False, f"parameters without a range or a tie: {missing}"
    return True, ""

### --- Experimental Design --- ###

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

def _cdf(p: np.ndarray) -> np.ndarray:
    """Normalised cumulative distribution."""
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    return cdf

def _draw(rng: np.random.Generator, cdf: np.ndarray) -> int:
    """Weighted index draw, same random stream as rng.choice(n, p=...)."""
    return int(cdf.searchsorted(rng.random(), side="right"))

def simulate_replicate(Nd, Nq, Nm, s, e, m, T, ic, zw, zq, sq, sw, dq, dw,
                       rng: np.random.Generator) -> float:
    """
    One replicate: returns the final mean p(A) across all demes and queen slots.
    demes[:, :, 0] maternal allele, [:, :, 1] paternal allele (0/1),
    demes[:, :, 2] sperm-pool p(A) of the queen.
    """
    demes = np.zeros((Nd, Nq, 3))
    worker_pheno = np.full(Nd, zw, dtype=float)
    if ic > 0:
        demes[:ic, 0, 0] = 1.0  # Seed the A allele in queen 1 of the first `ic` demes.

    p_A = float(demes[:, :, :2].mean())

    for _ in range(T):
        past = demes.copy()

        # ---- 1. Extinction -------------------------------------------------
        extinct = rng.random(Nd) < e
        alive = ~extinct

        # ---- 2. Phenotypes and fecundities (all demes at once) -------------
        geno = demes[:, :, :2]
        p_queen = geno.mean(axis=2)            # queen p(A)
        p_sperm = demes[:, :, 2]               # sperm-pool p(A)

        w_AA = p_queen * p_sperm
        w_Aa = (1 - p_queen) * p_sperm + (1 - p_sperm) * p_queen
        q_pheno = zq + sq * (2 * dq * p_queen + (1 - 2 * dq) * geno[:, :, 0] * geno[:, :, 1])

        # Queen weights use the worker phenotype carried over from last generation.
        f_first = fecundity(q_pheno, worker_pheno[:, None])
        f_sum = f_first.sum(axis=1, keepdims=True)
        weights = np.where(f_sum > 0, f_first / np.where(f_sum > 0, f_sum, 1.0), 1.0 / Nq)

        p_AA = (w_AA * weights).sum(axis=1)
        p_Aa = (w_Aa * weights).sum(axis=1)
        worker_pheno = np.where(alive, zw + p_AA * sw + p_Aa * (dw * sw), worker_pheno)

        fec = fecundity(q_pheno, worker_pheno[:, None])
        fec[extinct] = 0.0
        males = p_queen.copy()
        males[extinct] = -1.0

        # Wipe extinct demes (kept as -1 so they are visibly empty until refilled).
        demes[extinct] = -1.0
        worker_pheno[extinct] = -1.0

        # ---- 3. Replacement of vacant queen slots --------------------------
        all_fs = fec.ravel()
        males_flat = males.ravel()
        deme_mean_f = fec.mean(axis=1)
        alive_idx = np.flatnonzero(alive)
        total_f = all_fs.sum()
        global_cdf = None if total_f == 0 else _cdf(all_fs / total_f)

        for i in range(Nd):
            if extinct[i]:
                spots = np.arange(Nq)
            else:
                spots = np.flatnonzero(rng.random(Nq) > s)
            if spots.size == 0:
                continue

            others = alive_idx[alive_idx != i]
            if others.size:
                av_fec = deme_mean_f[others]
                av_sum = float(av_fec.sum())
                immigrant_mean = av_sum / others.size
            else:
                av_fec, av_sum, immigrant_mean = None, 0.0, 0.0

            local_mean = float(deme_mean_f[i])
            denom = m * immigrant_mean + (1 - m) * local_mean
            p_imm = (m * immigrant_mean / denom) if denom > 0 else 0.5

            src_cdf = _cdf(av_fec / av_sum) if (others.size and av_sum > 0) else None
            local_block = fec[i]
            local_sum = local_block.sum()
            local_cdf = _cdf(local_block / local_sum) if local_sum > 0 else None
            local_av = float(np.clip(males[i].mean(), 0.0, 1.0))

            for qspot in spots:
                go_immigrant = (rng.random() < p_imm) and src_cdf is not None

                if go_immigrant:
                    hits = 0
                    for _mate in range(Nm):
                        idx = (int(rng.integers(males_flat.size)) if global_cdf is None
                               else _draw(rng, global_cdf))
                        if rng.random() < males_flat[idx]:
                            hits += 1
                    sperm = hits / Nm

                    src_deme = int(others[_draw(rng, src_cdf)])
                    src_block = fec[src_deme]
                    src_sum = src_block.sum()
                    src_q = (_draw(rng, _cdf(src_block / src_sum)) if src_sum > 0
                             else int(rng.integers(Nq)))

                    maternal = float(past[src_deme, src_q, rng.integers(2)])
                    paternal = 1.0 if rng.random() < past[src_deme, src_q, 2] else 0.0
                else:
                    sperm = float(rng.binomial(Nm, local_av) / Nm)
                    src_q = _draw(rng, local_cdf) if local_cdf is not None else int(rng.integers(Nq))
                    maternal = float(past[i, src_q, rng.integers(2)])
                    paternal = 1.0 if rng.random() < demes[i, src_q, 2] else 0.0

                demes[i, qspot, 0] = maternal
                demes[i, qspot, 1] = paternal
                demes[i, qspot, 2] = sperm

        p_A = float(demes[:, :, :2].mean())
        if p_A == 0.0:       # Allele lost: absorbing state (no mutation).
            break

    return p_A

def final_frequencies(params: dict, rng: np.random.Generator) -> np.ndarray:
    """Final p(A) of `params['reps']` independent replicates."""
    keys = ("Nd", "Nq", "Nm", "s", "e", "m", "T", "ic", "zw", "zq", "sq", "sw", "dq", "dw")
    kwargs = {k: params[k] for k in keys}
    return np.array([simulate_replicate(**kwargs, rng=rng) for _ in range(params["reps"])])

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

### --- Workers -- ###

def _run_sample(task: tuple) -> tuple:
    """Pool worker. Returns ('ok', sample_id, row) or ('error', sample_id, message)."""
    sample_id, params, base_seed = task
    try:
        ok, msg = validate_params(params)
        if not ok:
            return "error", sample_id, msg
        rng = np.random.default_rng(np.random.SeedSequence(entropy=base_seed, spawn_key=(sample_id,)))
        row = {"sample_id": sample_id, **{c: params[c] for c in FEATURE_COLUMNS}}
        row.update(summarise(final_frequencies(params, rng)))
        return "ok", sample_id, row
    except Exception as exc:  # keep the batch alive, report the bad sample instead
        return "error", sample_id, f"{type(exc).__name__}: {exc}"

def _to_python(value):
    return value.item() if isinstance(value, np.generic) else value

def generate_dataset(n_samples: int, out_path: str | None = None, seed: int = 0,
                     Nd: int = FIXED_DEFAULTS["Nd"], T: int = FIXED_DEFAULTS["T"],
                     ic: int = FIXED_DEFAULTS["ic"], reps: int = FIXED_DEFAULTS["reps"],
                     workers: int | None = None, resume: bool = True,
                     space: dict | None = None, tied: dict | None = None) -> tuple[bool, str]:
    """
    Build the ML dataset. Returns (ok, message); never raises on bad input.

    n_samples : number of parameter combinations (rows), e.g. 3000
    out_path  : CSV location. Default: ant_ml_dataset_<n_samples>.csv in the current
                folder, which is the notebook's folder in Jupyter / VS Code.
    reps      : replicate simulations per combination (more reps = less noisy targets)
    workers   : processes to use (default: all CPU cores; 1 = no multiprocessing)
    resume    : continue an interrupted run with identical settings
    """
    space = PARAM_SPACE if space is None else space
    tied = TIED if tied is None else tied
    if not (isinstance(n_samples, int) and n_samples >= 1):
        return False, "n_samples must be an integer >= 1"
    out = Path(out_path or f"ant_ml_dataset_{n_samples}.csv").expanduser().resolve()
    meta_path = out.with_suffix(out.suffix + ".meta.json")

    ok, msg = validate_space(space, tied)
    if not ok:
        return False, f"Invalid parameter space: {msg}"
    fixed = {"Nd": Nd, "T": T, "ic": ic, "reps": reps}
    ok, msg = validate_params({**{c: 0.5 for c in FEATURE_COLUMNS}, "Nq": 1, "Nm": 1, **fixed})
    if not ok:
        return False, f"Invalid fixed settings: {msg}"

    config = {"n_samples": n_samples, "seed": seed, **fixed,
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
        tasks.append((sid, {**params, **fixed}, seed))

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
                  "Save it as ant_dataset.py next to the notebook and import it to use several workers.")
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

### --- Command Line --- ###

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the ant-polymorphism ML dataset.")
    ap.add_argument("--n-samples", type=int, required=True, help="parameter combinations (rows)")
    ap.add_argument("--out", default=None, help="default: ant_ml_dataset_<n>.csv in the current folder")
    ap.add_argument("--reps", type=int, default=FIXED_DEFAULTS["reps"])
    ap.add_argument("--T", type=int, default=FIXED_DEFAULTS["T"])
    ap.add_argument("--Nd", type=int, default=FIXED_DEFAULTS["Nd"])
    ap.add_argument("--ic", type=int, default=FIXED_DEFAULTS["ic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true", help="overwrite an existing CSV")
    a = ap.parse_args()

    ok, msg = generate_dataset(a.n_samples, a.out, a.seed, a.Nd, a.T, a.ic, a.reps,
                               a.workers, resume=not a.no_resume)
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