"""
Ant polymorphism simulation, male selection on the allele fraction: GPU version (PyTorch)

Same model, parameter space and CSV columns as ant_dataset_male_freq.py. Put both files in
the same folder: this file imports the design, validation and target code from it, so there
is a single source of truth (read that file's docstring for the model).

GPU safety:

    pip install torch numpy pandas tqdm
    python ant_dataset_male_freq_gpu.py --selftest                  # 1-2 min check of your GPU
    python ant_dataset_male_freq_gpu.py --n-samples 2000 --reps 100 --out ant_male_freq_gpu.csv

Device: automatic (mps -> cuda -> cpu); force one with --device mps|cuda|cpu.

Why it is not a line-by-line port
---------------------------------
One replicate (100 demes x <=10 queens) is far too small to keep a GPU busy, so many
replicates (--rows-per-batch, default 500, i.e. several parameter sets) advance together,
one generation at a time, as dense tensors with a leading "replicate" axis. Variable-length
index lists become masked dense operations. Replicates that reach fixation or loss are dropped
from the batch, so early stopping still saves time.

Differences from the CPU file (statistically equivalent, not bit-identical):
  * float32 (the Apple GPU has no float64).
  * Binomial(Nm, p) is the sum of Nm Bernoulli draws (exact; torch.binomial is not on MPS).
  * Random numbers differ: a run is reproducible only with the same --rows-per-batch and seed.
  * Operations the Apple GPU lacks fall back to the CPU automatically (PYTORCH_ENABLE_MPS_FALLBACK).

Memory: a batch of 500 replicates uses a few hundred MB. If you get an out-of-memory error,
lower --rows-per-batch (the failing batch is skipped with a warning and the run continues).
"""

### --- Useful Libraries --- ###

from __future__ import annotations
import argparse
import csv
import json
import os
import time
import warnings
from datetime import timedelta
from pathlib import Path

# Let the few operations the Apple GPU does not implement fall back to the CPU instead of
# crashing (must be set before torch is imported).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import torch
except ImportError:  # Reported cleanly by generate_dataset_gpu.
    torch = None

import ant_dataset_male_freq as cpu  # Shared design, validation, targets, defaults.

FLOAT_KEYS = ("s", "e", "mq", "mm", "zw", "zq", "zm", "sq", "sw", "sm", "dq", "dw", "om")

### --- Device --- ###

def pick_device(name: str = "auto"):
    """Returns (torch.device | None, message)."""
    if torch is None:
        return None, "PyTorch is not installed (pip install torch)"
    try:
        if name == "auto":
            if torch.backends.mps.is_available():
                name = "mps"
            elif torch.cuda.is_available():
                name = "cuda"
            else:
                name = "cpu"
        return torch.device(name), ""
    except Exception as exc:
        return None, f"Cannot use device '{name}': {exc}"

### --- Batched Simulation --- ###

def _select(f, sA, sa):
    """Fraction of A males after selection (see the CPU file); unchanged if no male survives.
    sA, sa broadcast against f."""
    den = f * sA + (1.0 - f) * sa
    return torch.where(den > 0, f * sA / torch.where(den > 0, den, torch.ones_like(den)), f)

def simulate_batch(rows: dict, Nd: int, T: int, ic: int, male_exponent: float, device,
                   early_stop: bool = True, seed: int = 0) -> np.ndarray:
    """
    Simulate many replicates at once. `rows` has one entry per replicate (parameters repeated
    for the replicates of one parameter set): every key in FLOAT_KEYS plus the integers 'Nq'
    and 'Nm'. Returns the final mean p(A) of each replicate (same meaning as
    ant_dataset_male_freq.simulate_replicate).
    """
    dt = torch.float32
    torch.manual_seed(seed)
    B0 = len(rows["Nq"])
    P = {k: torch.as_tensor(np.asarray(rows[k], dtype=np.float32), device=device) for k in FLOAT_KEYS}
    Nq_b = torch.as_tensor(np.asarray(rows["Nq"], dtype=np.int64), device=device)
    Nm_b = torch.as_tensor(np.asarray(rows["Nm"], dtype=np.int64), device=device)
    NQ, NM = int(Nq_b.max()), int(Nm_b.max())
    arange_d = torch.arange(Nd, device=device)
    arange_m = torch.arange(NM, device=device)

    def male_surv(z):                                          # 1 - min(|om - z|, 1) ** exponent.
        return (1.0 - (P["om"] - z).abs().clamp_max(1.0) ** male_exponent).clamp_min(0.0)

    slot_ok = torch.arange(NQ, device=device)[None, None, :] < Nq_b[:, None, None]   # (B,1,NQ) real slots.
    slotf = slot_ok.to(dt)
    alleles = torch.zeros(B0, Nd, NQ, 2, device=device, dtype=dt)
    if ic > 0:
        alleles[:, :ic, 0, 0] = 1.0                            # Seed A in queen 1 of the first ic demes.
    sperm = torch.zeros(B0, Nd, NQ, device=device, dtype=dt)
    wp = P["zw"][:, None].expand(B0, Nd).clone()               # Worker phenotype per deme.
    surv_a, surv_A = male_surv(P["zm"]), male_surv(P["zm"] + P["sm"])
    row_id = torch.arange(B0, device=device)
    done = torch.zeros(B0, dtype=torch.bool, device=device)
    final = np.full(B0, np.nan)
    pA = None

    for gen in range(T):
        # ---- Drop finished replicates from the batch (every 20 generations). --------
        if gen > 0 and gen % 20 == 0 and float(done.float().mean()) > 0.25:
            keep = ~done
            P = {k: v[keep] for k, v in P.items()}
            Nq_b, Nm_b, slot_ok, slotf = Nq_b[keep], Nm_b[keep], slot_ok[keep], slotf[keep]
            alleles, sperm, wp = alleles[keep], sperm[keep], wp[keep]
            surv_a, surv_A, row_id = surv_a[keep], surv_A[keep], row_id[keep]
            done = torch.zeros(int(keep.sum()), dtype=torch.bool, device=device)
        Bc = alleles.shape[0]
        c3 = lambda x: x[:, None, None]

        # ---- 1. Extinction -------------------------------------------------------
        extinct = torch.rand(Bc, Nd, device=device) < P["e"][:, None]
        alive = ~extinct
        skip = ~alive.any(dim=1)                               # Every deme extinct: roll back.

        # ---- 2. Phenotypes and fecundities -----------------------------------------
        a0, a1 = alleles[..., 0], alleles[..., 1]
        p_queen = 0.5 * (a0 + a1)
        w_AA = p_queen * sperm
        w_Aa = (1 - p_queen) * sperm + (1 - sperm) * p_queen
        dq3 = c3(P["dq"])
        q_pheno = c3(P["zq"]) + c3(P["sq"]) * (2 * dq3 * p_queen + (1 - 2 * dq3) * a0 * a1)

        f_first = ((1 - q_pheno) * wp[:, :, None]).clamp_min(0.0) * slotf
        f_sum = f_first.sum(2, keepdim=True)
        weights = torch.where(f_sum > 0, f_first / torch.where(f_sum > 0, f_sum, torch.ones_like(f_sum)),
                              slotf / c3(Nq_b.to(dt)))
        p_AA = (w_AA * weights).sum(2)
        p_Aa = (w_Aa * weights).sum(2)
        wp_old = wp
        wp = torch.where(alive, P["zw"][:, None] + p_AA * P["sw"][:, None]
                         + p_Aa * (P["dw"][:, None] * P["sw"][:, None]), wp)
        fec = ((1 - q_pheno) * wp[:, :, None]).clamp_min(0.0) * slotf * alive[:, :, None]
        wp = torch.where(extinct, torch.full_like(wp, -1.0), wp)

        # ---- 3. Which slots are refilled ---------------------------------------------
        replace = slot_ok & (extinct[:, :, None] | (torch.rand(Bc, Nd, NQ, device=device) > c3(P["s"])))

        # ---- 4. Mother of each new queen: local, or migrant from another deme -----------
        dmf = fec.sum(2) / Nq_b.to(dt)[:, None]                # Deme mean fecundity (0 if extinct).
        cnt_oth = alive.sum(1, keepdim=True).to(dt) - alive.to(dt)
        imm_mean = torch.where(cnt_oth > 0, (dmf.sum(1, keepdim=True) - dmf) / cnt_oth.clamp_min(1.0),
                               torch.zeros_like(dmf))
        mq2 = P["mq"][:, None]
        denom = mq2 * imm_mean + (1 - mq2) * dmf
        p_imm = torch.where(denom > 0, mq2 * imm_mean / torch.where(denom > 0, denom, torch.ones_like(denom)),
                            torch.full_like(denom, 0.5))
        cum = dmf.cumsum(1)
        W = cum[:, -1:]
        can_imm = (W - dmf) > 1e-6                             # Another deme can supply a mother.
        go_imm = can_imm[:, :, None] & (torch.rand(Bc, Nd, NQ, device=device) < p_imm[:, :, None])

        u = torch.rand(Bc, Nd, NQ, device=device) * (W - dmf)[:, :, None]
        u = torch.where(u >= (cum - dmf)[:, :, None], u + dmf[:, :, None], u)    # skip the own deme
        src_im = torch.searchsorted(cum.contiguous(), u.reshape(Bc, -1).contiguous(),
                                    right=True).reshape(Bc, Nd, NQ).clamp_max(Nd - 1)
        src = torch.where(go_imm, src_im, arange_d[None, :, None].expand(Bc, Nd, NQ))

        # Mother queen inside the source deme: fecundity-weighted (uniform if the deme has none).
        src_w = torch.where((fec.sum(2) > 0)[:, :, None], fec, slotf.expand(Bc, Nd, NQ))
        cdf_q = (src_w / src_w.sum(2, keepdim=True)).cumsum(2)
        flat_src = (torch.arange(Bc, device=device)[:, None, None] * Nd + src).reshape(-1)
        cdf_src = cdf_q.reshape(Bc * Nd, NQ)[flat_src].reshape(Bc, Nd, NQ, NQ)
        q_src = (torch.rand(Bc, Nd, NQ, 1, device=device) >= cdf_src).sum(3)
        q_src = torch.minimum(q_src, (Nq_b - 1)[:, None, None])
        mom = flat_src * NQ + q_src.reshape(-1)                # Flat index of the mother's slot.
        pair = alleles.reshape(-1, 2)[mom].reshape(Bc, Nd, NQ, 2)
        mat = torch.where(torch.rand(Bc, Nd, NQ, device=device) < 0.5, pair[..., 1], pair[..., 0])
        pat = (torch.rand(Bc, Nd, NQ, device=device) < sperm.reshape(-1)[mom].reshape(Bc, Nd, NQ)).to(dt)

        # ---- 5. Males: pool fractions, selection, Nm matings ------------------------------
        tot = fec.sum((1, 2))
        alivef = alive.to(dt)[:, :, None] * slotf
        unw = (p_queen * alivef).sum((1, 2)) / alivef.sum((1, 2)).clamp_min(1.0)
        f_glob = torch.where(tot > 0, (fec * p_queen).sum((1, 2)) / tot.clamp_min(1e-30), unw)
        f_loc = torch.where(alive, (p_queen * slotf).sum(2) / Nq_b.to(dt)[:, None], f_glob[:, None])
        mm2 = P["mm"][:, None]
        p_mix = (mm2 * _select(f_glob[:, None], surv_A[:, None], surv_a[:, None])
                 + (1 - mm2) * _select(f_loc, surv_A[:, None], surv_a[:, None])).clamp(0.0, 1.0)
        coin = torch.rand(Bc, Nd, NQ, NM, device=device) < p_mix[:, :, None, None]
        coin &= arange_m[None, None, None, :] < Nm_b[:, None, None, None]
        new_sperm = coin.sum(3).to(dt) / Nm_b.to(dt)[:, None, None]

        # ---- 6. Write the new queens (rolled-back replicates keep their old state) -------
        upd_alleles = torch.where(replace.unsqueeze(-1), torch.stack((mat, pat), dim=-1), alleles)
        upd_sperm = torch.where(replace, new_sperm, sperm)
        alleles = torch.where(skip[:, None, None, None], alleles, upd_alleles)
        sperm = torch.where(skip[:, None, None], sperm, upd_sperm)
        wp = torch.where(skip[:, None], wp_old, wp)

        # ---- 7. Statistics and per-replicate stopping -------------------------------------
        pA = alleles.sum((1, 2, 3)) / (2.0 * Nd * Nq_b.to(dt))
        if early_stop:
            lost = (pA == 0) & (sperm.sum((1, 2)) == 0)
            fixed = (pA == 1) & ((sperm + (1 - slotf)).amin(dim=(1, 2)) == 1)
            newly = (lost | fixed) & ~done
            if bool(newly.any()):
                final[row_id[newly].cpu().numpy()] = pA[newly].cpu().numpy()
                done = done | newly
                if bool(done.all()):
                    return final

    rem = ~done                                                # Reached T: record the last state.
    if pA is not None and bool(rem.any()):
        final[row_id[rem].cpu().numpy()] = pA[rem].cpu().numpy()
    return final

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset_gpu(n_samples: int, out_path: str | None = None, seed: int = 0,
                         Nd: int = cpu.FIXED_DEFAULTS["Nd"], T: int = cpu.FIXED_DEFAULTS["T"],
                         ic: int = cpu.FIXED_DEFAULTS["ic"], reps: int = cpu.FIXED_DEFAULTS["reps"],
                         resume: bool = True, space: dict | None = None, tied: dict | None = None,
                         early_stop: bool = True,
                         male_exponent: float = cpu.FIXED_DEFAULTS["male_exponent"],
                         device: str = "auto", rows_per_batch: int = 500) -> tuple[bool, str]:
    """
    Build the ML dataset on the GPU. Returns (ok, message); never raises on bad input.

    rows_per_batch : replicates simulated together (parameter sets per batch = rows // reps,
                     at least 1). Larger = faster until the GPU is saturated or memory runs out.
    n_samples     : number of parameter combinations (rows), e.g. 3000
    out_path      : CSV location. Default: ant_male_freq_dataset_<n_samples>.csv in the current folder
    reps          : replicate simulations per combination (more reps = less noisy targets)
    workers       : processes to use (default: all CPU cores; 1 = no multiprocessing)
    resume        : continue an interrupted run with identical settings
    early_stop    : stop a replicate once the allele is fixed or lost (absorbing states)
    male_exponent : male survival = 1 - distance ** male_exponent (see module docstring)
    """
    dev, msg = pick_device(device)
    if dev is None:
        return False, msg
    space = cpu.PARAM_SPACE if space is None else space
    tied = cpu.TIED if tied is None else tied
    if not (isinstance(n_samples, int) and n_samples >= 1):
        return False, "n_samples must be an integer >= 1"
    if not (isinstance(rows_per_batch, int) and rows_per_batch >= 1):
        return False, "rows_per_batch must be an integer >= 1"
    out = Path(out_path or f"ant_male_freq_gpu_dataset_{n_samples}.csv").expanduser().resolve()
    meta_path = out.with_suffix(out.suffix + ".meta.json")

    ok, msg = cpu.validate_space(space, tied)
    if not ok:
        return False, f"Invalid parameter space: {msg}"
    fixed = {"Nd": Nd, "T": T, "ic": ic, "reps": reps, "male_exponent": male_exponent}
    ok, msg = cpu.validate_params({**{c: 0.5 for c in cpu.FEATURE_COLUMNS}, "Nq": 1, "Nm": 1, **fixed})
    if not ok:
        return False, f"Invalid fixed settings: {msg}"

    config = {"n_samples": n_samples, "seed": seed, **fixed, "early_stop": early_stop,
              "space": {k: list(v) for k, v in space.items()}, "tied": tied,
              "backend": "torch", "rows_per_batch": rows_per_batch}

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

    design = cpu.build_design(n_samples, seed, space, tied)
    tasks, skipped = [], []
    for rec in design.to_dict("records"):
        sid = int(rec["sample_id"])
        if sid in done_ids:
            continue
        params = {k: cpu._to_python(v) for k, v in rec.items() if k != "sample_id"}
        ok, msg = cpu.validate_params({**params, **fixed})
        if ok:
            tasks.append((sid, params))
        else:
            skipped.append((sid, msg))
            warnings.warn(f"sample {sid} skipped: {msg}")
    if not tasks and not skipped:
        return True, f"Nothing to do: all {n_samples} rows already in {out}."

    per_batch = max(1, rows_per_batch // reps)
    print(f"Output file : {out}")
    print(f"Device      : {dev}")
    print(f"Combinations: {n_samples} total, {len(done_ids)} already done, {len(tasks)} to run "
          f"({reps} replicates each, {per_batch} parameter set(s) per batch = {per_batch * reps} replicates)")
    t_start = time.perf_counter()
    write_header = not out.exists() or out.stat().st_size == 0

    try:
        with open(out, "a", newline="") as fh, \
                tqdm(total=n_samples, initial=len(done_ids), desc="Generating", unit="set",
                     dynamic_ncols=True) as bar:
            writer = csv.DictWriter(fh, fieldnames=cpu.CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            bar.update(len(skipped))
            for start in range(0, len(tasks), per_batch):
                chunk = tasks[start:start + per_batch]
                rows = {k: np.repeat([p[k] for _, p in chunk], reps) for k in FLOAT_KEYS + ("Nq", "Nm")}
                batch_seed = (seed * 1_000_003 + chunk[0][0]) % (2 ** 31)
                try:
                    finals = simulate_batch(rows, Nd, T, ic, male_exponent, dev, early_stop, batch_seed)
                except Exception as exc:  # E.g. out of memory: skip the batch, keep the run alive.
                    for sid, _ in chunk:
                        skipped.append((sid, f"{type(exc).__name__}: {exc}"))
                    warnings.warn(f"batch starting at sample {chunk[0][0]} skipped: {type(exc).__name__}: {exc}"
                                  " (try a smaller --rows-per-batch)")
                    bar.update(len(chunk))
                    continue
                for i, (sid, params) in enumerate(chunk):
                    row = {"sample_id": sid, **{c: params[c] for c in cpu.FEATURE_COLUMNS}}
                    row.update(cpu.summarise(finals[i * reps:(i + 1) * reps]))
                    writer.writerow(row)
                fh.flush()
                bar.update(len(chunk))
    except KeyboardInterrupt:
        return False, f"Interrupted. Finished rows are saved in {out}; run again to resume."

    n_rows = n_samples - len(skipped)
    elapsed = timedelta(seconds=int(time.perf_counter() - t_start))
    msg = f"Saved {n_rows} rows to {out} (this run took {elapsed})."
    if skipped:
        msg += f" {len(skipped)} sample(s) skipped (ids: {[s for s, _ in skipped][:10]}...)."
    return True, msg

### --- Self-test: GPU model vs the CPU model --- ###

def selftest(device: str = "auto", reps: int = 300, Nd: int = 40, T: int = 150) -> bool:
    """Run the same parameter sets through the CPU and GPU models and compare the targets."""
    dev, msg = pick_device(device)
    if dev is None:
        print(msg)
        return False
    base = dict(Nq=3, Nm=6, s=0.5, e=0.05, mq=0.5, mm=0.5, zw=0.5, zq=0.5, zm=0.5, dw=0.5, dq=0.5,
                sw=0.1, sq=0.1, sm=0.1, om=0.5)
    sets = {"neutral-ish (om=0.55)": {**base, "om": 0.55},
            "male-favoured (om=1)": {**base, "om": 1.0},
            "male-disfavoured (om=0)": {**base, "om": 0.0, "sw": 0.0, "sq": 0.0},
            "all local, s=0.8": {**base, "mq": 0.0, "mm": 0.0, "s": 0.8}}
    print(f"Device: {dev} | {reps} replicates per set, Nd={Nd}, T={T}")
    print(f"{'set':26s} {'':4s}{'mean_pA':>16s} {'lost':>8s} {'fixed':>8s}")
    ok_all = True
    for name, p in sets.items():
        full = {**p, "Nd": Nd, "T": T, "ic": 10, "reps": reps, "male_exponent": 1.0}
        c = cpu.final_frequencies(full, np.random.default_rng(1))
        rows = {k: np.full(reps, full[k]) for k in FLOAT_KEYS + ("Nq", "Nm")}
        t = time.perf_counter()
        g = simulate_batch(rows, Nd, T, 10, 1.0, dev, True, 2)
        dt_g = time.perf_counter() - t
        se = np.hypot(c.std(ddof=1), g.std(ddof=1)) / np.sqrt(reps)
        agree = abs(c.mean() - g.mean()) < 3.5 * se + 1e-3
        ok_all &= bool(agree)
        for lab, x in (("CPU", c), ("GPU", g)):
            print(f"{name if lab == 'CPU' else '':26s} {lab:4s}{x.mean():10.4f}±{x.std(ddof=1) / np.sqrt(reps):.4f} "
                  f"{np.mean(x == 0):8.3f} {np.mean(x == 1):8.3f}" + (f"   [{dt_g:.1f}s]  {'OK' if agree else 'DIFFERENT'}" if lab == 'GPU' else ""))
    print("SELFTEST PASSED" if ok_all else "SELFTEST: differences found (rerun with more --reps before worrying)")
    return ok_all

### --- Command Lines --- ###

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the ant-polymorphism ML dataset on the GPU (male selection on the allele fraction).")
    ap.add_argument("--n-samples", type=int, help="parameter combinations (rows)")
    ap.add_argument("--out", default=None, help="default: ant_male_freq_gpu_dataset_<n>.csv in the current folder")
    ap.add_argument("--reps", type=int, default=cpu.FIXED_DEFAULTS["reps"])
    ap.add_argument("--T", type=int, default=cpu.FIXED_DEFAULTS["T"])
    ap.add_argument("--Nd", type=int, default=cpu.FIXED_DEFAULTS["Nd"])
    ap.add_argument("--ic", type=int, default=cpu.FIXED_DEFAULTS["ic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--rows-per-batch", type=int, default=500,
                    help="replicates simulated together (lower it if memory runs out)")
    ap.add_argument("--no-resume", action="store_true", help="overwrite an existing CSV")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="always run T generations (do not stop at fixation or loss)")
    ap.add_argument("--male-exponent", type=float, default=cpu.FIXED_DEFAULTS["male_exponent"],
                    help="male survival = 1 - distance ** exponent (default 1 = linear)")
    ap.add_argument("--selftest", action="store_true", help="compare this GPU model with the CPU model and exit")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if selftest(a.device) else 1)
    if a.n_samples is None:
        ap.error("--n-samples is required (or use --selftest)")
    ok, msg = generate_dataset_gpu(a.n_samples, a.out, a.seed, a.Nd, a.T, a.ic, a.reps,
                                   resume=not a.no_resume, early_stop=not a.no_early_stop,
                                   male_exponent=a.male_exponent, device=a.device,
                                   rows_per_batch=a.rows_per_batch)
    print(msg)
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
