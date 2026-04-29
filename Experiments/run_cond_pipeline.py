#!/usr/bin/env python
"""End-to-end driver for *conditional* StrideDiff experiments.

Per (dataset, seed) it does, in order:
  1) train        -> Checkpoints_cond_{dataset}_seed{seed}_L{seq}_{seq}/checkpoint-{M}.pt
                     (with dataloader.train_dataset.params.proportion=0.9 to match
                      the 90/10 split used by the conditional test dataset)
  2) sample (x tasks x modes) -> OUTPUT/{dataset}_seed{seed}_L{seq}/cond/{task_tag}/
                                    fake_{mode}.npy, real.npy, mask.npy,
                                    time_{mode}.json
  3) metric (x tasks x modes) -> OUTPUT/{dataset}_seed{seed}_L{seq}/cond/{task_tag}/
                                    metric_{mode}.json

Tasks are specified as comma-separated "mode:param" pairs, where mode is
``infill`` (imputation) or ``predict`` (forecasting):
    --tasks infill:0.5
    --tasks infill:0.3,infill:0.5,infill:0.7
    --tasks predict:24
The task_tag used on disk is ``infill_mr0.5`` / ``predict_pl24``.

Each (dataset, seed) is one job; jobs run on user-selected GPUs in parallel.

Datasets: Stocks / ETTh / Energy / fMRI (matches the paper's conditional table).

Inference modes (pass via --modes, comma-separated):
  ddpm    : full-T DDPM (sample_infill)
  fast200 : 200-step DDIM (fast_sample_infill)
  banded  : stridediff frequency-aware dynamic-jump sampler
            (sample_infill_banded). Per-dataset schedule is taken from
            BAND_HPARAMS (big_k / med_k / small_k / last_k_always_micro /
            tau_energy / tau_dlogP / tau_pv).

Examples:

    # Full run: 4 datasets x 5 seeds x {ddpm, fast200, banded} x 5 imputation ratios
    python Experiments/run_cond_pipeline.py --gpus 0,1,2,3,4,5,6,7

    # Only banded (the stridediff frequency-aware jump schedule) on ETTh:
    python Experiments/run_cond_pipeline.py --gpus 0,1 \
        --datasets etth --modes banded

    # Reuse unconditional checkpoints already produced elsewhere:
    python Experiments/run_cond_pipeline.py --gpus 0,1,2,3 --reuse_uncond_ckpt \
        --seq_length 24

    # Quick smoke test of the whole train+sample+metric loop:
    python Experiments/run_cond_pipeline.py --gpus 0 --debug
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Empty

# -----------------------------------------------------------------------------
# Constants

DATASETS = ["stocks", "etth", "energy", "fmri"]
# StrideDiff supports three conditional sampling paths:
#   ddpm    - full-T DDPM  (sample_infill)
#   fast200 - DDIM 200     (fast_sample_infill)
#   banded  - stridediff frequency-aware dynamic-jump sampler
#             (sample_infill_banded), the conditional counterpart of
#             the unconditional `sample()` used by the stridediff paper
#             table. Per-dataset hparams come from BAND_HPARAMS below.
MODES_DEFAULT = ["ddpm", "fast200", "banded"]
ALL_MODES = ["ddpm", "fast200", "banded"]
SEEDS_DEFAULT = [12345, 2026, 2027, 2028, 2029]

# Per-dataset frequency-aware jump schedule, matching the table used for
# the stridediff (unconditional) results. big_k / med_k / small_k set the
# jump sizes when no / only-low / high bands are active; last_k_always_micro
# forces small_k for the final N steps; tau_energy / tau_dlogP / tau_pv
# are the thresholds of the band-activity gate (energy fraction, |Δ log P|,
# phase-velocity). Keys align with the CONFIGS map above; we also include
# sines/mujoco as fallbacks in case we extend the cond pipeline later.
BAND_HPARAMS: dict[str, dict[str, float]] = {
    "sines":  {"big_k": 50, "med_k": 20, "small_k": 1, "last_k_always_micro": 20,
               "tau_energy": 0.1, "tau_dlogP": 0.01, "tau_pv": 0.04},
    "stocks": {"big_k": 50, "med_k": 10, "small_k": 1, "last_k_always_micro": 20,
               "tau_energy": 0.1, "tau_dlogP": 0.01, "tau_pv": 0.04},
    "etth":   {"big_k": 50, "med_k": 10, "small_k": 1, "last_k_always_micro": 12,
               "tau_energy": 0.7, "tau_dlogP": 0.02, "tau_pv": 0.08},
    "mujoco": {"big_k": 20, "med_k":  5, "small_k": 1, "last_k_always_micro": 20,
               "tau_energy": 0.5, "tau_dlogP": 0.02, "tau_pv": 0.04},
    "energy": {"big_k": 40, "med_k":  5, "small_k": 1, "last_k_always_micro": 12,
               "tau_energy": 0.1, "tau_dlogP": 0.01, "tau_pv": 0.06},
    "fmri":   {"big_k": 20, "med_k":  5, "small_k": 1, "last_k_always_micro": 20,
               "tau_energy": 0.5, "tau_dlogP": 0.01, "tau_pv": 0.06},
}

CONFIGS = {
    "stocks": "Config/stocks.yaml",
    "etth":   "Config/etth.yaml",
    "energy": "Config/energy.yaml",
    "fmri":   "Config/fmri.yaml",
}

# Base for conditional-training checkpoints (separate from unconditional ones,
# because we retrain with proportion=0.9 to match the test split).
CKPT_BASE_COND = {
    "stocks": "./Checkpoints_cond_stock",
    "etth":   "./Checkpoints_cond_etth",
    "energy": "./Checkpoints_cond_energy",
    "fmri":   "./Checkpoints_cond_fmri",
}

# If --reuse_uncond_ckpt is given we point to the unconditional checkpoints
# (mirroring the per-run_pipeline naming convention used by stridediff).
CKPT_BASE_UNCOND = {
    "stocks": "./Checkpoints_stock",
    "etth":   "./Checkpoints_etth",
    "energy": "./Checkpoints_energy",
    "fmri":   "./Checkpoints_fmri",
}

ROOT = Path(__file__).resolve().parent.parent  # stridediff/
LOG_ROOT = ROOT / "Experiments" / "pipeline_logs"

DEBUG_TAG = "_debug"

# Opt that switches training into the 90/10 split used by the conditional
# test dataset. Required unless you reuse an unconditional checkpoint trained
# on the full 100% of the data.
COND_TRAIN_OPTS = [
    "dataloader.train_dataset.params.proportion", "0.9",
]


# Paper Section 4.5 (and Figure 5/6) uses 48-step time series for both the
# imputation and forecasting experiments on Stocks/ETTh/Energy/fMRI. The
# reference yamls in Config/ default to seq_length=24 (which matches the
# unconditional 24-length generation table); for the conditional pipeline we
# expose --seq_length so the default is paper-faithful (48). Override sets the
# three places the dataloader/model read the window from.
def seq_length_opts(seq_length: int) -> list[str]:
    s = str(int(seq_length))
    return [
        "model.params.seq_length",                         s,
        "dataloader.train_dataset.params.window",          s,
        "dataloader.test_dataset.params.window",           s,
    ]


# -----------------------------------------------------------------------------
# Task parsing

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "t"}:
        return True
    if s in {"0", "false", "no", "n", "off", "f"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Boolean value expected (true/false), got {v!r}")


def parse_tasks(s: str) -> list[tuple[str, float]]:
    """Parse "infill:0.5,predict:24" -> [("infill", 0.5), ("predict", 24.0)]."""
    out = []
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise SystemExit(f"Bad --tasks entry {raw!r}; expected 'infill:<mr>' or 'predict:<pl>'.")
        mode, val = raw.split(":", 1)
        mode = mode.strip()
        if mode not in ("infill", "predict"):
            raise SystemExit(f"Bad task mode {mode!r}; choices: infill, predict.")
        try:
            v = float(val)
        except ValueError:
            raise SystemExit(f"Bad task value {val!r} in {raw!r}; must be numeric.")
        out.append((mode, v))
    if not out:
        raise SystemExit("--tasks parsed to empty list.")
    return out


def task_tag(mode: str, val: float) -> str:
    if mode == "infill":
        return f"infill_mr{val:g}"
    return f"predict_pl{int(val)}"


# -----------------------------------------------------------------------------
# Path helpers

def run_name(dataset: str, seed: int, seq_length: int, debug: bool = False) -> str:
    # Include _L{seq_length} so OUTPUT/, logs, and the name passed to main.py
    # cleanly separate runs at different window lengths (e.g. 24 vs 48).
    return f"{dataset}_seed{seed}_L{int(seq_length)}" + (DEBUG_TAG if debug else "")


def output_dir(dataset: str, seed: int, seq_length: int, debug: bool = False) -> Path:
    return ROOT / "OUTPUT" / run_name(dataset, seed, seq_length, debug)


def cond_task_dir(dataset: str, seed: int, tag: str, seq_length: int,
                  debug: bool = False) -> Path:
    return output_dir(dataset, seed, seq_length, debug) / "cond" / tag


def ckpt_results_folder(dataset: str, seed: int,
                        reuse_uncond: bool, seq_length: int,
                        debug: bool = False) -> str:
    if reuse_uncond:
        base = CKPT_BASE_UNCOND[dataset]
        return f"{base}_seed{seed}" + (DEBUG_TAG if debug else "")
    base = CKPT_BASE_COND[dataset]
    # Trainer appends "_{seq_length}", so final dir is <this>_<seq_length>.
    return f"{base}_seed{seed}_L{int(seq_length)}" + (DEBUG_TAG if debug else "")


def ckpt_dir(dataset: str, seed: int, reuse_uncond: bool, seq_length: int,
             debug: bool = False) -> Path:
    return ROOT / f"{ckpt_results_folder(dataset, seed, reuse_uncond, seq_length, debug).lstrip('./')}_{int(seq_length)}"


def ckpt_file(dataset: str, seed: int, milestone: int,
              reuse_uncond: bool, seq_length: int, debug: bool = False) -> Path:
    return ckpt_dir(dataset, seed, reuse_uncond, seq_length, debug) / f"checkpoint-{milestone}.pt"


def log_file(dataset: str, seed: int, kind: str, seq_length: int,
             extra: str = "", debug: bool = False) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{extra}" if extra else ""
    return LOG_ROOT / f"cond__{run_name(dataset, seed, seq_length, debug)}__{kind}{suffix}.log"


# -----------------------------------------------------------------------------
# Command builders

# Tiny overrides used for --debug. Lets you smoke-test the whole cond loop
# (train+sample+metric) in a couple of minutes per job.
DEBUG_OPTS = [
    "model.params.timesteps", "20",
    "model.params.sampling_timesteps", "20",
    "solver.save_cycle", "50",
    "solver.max_epochs", "100",
]
DEBUG_FAST_STEPS = 10  # for --inference_mode fast200 under --debug (must be <= timesteps)


def build_train_cmd(dataset: str, seed: int, reuse_uncond: bool,
                    seq_length: int,
                    extra_opts: list[str] | None = None,
                    debug: bool = False) -> list[str]:
    cmd = [
        "python", "-u", "main.py",
        "--name", run_name(dataset, seed, seq_length, debug),
        "--config_file", CONFIGS[dataset],
        "--gpu", "0",
        "--seed", str(seed),
        "--train",
        "solver.results_folder",
        ckpt_results_folder(dataset, seed, reuse_uncond, seq_length, debug),
    ]
    cmd += seq_length_opts(seq_length)
    if not reuse_uncond:
        cmd += list(COND_TRAIN_OPTS)
    if extra_opts:
        cmd += extra_opts
    return cmd


def build_sample_cmd(dataset: str, seed: int, milestone: int, mode: str,
                     task_mode: str, task_val: float, tag: str,
                     reuse_uncond: bool, seq_length: int,
                     extra_opts: list[str] | None = None,
                     fast_steps: int | None = None,
                     band_langevin: bool = True,
                     band_projection: bool = False,
                     debug: bool = False) -> list[str]:
    cmd = [
        "python", "-u", "main.py",
        "--name", run_name(dataset, seed, seq_length, debug),
        "--config_file", CONFIGS[dataset],
        "--gpu", "0",
        "--seed", str(seed),
        "--sample", "1",
        "--mode", task_mode,
        "--milestone", str(milestone),
        "--inference_mode", mode,
        "--cond_task_tag", tag,
    ]
    if task_mode == "infill":
        cmd += ["--missing_ratio", str(task_val)]
    else:
        cmd += ["--pred_len", str(int(task_val))]
    if fast_steps is not None and mode == "fast200":
        cmd += ["--fast_steps", str(fast_steps)]
    if mode == "banded":
        hp = BAND_HPARAMS.get(dataset)
        if hp is None:
            raise SystemExit(
                f'[banded] no BAND_HPARAMS entry for dataset={dataset!r}; '
                f'known keys: {list(BAND_HPARAMS.keys())}')
        # In --debug we shrink big_k / med_k / last_k_always_micro so the
        # schedule still makes sense under the tiny T=20 debug horizon.
        if debug:
            T_debug = 20
            big_k = min(int(hp["big_k"]), max(T_debug - 1, 1))
            med_k = min(int(hp["med_k"]), max(T_debug - 1, 1))
            last_micro = min(int(hp["last_k_always_micro"]), max(T_debug - 1, 1))
        else:
            big_k = int(hp["big_k"])
            med_k = int(hp["med_k"])
            last_micro = int(hp["last_k_always_micro"])
        cmd += [
            "--big_k", str(big_k),
            "--med_k", str(med_k),
            "--small_k", str(int(hp["small_k"])),
            "--last_k_always_micro", str(last_micro),
            "--tau_energy", f"{float(hp['tau_energy']):g}",
            "--tau_dlogP",  f"{float(hp['tau_dlogP']):g}",
            "--tau_pv",     f"{float(hp['tau_pv']):g}",
        ]
        # Must come BEFORE solver.results_folder (which triggers
        # main.py's REMAINDER positional capture).
        cmd += ["--use_band_langevin"] if band_langevin else ["--no_band_langevin"]
        cmd += ["--use_band_projection"] if band_projection else ["--no_band_projection"]
    cmd += [
        "solver.results_folder",
        ckpt_results_folder(dataset, seed, reuse_uncond, seq_length, debug),
    ]
    cmd += seq_length_opts(seq_length)
    if extra_opts:
        cmd += extra_opts
    return cmd


def build_metric_cmd(cond_dir: Path, mode: str) -> list[str]:
    return [
        "python", "-u", "Experiments/compute_cond_metrics.py",
        "--cond_dir", str(cond_dir),
        "--mode", mode,
    ]


# -----------------------------------------------------------------------------
# Subprocess runner with logging

def run_subprocess(cmd: list[str], log_path: Path, gpu_id: int, label: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    header = (
        f"=== {label} ===\n"
        f"time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"gpu (CUDA_VISIBLE_DEVICES): {gpu_id}\n"
        f"cwd: {ROOT}\n"
        f"cmd: {pretty}\n"
        f"{'-'*80}\n"
    )
    with open(log_path, "w") as f:
        f.write(header)
        f.flush()
        ret = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n{'-'*80}\nexit_code: {ret.returncode}\n")
    return ret.returncode


# -----------------------------------------------------------------------------
# Per-job pipeline (one (dataset, seed) on one GPU)

def run_one_combo(gpu_id: int, dataset: str, seed: int, args) -> tuple[str, int, str]:
    debug = bool(args.debug)
    reuse = bool(args.reuse_uncond_ckpt)
    seq_length = int(args.seq_length)
    label = (f"{dataset}/seed={seed}/L{seq_length}"
             + (" [debug]" if debug else "")
             + (" [reuse-uncond]" if reuse else ""))
    milestone = args.milestone
    steps = set(args.steps.split(","))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tasks = parse_tasks(args.tasks)

    extra_opts: list[str] = []
    if debug:
        extra_opts += list(DEBUG_OPTS)
    if args.extra_opts:
        extra_opts += args.extra_opts.split()
    fast_steps = DEBUG_FAST_STEPS if debug else None

    # 1) train (skip if reusing uncond ckpts)
    if "train" in steps:
        if reuse:
            need = ckpt_file(dataset, seed, milestone, reuse_uncond=True,
                             seq_length=seq_length, debug=debug)
            if not need.exists():
                return (label, 2, f"reuse-uncond but ckpt missing: {need}")
            print(f"[{label}] train: reuse uncond ckpt {need.name}")
        else:
            out = ckpt_file(dataset, seed, milestone, reuse_uncond=False,
                            seq_length=seq_length, debug=debug)
            if args.skip_existing and out.exists():
                print(f"[{label}] train: skip (exists {out.name})")
            else:
                print(f"[{label}] train: starting on GPU {gpu_id}")
                t0 = time.time()
                rc = run_subprocess(
                    build_train_cmd(dataset, seed, reuse_uncond=False,
                                    seq_length=seq_length,
                                    extra_opts=extra_opts, debug=debug),
                    log_file(dataset, seed, "train", seq_length=seq_length, debug=debug),
                    gpu_id, f"{label} train",
                )
                dt = time.time() - t0
                if rc != 0:
                    return (label, rc, f"train failed (rc={rc}, {dt:.1f}s)")
                print(f"[{label}] train: done in {dt:.1f}s")

    # 2) sample + 3) metric per (task, mode)
    for t_mode, t_val in tasks:
        tag = task_tag(t_mode, t_val)
        cdir = cond_task_dir(dataset, seed, tag, seq_length, debug)

        for mode in modes:
            fake_p = cdir / f"fake_{mode}.npy"
            metric_p = cdir / f"metric_{mode}.json"

            if "sample" in steps:
                if args.skip_existing and fake_p.exists():
                    print(f"[{label}/{tag}/{mode}] sample: skip (exists)")
                else:
                    print(f"[{label}/{tag}/{mode}] sample: starting on GPU {gpu_id}")
                    t0 = time.time()
                    rc = run_subprocess(
                        build_sample_cmd(dataset, seed, milestone, mode,
                                         task_mode=t_mode, task_val=t_val, tag=tag,
                                         reuse_uncond=reuse, seq_length=seq_length,
                                         extra_opts=extra_opts,
                                         fast_steps=fast_steps,
                                         band_langevin=bool(args.band_langevin),
                                         band_projection=bool(args.band_projection),
                                         debug=debug),
                        log_file(dataset, seed, "sample", seq_length=seq_length,
                                 extra=f"{tag}_{mode}", debug=debug),
                        gpu_id, f"{label} sample/{tag}/{mode}",
                    )
                    dt = time.time() - t0
                    if rc != 0:
                        return (label, rc, f"sample/{tag}/{mode} failed (rc={rc}, {dt:.1f}s)")
                    print(f"[{label}/{tag}/{mode}] sample: done in {dt:.1f}s")

            if "metric" in steps:
                if args.skip_existing and metric_p.exists():
                    print(f"[{label}/{tag}/{mode}] metric: skip (exists)")
                else:
                    if not fake_p.exists():
                        return (label, 2, f"metric/{tag}/{mode} missing fake: {fake_p}")
                    if not (cdir / "real.npy").exists() or not (cdir / "mask.npy").exists():
                        return (label, 2, f"metric/{tag}/{mode} missing real.npy/mask.npy in {cdir}")
                    print(f"[{label}/{tag}/{mode}] metric: starting on GPU {gpu_id}")
                    t0 = time.time()
                    rc = run_subprocess(
                        build_metric_cmd(cdir, mode),
                        log_file(dataset, seed, "metric", seq_length=seq_length,
                                 extra=f"{tag}_{mode}", debug=debug),
                        gpu_id, f"{label} metric/{tag}/{mode}",
                    )
                    dt = time.time() - t0
                    if rc != 0:
                        return (label, rc, f"metric/{tag}/{mode} failed (rc={rc}, {dt:.1f}s)")
                    print(f"[{label}/{tag}/{mode}] metric: done in {dt:.1f}s")

    return (label, 0, "ok")


# -----------------------------------------------------------------------------
# Worker / dispatcher (one worker per GPU)

def gpu_worker(gpu_id: int, job_q: mp.Queue, result_q: mp.Queue, args):
    # Workers are spawned as fresh Python processes without -u, so their
    # stdout/stderr are block-buffered when the parent's stdout is redirected
    # to a file (nohup ... > out.log). Force line buffering so the per-step
    # "starting / done" prints show up live.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    print(f"[worker] started on GPU {gpu_id} (pid={os.getpid()})", flush=True)
    while True:
        item = job_q.get()
        if item is None:
            return
        dataset, seed = item
        try:
            res = run_one_combo(gpu_id, dataset, seed, args)
        except Exception as e:
            res = (f"{dataset}/seed={seed}", 1, f"exception: {e!r}")
        result_q.put(res)


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", required=True,
                   help="Comma-separated GPU ids to use, e.g. '0,1,2,3'.")
    p.add_argument("--datasets", default=None,
                   help=f"Subset of {DATASETS}. Default: all (or 'stocks' in --debug).")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seeds. Default: 12345,2026,2027,2028,2029 "
                        "(or just 12345 in --debug).")
    p.add_argument("--modes", default=",".join(MODES_DEFAULT),
                   help=f"Subset of {ALL_MODES}. Default: {MODES_DEFAULT}. "
                        "Includes banded, which dispatches to the stridediff "
                        "frequency-aware dynamic-jump cond sampler with "
                        "per-dataset schedule from BAND_HPARAMS.")
    p.add_argument("--tasks", default=None,
                   help='Conditional tasks, comma-separated. Each entry is '
                        '"infill:<missing_ratio>" or "predict:<pred_len>". '
                        'Default: "infill:0.1,infill:0.25,infill:0.5,infill:0.75,infill:0.9" '
                        '(the 5 missing ratios from the Diffusion-TS paper), '
                        'or just "infill:0.5" in --debug.')
    p.add_argument("--steps", default="train,sample,metric",
                   help="Subset of {train,sample,metric}, comma-separated.")
    p.add_argument("--milestone", type=int, default=None,
                   help="Checkpoint milestone to train up to / load. Default 10 (or 1 in --debug).")
    p.add_argument("--reuse_uncond_ckpt",
                   type=str2bool, nargs="?", const=True, default=False,
                   help="Reuse unconditional checkpoints (trained with proportion=1.0). "
                        "Faster but test windows overlap with the training set; the "
                        "paper numbers assume a 90/10 split. Only valid when "
                        "--seq_length matches the one used for the uncond run "
                        "(by default 24).")
    p.add_argument("--no_reuse_uncond_ckpt", dest="reuse_uncond_ckpt",
                   action="store_false",
                   help="Shorthand for --reuse_uncond_ckpt=false.")
    p.add_argument("--seq_length", type=int, default=48,
                   help="Window length used for both training and cond sampling. "
                        "Default: 48, matching Diffusion-TS paper §4.5 / Fig. 5-6 "
                        "(imputation and forecasting on Stocks/ETTh/Energy/fMRI). "
                        "Overrides model.params.seq_length and the train/test "
                        "dataset `window` in the yaml. Checkpoints land in a "
                        "separate folder per length, so you can coexist 24 vs 48 "
                        "runs.")
    p.add_argument("--skip_existing", action="store_true", default=True,
                   help="Skip a step if its output already exists. (default ON)")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                   help="Force re-run even if outputs exist.")
    p.add_argument("--debug", action="store_true",
                   help="Tiny smoke test: 1 dataset (stocks), 1 seed (12345), modes "
                        "from --modes, task from --tasks (default infill:0.5), "
                        "milestone=1, T=20 diffusion steps, save_cycle=50, fast_steps=10. "
                        "Forces --no_skip_existing. All debug artifacts land in *_debug "
                        "paths so they never collide with real runs.")
    p.add_argument("--extra_opts", default="",
                   help='Extra `key value ...` opts forwarded to main.py.')

    # banded-specific toggles. These must be inserted BEFORE the positional
    # REMAINDER opts (solver.results_folder ...) in the child cmd, so they
    # are real argparse flags rather than config overrides.
    p.add_argument("--band_langevin", dest="band_langevin",
                   type=str2bool, nargs="?", const=True, default=True,
                   help="[banded] Run Diffusion-TS langevin_fn per step. "
                        "Default True (matches cond MSE). Set False for "
                        "pure-jump speed (closer to unconditional sample()).")
    p.add_argument("--no_band_langevin", dest="band_langevin",
                   action="store_false",
                   help="Shorthand for --band_langevin=false.")
    p.add_argument("--band_projection", dest="band_projection",
                   type=str2bool, nargs="?", const=True, default=False,
                   help="[banded] Soft rFFT projection of the jump delta. "
                        "Default False (matches unconditional sample()'s default).")
    p.add_argument("--no_band_projection", dest="band_projection",
                   action="store_false",
                   help="Shorthand for --band_projection=false.")
    args = p.parse_args()

    # --debug defaults
    if args.debug:
        if args.datasets is None:
            args.datasets = "stocks"
        if args.seeds is None:
            args.seeds = "12345"
        if args.milestone is None:
            args.milestone = 1
        if args.tasks is None:
            args.tasks = "infill:0.5"
        args.skip_existing = False

    # normal defaults
    if args.datasets is None:
        args.datasets = ",".join(DATASETS)
    if args.seeds is None:
        args.seeds = ",".join(str(s) for s in SEEDS_DEFAULT)
    if args.milestone is None:
        args.milestone = 10
    if args.tasks is None:
        args.tasks = "infill:0.1,infill:0.25,infill:0.5,infill:0.75,infill:0.9"

    gpus = parse_int_list(args.gpus)
    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset: {d}; choices = {DATASETS}")
    for m in parse_str_list(args.modes):
        if m not in ALL_MODES:
            raise SystemExit(f"unknown mode: {m}; choices = {ALL_MODES}")
    tasks = parse_tasks(args.tasks)  # validate

    if args.reuse_uncond_ckpt and args.seq_length != 24:
        print(f"[warn] --reuse_uncond_ckpt with seq_length={args.seq_length} is "
              f"unusual: uncond runs typically default to seq_length=24. "
              f"Make sure an uncond ckpt at length {args.seq_length} exists, "
              f"or drop --reuse_uncond_ckpt to train fresh cond ckpts.")

    jobs = [(d, s) for d in datasets for s in seeds]
    print(f"\n=== StrideDiff CONDITIONAL pipeline {'[DEBUG]' if args.debug else ''} ===", flush=True)
    print(f"GPUs        : {gpus}", flush=True)
    print(f"Datasets    : {datasets}", flush=True)
    print(f"Seeds       : {seeds}", flush=True)
    print(f"Modes       : {parse_str_list(args.modes)}", flush=True)
    print(f"Tasks       : {tasks}", flush=True)
    print(f"Seq length  : {args.seq_length}  (paper §4.5 uses 48)", flush=True)
    print(f"Steps       : {args.steps}", flush=True)
    print(f"Milestone   : {args.milestone}", flush=True)
    print(f"Reuse uncond: {args.reuse_uncond_ckpt}", flush=True)
    print(f"Skip existing: {args.skip_existing}", flush=True)
    if "banded" in parse_str_list(args.modes):
        print(f"Banded opts : langevin={args.band_langevin} "
              f"projection={args.band_projection}", flush=True)
    if args.debug:
        print(f"Debug opts  : {' '.join(DEBUG_OPTS)}  (fast_steps={DEBUG_FAST_STEPS})", flush=True)
    if args.extra_opts:
        print(f"Extra opts  : {args.extra_opts}", flush=True)
    print(f"Total jobs  : {len(jobs)}  (one per (dataset, seed) pair)", flush=True)
    print(f"Logs go to  : {LOG_ROOT}", flush=True)
    print(flush=True)

    ctx = mp.get_context("spawn")
    job_q: mp.Queue = ctx.Queue()
    res_q: mp.Queue = ctx.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in gpus:
        job_q.put(None)

    workers = []
    for gid in gpus:
        w = ctx.Process(target=gpu_worker, args=(gid, job_q, res_q, args), name=f"gpu{gid}")
        w.start()
        workers.append(w)

    results = []
    t_start = time.time()
    while len(results) < len(jobs):
        try:
            res = res_q.get(timeout=1.0)
            results.append(res)
            label, rc, msg = res
            tag = "OK" if rc == 0 else "FAIL"
            print(f"[{tag}] {label}: {msg}  ({len(results)}/{len(jobs)} done)", flush=True)
        except Empty:
            if not any(w.is_alive() for w in workers):
                while True:
                    try:
                        results.append(res_q.get_nowait())
                    except Empty:
                        break
                break

    for w in workers:
        w.join()

    print(flush=True)
    print(f"=== Summary ({time.time()-t_start:.1f}s) ===", flush=True)
    n_ok = sum(1 for _, rc, _ in results if rc == 0)
    n_fail = len(results) - n_ok
    print(f"OK   : {n_ok}/{len(jobs)}", flush=True)
    print(f"FAIL : {n_fail}/{len(jobs)}", flush=True)
    for label, rc, msg in results:
        if rc != 0:
            print(f"  - {label}: {msg}", flush=True)
    print(f"Logs : {LOG_ROOT}", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
