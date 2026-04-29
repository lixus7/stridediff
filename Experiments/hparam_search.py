import argparse
import csv
import itertools
import os
import random
import re
import subprocess
import time
from datetime import datetime


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_metric_from_log(log_path, tag=None):
    """Parse a tagged metric from a trial log file.

    If *tag* is given, matches lines like ``[C-FID] Final Score:  X ± Y``.
    If *tag* is None, matches **untagged** ``Final Score:  X ± Y`` lines
    (backward compat with old logs).

    Returns (best, mean, mean_sigma) or (None, None, None).
    """
    if not os.path.exists(log_path):
        return None, None, None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    num_re = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    if tag:
        pat = re.compile(
            r"\[" + re.escape(tag) + r"\]\s*Final Score:\s*" + num_re + r"\s*±\s*" + num_re
        )
    else:
        pat = re.compile(r"Final Score:\s*" + num_re + r"\s*±\s*" + num_re)

    matches = pat.findall(content)
    if not matches:
        return None, None, None

    means = [float(m[0]) for m in matches]
    sigmas = [float(m[1]) for m in matches]
    return min(means), sum(means) / len(means), sum(sigmas) / len(sigmas)


def parse_cfid_from_log(log_path):
    best, mean, sigma = parse_metric_from_log(log_path, tag="C-FID")
    if best is not None:
        return best, mean, sigma
    return parse_metric_from_log(log_path, tag=None)


def parse_timing_from_log(log_path):
    if not os.path.exists(log_path):
        return None, None, 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    sample_fn_pat = re.compile(
        r"Sampling complete using frequency-aware sampler\. Total time taken from Diffusion-TS->sample fn:\s*([+-]?\d+(?:\.\d+)?)\s*seconds"
    )
    sampling_done_pat = re.compile(
        r"Sampling done,\s*time:\s*([+-]?\d+(?:\.\d+)?)"
    )

    sample_fn_matches = sample_fn_pat.findall(content)
    sampling_done_matches = sampling_done_pat.findall(content)

    sample_fn_last = float(sample_fn_matches[-1]) if sample_fn_matches else None
    sampling_done_last = float(sampling_done_matches[-1]) if sampling_done_matches else None
    sample_fn_count = len(sample_fn_matches)
    return sample_fn_last, sampling_done_last, sample_fn_count


def parse_topk_combos_from_report(report_path, top_k=50, dedup=False):
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Locate the "Top-50" section table.
    table_lines = []
    in_top_section = False
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip().startswith("## Top-"):
            in_top_section = True
            continue
        if in_top_section and line.strip().startswith("## "):
            break
        if in_top_section and line.strip().startswith("|"):
            table_lines.append(line)

    if len(table_lines) < 3:
        raise ValueError(
            "Top-k table not found or malformed in report. "
            "Expected a markdown table under a section like '## Top-50'."
        )

    # Header row like:
    # | rank | best_cfid | ... | big_k | med_k | ... | tau_pv | ... |
    header_cells = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
    col_idx = {name: idx for idx, name in enumerate(header_cells)}
    required = ["big_k", "med_k", "small_k", "last_k", "tau_energy", "tau_dlogP", "tau_pv"]
    missing = [k for k in required if k not in col_idx]
    if missing:
        raise ValueError(f"Missing required columns in top-k table: {missing}")

    combos = []
    seen = set()
    for row in table_lines[2:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < len(header_cells):
            continue
        combo = (
            int(cells[col_idx["big_k"]]),
            int(cells[col_idx["med_k"]]),
            int(cells[col_idx["small_k"]]),
            int(cells[col_idx["last_k"]]),
            float(cells[col_idx["tau_energy"]]),
            float(cells[col_idx["tau_dlogP"]]),
            float(cells[col_idx["tau_pv"]]),
        )
        if dedup:
            if combo in seen:
                continue
            seen.add(combo)
        combos.append(combo)
        if len(combos) >= top_k:
            break

    if not combos:
        raise ValueError("No hyperparameter rows parsed from the report top-k table.")
    return combos


def run_one_trial(args, combo, trial_idx, total_trials, run_dir):
    big_k, med_k, small_k, last_k, tau_energy, tau_dlogP, tau_pv = combo
    tag = (
        f"b{big_k}_m{med_k}_s{small_k}_l{last_k}"
        f"_te{tau_energy}_td{tau_dlogP}_tp{tau_pv}"
    )
    log_path = os.path.join(run_dir, f"trial_{trial_idx:03d}_{tag}.log")

    cmd = [
        "python",
        "-u",
        "main.py",
        "--name",
        args.name,
        "--config_file",
        args.config_file,
        "--sample",
        "0",
        "--milestone",
        str(args.milestone),
        "--gpu",
        str(args.gpu),
        "--big_k",
        str(big_k),
        "--med_k",
        str(med_k),
        "--small_k",
        str(small_k),
        "--last_k_always_micro",
        str(last_k),
        "--tau_energy",
        str(tau_energy),
        "--tau_dlogP",
        str(tau_dlogP),
        "--tau_pv",
        str(tau_pv),
        "--eval_cfid",
        "--eval_iterations",
        str(args.eval_iterations),
        "--eval_repeats",
        str(args.eval_repeats),
        "--no_save_npy",
    ]
    if args.eval_corr:
        cmd.append("--eval_corr")
    if args.eval_disc:
        cmd.append("--eval_disc")
    if args.eval_pred:
        cmd.append("--eval_pred")

    print(f"[Trial {trial_idx}/{total_trials}] start: {tag}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=args.workdir)
    elapsed = time.time() - t0

    best_cfid, mean_cfid, mean_sigma = parse_cfid_from_log(log_path)
    sample_fn_sec, sampling_done_sec, sample_fn_count = parse_timing_from_log(log_path)

    best_corr, mean_corr, _ = parse_metric_from_log(log_path, tag="Correlation")
    best_disc, mean_disc, _ = parse_metric_from_log(log_path, tag="Discriminative")
    best_pred, mean_pred, _ = parse_metric_from_log(log_path, tag="Predictive")

    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
    extra_parts = []
    if best_corr is not None:
        extra_parts.append(f"corr={best_corr:.6f}")
    if best_disc is not None:
        extra_parts.append(f"disc={best_disc:.6f}")
    if best_pred is not None:
        extra_parts.append(f"pred={best_pred:.6f}")
    extra_str = (", " + ", ".join(extra_parts)) if extra_parts else ""
    print(
        f"[Trial {trial_idx}/{total_trials}] done: {tag}, "
        f"status={status}, best_cfid={best_cfid}{extra_str}, elapsed={elapsed:.1f}s, "
        f"sample_fn_sec={sample_fn_sec}, sampling_done_sec={sampling_done_sec}"
    )

    return {
        "trial": trial_idx,
        "status": status,
        "return_code": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "big_k": big_k,
        "med_k": med_k,
        "small_k": small_k,
        "last_k": last_k,
        "tau_energy": tau_energy,
        "tau_dlogP": tau_dlogP,
        "tau_pv": tau_pv,
        "best_cfid": best_cfid,
        "mean_cfid": mean_cfid,
        "mean_sigma": mean_sigma,
        "best_corr": best_corr,
        "mean_corr": mean_corr,
        "best_disc": best_disc,
        "mean_disc": mean_disc,
        "best_pred": best_pred,
        "mean_pred": mean_pred,
        "sample_fn_sec": sample_fn_sec,
        "sampling_done_sec": sampling_done_sec,
        "sample_fn_count": sample_fn_count,
        "log_path": log_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid/random search for sampling + tau hyperparameters.")
    parser.add_argument("--name", type=str, default="etth")
    parser.add_argument("--config_file", type=str, default="./Config/etth.yaml")
    parser.add_argument("--milestone", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--workdir", type=str, default=".")

    parser.add_argument("--big_k_list", type=str, default="10,20,30,40,50")
    parser.add_argument("--med_k_list", type=str, default="5,10,20")
    parser.add_argument("--small_k_list", type=str, default="1")
    parser.add_argument("--last_k_list", type=str, default="12,20,40")
    parser.add_argument("--tau_energy_list", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--tau_dlogP_list", type=str, default="0.005,0.01,0.02,0.05")
    parser.add_argument("--tau_pv_list", type=str, default="0.04,0.06,0.08")

    parser.add_argument("--search_mode", type=str, choices=["grid", "random", "report_topk"], default="random")
    parser.add_argument("--max_trials", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--report_path", type=str, default="",
                        help="When --search_mode=report_topk, read Top-K params from this report.md.")
    parser.add_argument("--report_top_k", type=int, default=50,
                        help="How many rows to replay from report top table.")
    parser.add_argument("--report_dedup", action="store_true",
                        help="Deduplicate same param combos when replaying report top-k.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Total parallel workers. Use with --worker_id for disjoint sharding.")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Current worker index in [0, num_workers-1].")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Shared tag for a multi-worker run. Empty means auto timestamp.")

    parser.add_argument("--eval_iterations", type=int, default=2)
    parser.add_argument("--eval_repeats", type=int, default=1)
    parser.add_argument("--eval_corr", action="store_true", default=False,
                        help="Also compute cross-correlation score.")
    parser.add_argument("--eval_disc", action="store_true", default=False,
                        help="Also compute discriminative score (requires TensorFlow).")
    parser.add_argument("--eval_pred", action="store_true", default=False,
                        help="Also compute predictive score (requires TensorFlow).")
    args = parser.parse_args()

    big_k_list = parse_int_list(args.big_k_list)
    med_k_list = parse_int_list(args.med_k_list)
    small_k_list = parse_int_list(args.small_k_list)
    last_k_list = parse_int_list(args.last_k_list)
    tau_energy_list = parse_float_list(args.tau_energy_list)
    tau_dlogP_list = parse_float_list(args.tau_dlogP_list)
    tau_pv_list = parse_float_list(args.tau_pv_list)

    if args.search_mode == "report_topk":
        if not args.report_path.strip():
            raise ValueError("--report_path is required when --search_mode=report_topk.")
        combos = parse_topk_combos_from_report(
            args.report_path.strip(),
            top_k=args.report_top_k,
            dedup=args.report_dedup,
        )
    else:
        combos = list(
            itertools.product(
                big_k_list,
                med_k_list,
                small_k_list,
                last_k_list,
                tau_energy_list,
                tau_dlogP_list,
                tau_pv_list,
            )
        )

        # Basic pruning: med_k <= big_k, small_k <= med_k
        combos = [c for c in combos if c[1] <= c[0] and c[2] <= c[1]]
        if not combos:
            raise ValueError("No valid hyperparameter combinations after pruning.")

    if args.num_workers <= 0:
        raise ValueError("--num_workers must be >= 1.")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("--worker_id must be in [0, num_workers-1].")

    rng = random.Random(args.random_seed)
    if args.search_mode == "random":
        total_pool = min(args.max_trials, len(combos))
        rng.shuffle(combos)
        combos = combos[:total_pool]
    elif args.search_mode == "grid":
        if args.max_trials > 0:
            combos = combos[: min(args.max_trials, len(combos))]
        total_pool = len(combos)
    else:
        total_pool = len(combos)

    # Deterministic disjoint shard for this worker.
    combos = combos[args.worker_id::args.num_workers]
    total_trials = len(combos)

    ts = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        args.workdir,
        "OUTPUT",
        args.name,
        "hparam_search",
        ts,
        f"worker_{args.worker_id:02d}_of_{args.num_workers:02d}",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"Candidate pool size before sharding: {total_pool}")
    print(f"Assigned trials for this worker: {total_trials}")

    if total_trials == 0:
        print("No trial assigned to this worker. Exit.")
        return

    results = []
    for idx, combo in enumerate(combos, start=1):
        result = run_one_trial(args, combo, idx, total_trials, run_dir)
        results.append(result)

    result_csv = os.path.join(run_dir, "results.csv")
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    valid = [r for r in results if r["best_cfid"] is not None and r["status"] == "ok"]
    valid_sorted = sorted(valid, key=lambda x: x["best_cfid"])

    print("\n=== Search Summary ===")
    print(f"Results CSV: {result_csv}")
    if not valid_sorted:
        print("No successful trials with parsable C-FID.")
        return

    top_k = min(5, len(valid_sorted))
    print(f"Top-{top_k} by best_cfid:")
    for rank, r in enumerate(valid_sorted[:top_k], start=1):
        print(
            f"#{rank} best_cfid={r['best_cfid']:.6f} "
            f"(b={r['big_k']}, m={r['med_k']}, s={r['small_k']}, l={r['last_k']}, "
            f"te={r['tau_energy']}, td={r['tau_dlogP']}, tp={r['tau_pv']})"
        )


if __name__ == "__main__":
    main()
