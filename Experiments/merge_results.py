import argparse
import csv
import glob
import os
from collections import defaultdict


def maybe_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def maybe_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def load_one_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["source_csv"] = path
        r["best_cfid"] = maybe_float(r.get("best_cfid"))
        r["mean_cfid"] = maybe_float(r.get("mean_cfid"))
        r["mean_sigma"] = maybe_float(r.get("mean_sigma"))
        r["best_corr"] = maybe_float(r.get("best_corr"))
        r["mean_corr"] = maybe_float(r.get("mean_corr"))
        r["best_disc"] = maybe_float(r.get("best_disc"))
        r["mean_disc"] = maybe_float(r.get("mean_disc"))
        r["best_pred"] = maybe_float(r.get("best_pred"))
        r["mean_pred"] = maybe_float(r.get("mean_pred"))
        r["sample_fn_sec"] = maybe_float(r.get("sample_fn_sec"))
        r["sampling_done_sec"] = maybe_float(r.get("sampling_done_sec"))
        r["sample_fn_count"] = maybe_int(r.get("sample_fn_count"))
    return rows


def fmt_num(x, nd=6):
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def mean_of(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def group_stat(valid_rows, key):
    groups = defaultdict(list)
    for r in valid_rows:
        groups[str(r.get(key))].append(r)
    stat = []
    for k, rows in groups.items():
        stat.append({
            "value": k,
            "count": len(rows),
            "best_cfid_mean": mean_of([x.get("best_cfid") for x in rows]),
            "best_corr_mean": mean_of([x.get("best_corr") for x in rows]),
            "best_disc_mean": mean_of([x.get("best_disc") for x in rows]),
            "best_pred_mean": mean_of([x.get("best_pred") for x in rows]),
            "sample_fn_sec_mean": mean_of([x.get("sample_fn_sec") for x in rows]),
            "sampling_done_sec_mean": mean_of([x.get("sampling_done_sec") for x in rows]),
        })
    stat.sort(key=lambda x: (x["best_cfid_mean"] is None, x["best_cfid_mean"]))
    return stat


def rank_candidate_values(rows, key, cast_fn, max_items=3):
    """
    Pick candidate values from top rows by:
    1) appearing more often
    2) appearing at better rank
    """
    stat = {}
    for idx, r in enumerate(rows, start=1):
        v = cast_fn(r.get(key))
        if v is None:
            continue
        if v not in stat:
            stat[v] = {"count": 0, "best_rank": idx}
        stat[v]["count"] += 1
        stat[v]["best_rank"] = min(stat[v]["best_rank"], idx)

    ranked = sorted(
        stat.items(),
        key=lambda kv: (-kv[1]["count"], kv[1]["best_rank"], kv[0]),
    )
    picked = [k for k, _ in ranked[:max_items]]
    picked = sorted(picked)
    return picked


def build_next_search_suggestion(valid_sorted, seed_top_n=20):
    seed = valid_sorted[: min(seed_top_n, len(valid_sorted))]
    if not seed:
        return {}

    suggestion = {
        "big_k": rank_candidate_values(seed, "big_k", maybe_int, max_items=3),
        "med_k": rank_candidate_values(seed, "med_k", maybe_int, max_items=3),
        "small_k": rank_candidate_values(seed, "small_k", maybe_int, max_items=2),
        "last_k": rank_candidate_values(seed, "last_k", maybe_int, max_items=3),
        "tau_energy": rank_candidate_values(seed, "tau_energy", maybe_float, max_items=3),
        "tau_dlogP": rank_candidate_values(seed, "tau_dlogP", maybe_float, max_items=3),
        "tau_pv": rank_candidate_values(seed, "tau_pv", maybe_float, max_items=3),
    }
    return suggestion


def list_to_csv(values, nd=6):
    out = []
    for v in values:
        if isinstance(v, float):
            out.append(f"{v:.{nd}g}")
        else:
            out.append(str(v))
    return ",".join(out)


def to_md_table(rows, columns):
    # columns: list of (key, title)
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for r in rows:
        vals = []
        for k, _ in columns:
            v = r.get(k)
            if isinstance(v, float):
                vals.append(fmt_num(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


SORT_KEY_MAP = {
    "cfid": ("best_cfid", "C-FID"),
    "disc": ("best_disc", "Discriminative"),
    "corr": ("best_corr", "Correlation"),
    "pred": ("best_pred", "Predictive"),
}


def write_report_md(run_root, merged, valid_sorted, top, top_k, sort_by="cfid"):
    report_path = os.path.join(run_root, "report.md")
    if not valid_sorted:
        content = "# Hyperparameter Search Report\n\nNo valid rows found.\n"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        return report_path

    best = valid_sorted[0]
    avg_best_cfid = mean_of([r.get("best_cfid") for r in valid_sorted])
    avg_best_corr = mean_of([r.get("best_corr") for r in valid_sorted])
    avg_best_disc = mean_of([r.get("best_disc") for r in valid_sorted])
    avg_best_pred = mean_of([r.get("best_pred") for r in valid_sorted])
    avg_sample_fn_sec = mean_of([r.get("sample_fn_sec") for r in valid_sorted])
    avg_sampling_done_sec = mean_of([r.get("sampling_done_sec") for r in valid_sorted])
    has_corr = avg_best_corr is not None
    has_disc = avg_best_disc is not None
    has_pred = avg_best_pred is not None

    top_rows = []
    for i, r in enumerate(top, start=1):
        row = {
            "rank": i,
            "best_cfid": r.get("best_cfid"),
            "mean_cfid": r.get("mean_cfid"),
            "mean_sigma": r.get("mean_sigma"),
            "best_corr": r.get("best_corr"),
            "mean_corr": r.get("mean_corr"),
            "best_disc": r.get("best_disc"),
            "mean_disc": r.get("mean_disc"),
            "best_pred": r.get("best_pred"),
            "mean_pred": r.get("mean_pred"),
            "big_k": r.get("big_k"),
            "med_k": r.get("med_k"),
            "small_k": r.get("small_k"),
            "last_k": r.get("last_k"),
            "tau_energy": r.get("tau_energy"),
            "tau_dlogP": r.get("tau_dlogP"),
            "tau_pv": r.get("tau_pv"),
            "sample_fn_sec": r.get("sample_fn_sec"),
            "sampling_done_sec": r.get("sampling_done_sec"),
        }
        top_rows.append(row)

    by_big = group_stat(valid_sorted, "big_k")
    by_med = group_stat(valid_sorted, "med_k")
    by_last = group_stat(valid_sorted, "last_k")
    by_te = group_stat(valid_sorted, "tau_energy")
    by_td = group_stat(valid_sorted, "tau_dlogP")
    by_tp = group_stat(valid_sorted, "tau_pv")
    suggestion = build_next_search_suggestion(valid_sorted, seed_top_n=20)

    lines = []
    lines.append("# Hyperparameter Search Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- total_rows: {len(merged)}")
    lines.append(f"- valid_rows: {len(valid_sorted)}")
    lines.append(f"- top_k: {top_k}")
    lines.append(f"- avg_best_cfid: {fmt_num(avg_best_cfid)}")
    if has_corr:
        lines.append(f"- avg_best_corr: {fmt_num(avg_best_corr)}")
    if has_disc:
        lines.append(f"- avg_best_disc: {fmt_num(avg_best_disc)}")
    if has_pred:
        lines.append(f"- avg_best_pred: {fmt_num(avg_best_pred)}")
    lines.append(f"- avg_sample_fn_sec: {fmt_num(avg_sample_fn_sec)}")
    lines.append(f"- avg_sampling_done_sec: {fmt_num(avg_sampling_done_sec)}")
    sort_field, sort_label = SORT_KEY_MAP.get(sort_by, SORT_KEY_MAP["cfid"])
    lines.append("")
    lines.append(f"## Best Params (by {sort_label})")
    lines.append(f"- best_cfid: {fmt_num(best.get('best_cfid'))}")
    if has_corr:
        lines.append(f"- best_corr: {fmt_num(best.get('best_corr'))}")
    if has_disc:
        lines.append(f"- best_disc: {fmt_num(best.get('best_disc'))}")
    if has_pred:
        lines.append(f"- best_pred: {fmt_num(best.get('best_pred'))}")
    lines.append(
        f"- params: big_k={best.get('big_k')}, med_k={best.get('med_k')}, small_k={best.get('small_k')}, "
        f"last_k={best.get('last_k')}, tau_energy={best.get('tau_energy')}, "
        f"tau_dlogP={best.get('tau_dlogP')}, tau_pv={best.get('tau_pv')}"
    )
    lines.append(f"- sample_fn_sec: {fmt_num(best.get('sample_fn_sec'))}")
    lines.append(f"- sampling_done_sec: {fmt_num(best.get('sampling_done_sec'))}")
    lines.append("")
    lines.append("## Suggested Next Search Range")
    if suggestion:
        lines.append("- based_on: global top-20")
        lines.append(f"- big_k_list: {list_to_csv(suggestion.get('big_k', []))}")
        lines.append(f"- med_k_list: {list_to_csv(suggestion.get('med_k', []))}")
        lines.append(f"- small_k_list: {list_to_csv(suggestion.get('small_k', []))}")
        lines.append(f"- last_k_list: {list_to_csv(suggestion.get('last_k', []))}")
        lines.append(f"- tau_energy_list: {list_to_csv(suggestion.get('tau_energy', []))}")
        lines.append(f"- tau_dlogP_list: {list_to_csv(suggestion.get('tau_dlogP', []))}")
        lines.append(f"- tau_pv_list: {list_to_csv(suggestion.get('tau_pv', []))}")
        lines.append("")
        lines.append("Recommended command snippet:")
        lines.append("```bash")
        lines.append("python -u Experiments/hparam_search.py \\")
        lines.append("  --name etth \\")
        lines.append("  --config_file ./Config/etth.yaml \\")
        lines.append("  --milestone 10 \\")
        lines.append("  --gpu 0 \\")
        lines.append(f"  --big_k_list \"{list_to_csv(suggestion.get('big_k', []))}\" \\")
        lines.append(f"  --med_k_list \"{list_to_csv(suggestion.get('med_k', []))}\" \\")
        lines.append(f"  --small_k_list \"{list_to_csv(suggestion.get('small_k', []))}\" \\")
        lines.append(f"  --last_k_list \"{list_to_csv(suggestion.get('last_k', []))}\" \\")
        lines.append(f"  --tau_energy_list \"{list_to_csv(suggestion.get('tau_energy', []))}\" \\")
        lines.append(f"  --tau_dlogP_list \"{list_to_csv(suggestion.get('tau_dlogP', []))}\" \\")
        lines.append(f"  --tau_pv_list \"{list_to_csv(suggestion.get('tau_pv', []))}\" \\")
        lines.append("  --search_mode random --max_trials 200 --eval_iterations 2 --eval_repeats 1")
        lines.append("```")
    else:
        lines.append("- no_suggestion: insufficient valid rows")
    lines.append("")
    lines.append(f"## Top-{top_k}")
    top_columns = [
        ("rank", "rank"),
        ("best_cfid", "best_cfid"),
        ("mean_cfid", "mean_cfid"),
        ("mean_sigma", "mean_sigma"),
    ]
    if has_corr:
        top_columns += [("best_corr", "best_corr"), ("mean_corr", "mean_corr")]
    if has_disc:
        top_columns += [("best_disc", "best_disc"), ("mean_disc", "mean_disc")]
    if has_pred:
        top_columns += [("best_pred", "best_pred"), ("mean_pred", "mean_pred")]
    top_columns += [
        ("big_k", "big_k"),
        ("med_k", "med_k"),
        ("small_k", "small_k"),
        ("last_k", "last_k"),
        ("tau_energy", "tau_energy"),
        ("tau_dlogP", "tau_dlogP"),
        ("tau_pv", "tau_pv"),
        ("sample_fn_sec", "sample_fn_sec"),
        ("sampling_done_sec", "sampling_done_sec"),
    ]
    lines.append(to_md_table(top_rows, top_columns))
    lines.append("")
    lines.append("## Simple Parameter Stats (mean over rows with same value)")
    stat_columns = [
        ("value", "value"),
        ("count", "count"),
        ("best_cfid_mean", "best_cfid_mean"),
    ]
    if has_corr:
        stat_columns.append(("best_corr_mean", "best_corr_mean"))
    if has_disc:
        stat_columns.append(("best_disc_mean", "best_disc_mean"))
    if has_pred:
        stat_columns.append(("best_pred_mean", "best_pred_mean"))
    stat_columns += [
        ("sample_fn_sec_mean", "sample_fn_sec_mean"),
        ("sampling_done_sec_mean", "sampling_done_sec_mean"),
    ]
    for title, rows in [
        ("big_k", by_big),
        ("med_k", by_med),
        ("last_k", by_last),
        ("tau_energy", by_te),
        ("tau_dlogP", by_td),
        ("tau_pv", by_tp),
    ]:
        lines.append(f"### {title}")
        lines.append(to_md_table(rows, stat_columns))
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Merge worker results.csv and output global Top-K.")
    parser.add_argument("--run_root", type=str, required=True,
                        help="Path like OUTPUT/etth/hparam_search/<run_tag>")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--sort_by", type=str, choices=["cfid", "disc", "corr", "pred"], default="cfid",
                        help="Metric to sort/rank by. Default: cfid")
    args = parser.parse_args()

    pattern = os.path.join(args.run_root, "worker_*_of_*", "results.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No results.csv found under: {pattern}")

    merged = []
    for p in csv_files:
        merged.extend(load_one_csv(p))

    merged_csv = os.path.join(args.run_root, "merged_results.csv")
    top_csv = os.path.join(args.run_root, f"top_{args.top_k}.csv")

    # Keep stable field order.
    fieldnames = list(merged[0].keys())
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    sort_field = SORT_KEY_MAP[args.sort_by][0]
    valid = [r for r in merged if r.get("status") == "ok" and r.get(sort_field) is not None]
    valid_sorted = sorted(valid, key=lambda x: x[sort_field])
    top = valid_sorted[: max(args.top_k, 0)]

    with open(top_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top)

    report_md = write_report_md(args.run_root, merged, valid_sorted, top, args.top_k, sort_by=args.sort_by)

    sort_label = SORT_KEY_MAP[args.sort_by][1]
    print(f"Merged {len(csv_files)} worker csv files.")
    print(f"Total rows: {len(merged)}")
    print(f"Valid rows: {len(valid)}")
    print(f"Sort by: {sort_label} ({sort_field})")
    print(f"Merged CSV: {merged_csv}")
    print(f"Top-{args.top_k} CSV: {top_csv}")
    print(f"Report MD: {report_md}")

    if not top:
        print("No valid rows for ranking.")
        return

    print(f"\nTop entries (by {sort_label}):")
    for i, r in enumerate(top[: min(10, len(top))], start=1):
        extra = ""
        if r.get("best_corr") is not None:
            extra += f", corr={r['best_corr']:.6f}"
        if r.get("best_disc") is not None:
            extra += f", disc={r['best_disc']:.6f}"
        if r.get("best_pred") is not None:
            extra += f", pred={r['best_pred']:.6f}"
        print(
            f"#{i} best_cfid={r['best_cfid']:.6f}{extra}, "
            f"b={r.get('big_k')}, m={r.get('med_k')}, s={r.get('small_k')}, l={r.get('last_k')}, "
            f"te={r.get('tau_energy')}, td={r.get('tau_dlogP')}, tp={r.get('tau_pv')}, "
            f"sample_fn_sec={r.get('sample_fn_sec')}, sampling_done_sec={r.get('sampling_done_sec')}"
        )


if __name__ == "__main__":
    main()
