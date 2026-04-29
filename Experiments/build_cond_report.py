#!/usr/bin/env python
"""Aggregate per-(dataset, seed, task, mode) StrideDiff conditional metric JSONs
into the paper-style table (Markdown + LaTeX).

Per task (e.g. ``infill_mr0.5``), the produced table mirrors the paper's
conditional figure:

    | Metric         | Method             | Stocks | ETTh | Energy | fMRI |
    | MSE Score (v)  | StrideDiff           | ...    | ...  | ...    | ...  |
    |                | StrideDiff-fast      | ...    | ...  | ...    | ...  |
    |                | Ours               | 0.000  | 0.000| 0.000  | 0.000|
    | Time (s)  (v)  | StrideDiff           | ...    | ...  | ...    | ...  |
    |                | StrideDiff-fast      | ...    | ...  | ...    | ...  |
    |                | Ours               | 0.000  | 0.000| 0.000  | 0.000|

Time is taken from ``time_{mode}.json`` (written by main.py during conditional
sampling). MSE is taken from ``metric_{mode}.json`` (written by
compute_cond_metrics.py).

Run from stridediff/ root:
    python Experiments/build_cond_report.py \
        --tasks infill:0.5 \
        --md_out Experiments/cond_report.md \
        --tex_out Experiments/cond_report.tex
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

DATASETS_ALL = ["stocks", "etth", "energy", "fmri"]
DATASETS_LABEL = {
    "stocks": "Stocks",
    "etth":   "ETTh",
    "energy": "Energy",
    "fmri":   "fMRI",
}

ALL_MODES = ["ddpm", "fast200", "banded"]
# StrideDiff supports three conditional paths:
#   ddpm    - full-T DDPM (sample_infill)
#   fast200 - DDIM 200    (fast_sample_infill)
#   banded  - frequency-aware dynamic-jump sampler (sample_infill_banded);
#             the conditional counterpart of stridediff's unconditional
#             `sample()`, parameterised per-dataset by BAND_HPARAMS in
#             run_cond_pipeline.py.
DEFAULT_MODE_LABEL = {
    "ddpm":    "StrideDiff",
    "fast200": "StrideDiff-fast",
    "banded":  "Ours (banded)",
}

EXTRA_BASELINES = ["Ours"]  # placeholder row

ROOT = Path(__file__).resolve().parent.parent  # stridediff/


# -----------------------------------------------------------------------------
# Task parsing

def parse_tasks(s: str) -> list[tuple[str, float]]:
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


def task_title(mode: str, val: float) -> str:
    if mode == "infill":
        return f"Imputation (missing\\_ratio={val:g})"
    return f"Forecasting (pred\\_len={int(val)})"


def task_short_label(mode: str, val: float, kind: str) -> str:
    """Compact row label used inside the combined table's leftmost column."""
    if kind == "mixed":
        return f"mr={val:g}" if mode == "infill" else f"pred_len={int(val)}"
    return f"{val:g}" if mode == "infill" else f"{int(val)}"


def task_short_label_tex(mode: str, val: float, kind: str) -> str:
    if kind == "mixed":
        return f"mr={val:g}" if mode == "infill" else f"pred\\_len={int(val)}"
    return f"{val:g}" if mode == "infill" else f"{int(val)}"


def tasks_kind(tasks: list[tuple[str, float]]) -> str:
    """'forecast', 'impute', or 'mixed' for caption/label wording."""
    kinds = {"predict": "forecast", "infill": "impute"}
    uniq = {kinds[m] for m, _ in tasks}
    return next(iter(uniq)) if len(uniq) == 1 else "mixed"


def setting_column_header(kind: str, use_tex: bool = False) -> str:
    if kind == "forecast":
        return "Pred.\\ Len" if use_tex else "Pred. Len"
    if kind == "impute":
        return "Missing Ratio"
    return "Setting"


# -----------------------------------------------------------------------------
# Loaders

def cond_dir(dataset: str, seed: int, tag: str, seq_length: int) -> Path:
    # Must mirror run_cond_pipeline.py's run_name: <ds>_seed<s>_L<seq>/cond/<tag>.
    return ROOT / "OUTPUT" / f"{dataset}_seed{seed}_L{int(seq_length)}" / "cond" / tag


def load_mse(dataset: str, seed: int, tag: str, mode: str, seq_length: int) -> float | None:
    p = cond_dir(dataset, seed, tag, seq_length) / f"metric_{mode}.json"
    if not p.is_file():
        return None
    with open(p) as f:
        j = json.load(f)
    v = j.get("mse")
    return float(v) if v is not None else None


def load_time(dataset: str, seed: int, tag: str, mode: str, seq_length: int) -> float | None:
    cdir = cond_dir(dataset, seed, tag, seq_length)
    p1 = cdir / f"time_{mode}.json"
    if p1.is_file():
        with open(p1) as f:
            j = json.load(f)
        v = j.get("time_s")
        if v is not None:
            return float(v)
    p2 = cdir / f"metric_{mode}.json"
    if p2.is_file():
        with open(p2) as f:
            j = json.load(f)
        v = j.get("time_s")
        if v is not None:
            return float(v)
    return None


# -----------------------------------------------------------------------------
# Aggregation

def aggregate(values: list[float]) -> tuple[float, float] | None:
    vs = [v for v in values if v is not None and not math.isnan(v)]
    if not vs:
        return None
    if len(vs) == 1:
        return (vs[0], 0.0)
    m = mean(vs)
    s = (sum((x - m) ** 2 for x in vs) / (len(vs) - 1)) ** 0.5
    return (m, s)


def fmt_num(stat, decimals: int, show_std: bool) -> str:
    if stat is None:
        return "N/A"
    m, s = stat
    if show_std:
        return f"{m:.{decimals}f}\u00b1{s:.{decimals}f}"
    return f"{m:.{decimals}f}"


# -----------------------------------------------------------------------------
# Build table for one task

def build_task_table(task_tag_str: str, datasets: list[str], seeds: list[int],
                     modes: list[str], seq_length: int) -> dict:
    table: dict = {'mse': {}, 'time_s': {}}
    for mode in modes:
        label = DEFAULT_MODE_LABEL[mode]
        table['mse'][label] = {}
        table['time_s'][label] = {}
        for d in datasets:
            mses = [load_mse(d, s, task_tag_str, mode, seq_length) for s in seeds]
            times = [load_time(d, s, task_tag_str, mode, seq_length) for s in seeds]
            table['mse'][label][d] = aggregate(mses)
            table['time_s'][label][d] = aggregate(times)
    for extra in EXTRA_BASELINES:
        table['mse'][extra] = {d: (0.0, 0.0) for d in datasets}
        table['time_s'][extra] = {d: (0.0, 0.0) for d in datasets}
    return table


def method_labels(modes: list[str]) -> list[str]:
    return [DEFAULT_MODE_LABEL[m] for m in modes] + list(EXTRA_BASELINES)


# -----------------------------------------------------------------------------
# Rendering

METRIC_ROWS = [
    ("mse",    "MSE Score (\u2193)",    3),
    ("time_s", "Time (s) (\u2193)",     2),
]


def render_markdown_task(task_tag_str: str, task_label: str, table: dict,
                         datasets: list[str], modes: list[str],
                         show_std: bool) -> str:
    methods = method_labels(modes)
    headers = ["Metric", "Method"] + [DATASETS_LABEL[d] for d in datasets]
    lines = []
    lines.append(f"## Task: {task_label}  (`{task_tag_str}`)")
    lines.append("")
    if show_std:
        lines.append("> Mean $\\pm$ standard deviation across seeds. Lower is better for all metrics.")
    else:
        lines.append("> Mean across seeds. Lower is better for all metrics.")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for mk, mlabel, decs in METRIC_ROWS:
        for i, method in enumerate(methods):
            row = [mlabel if i == 0 else "", method]
            for d in datasets:
                row.append(fmt_num(table[mk][method].get(d), decs, show_std))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def _to_tex_cell(s: str) -> str:
    if s == "N/A":
        return "N/A"
    return s.replace("\u00b1", "$\\pm$")


def render_latex_task(task_tag_str: str, task_label: str, table: dict,
                      datasets: list[str], modes: list[str],
                      show_std: bool) -> str:
    methods = method_labels(modes)
    n_data = len(datasets)
    col_spec = "cl" + "c" * n_data
    headers = ["Metric", "Method"] + [DATASETS_LABEL[d] for d in datasets]

    out = []
    out.append("\\begin{table}[h!]")
    out.append("  \\centering")
    caption = (f"StrideDiff conditional generation results for {task_label}. "
               f"Mean$\\pm$std across seeds." if show_std else
               f"StrideDiff conditional generation results for {task_label}. Mean across seeds.")
    out.append(f"  \\caption{{{caption}}}")
    out.append(f"  \\label{{tab:bd_cond_{task_tag_str}}}")
    out.append("  \\resizebox{\\linewidth}{!}{%")
    out.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")
    out.append("    " + " & ".join(_tex_escape(h) for h in headers) + " \\\\")
    out.append("    \\midrule")

    n_methods = len(methods)
    for mk, mlabel, decs in METRIC_ROWS:
        mlabel_tex = mlabel.replace("\u2193", "$\\downarrow$")
        for i, method in enumerate(methods):
            cells = []
            if i == 0:
                cells.append(f"\\multirow{{{n_methods}}}{{*}}{{{_tex_escape(mlabel_tex)}}}")
            else:
                cells.append("")
            cells.append(_tex_escape(method))
            for d in datasets:
                cells.append(_to_tex_cell(fmt_num(table[mk][method].get(d), decs, show_std)))
            out.append("    " + " & ".join(cells) + " \\\\")
        out.append("    \\midrule")
    out[-1] = "    \\bottomrule"
    out.append("  \\end{tabular}}")
    out.append("\\end{table}")
    return "\n".join(out)


# -----------------------------------------------------------------------------
# Combined table (all tasks in a single table)

def _combined_caption(kind: str, show_std: bool) -> str:
    suffix = "Mean$\\pm$std across seeds." if show_std else "Mean across seeds."
    if kind == "forecast":
        return f"StrideDiff conditional generation results for Forecasting across prediction lengths. {suffix}"
    if kind == "impute":
        return f"StrideDiff conditional generation results for Imputation across missing ratios. {suffix}"
    return f"StrideDiff conditional generation results across tasks. {suffix}"


def _combined_label(kind: str) -> str:
    return {"forecast": "bd_cond_forecast", "impute": "bd_cond_impute"}.get(kind, "bd_cond_combined")


def render_markdown_combined(tasks: list[tuple[str, float]],
                             tables_by_tag: dict, datasets: list[str],
                             modes: list[str], show_std: bool) -> str:
    methods = method_labels(modes)
    kind = tasks_kind(tasks)
    title = {"forecast": "Forecasting across prediction lengths",
             "impute":   "Imputation across missing ratios"}.get(kind, "Conditional generation")

    headers = [setting_column_header(kind), "Method"]
    for d in datasets:
        headers += [f"{DATASETS_LABEL[d]} MSE ↓",
                    f"{DATASETS_LABEL[d]} Time(s) ↓"]

    lines = [f"## {title}", ""]
    if show_std:
        lines.append("> Mean $\\pm$ standard deviation across seeds. Lower is better for all metrics.")
    else:
        lines.append("> Mean across seeds. Lower is better for all metrics.")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for t_mode, t_val in tasks:
        tag = task_tag(t_mode, t_val)
        label = task_short_label(t_mode, t_val, kind)
        table = tables_by_tag[tag]
        for j, method in enumerate(methods):
            row = [label if j == 0 else "", method]
            for d in datasets:
                row.append(fmt_num(table['mse'][method].get(d), 3, show_std))
                row.append(fmt_num(table['time_s'][method].get(d), 2, show_std))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_latex_combined(tasks: list[tuple[str, float]],
                          tables_by_tag: dict, datasets: list[str],
                          modes: list[str], show_std: bool) -> str:
    methods = method_labels(modes)
    n_data = len(datasets)
    n_methods = len(methods)
    col_spec = "ll" + "cc" * n_data
    kind = tasks_kind(tasks)

    out = []
    out.append("\\begin{table}[t]")
    out.append("  \\centering")
    out.append(f"  \\caption{{{_combined_caption(kind, show_std)}}}")
    out.append(f"  \\label{{tab:{_combined_label(kind)}}}")
    out.append("  \\resizebox{\\linewidth}{!}{%")
    out.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    out.append("    \\toprule")

    h1 = [setting_column_header(kind, use_tex=True), "Method"] + [
        f"\\multicolumn{{2}}{{c}}{{{_tex_escape(DATASETS_LABEL[d])}}}" for d in datasets
    ]
    out.append("    " + " & ".join(h1) + " \\\\")
    cmids = " ".join(
        f"\\cmidrule(lr){{{3 + 2*k}-{4 + 2*k}}}" for k in range(n_data)
    )
    out.append(f"    {cmids}")
    h2 = ["", ""]
    for _ in datasets:
        h2 += ["MSE $\\downarrow$", "Time(s) $\\downarrow$"]
    out.append("    " + " & ".join(h2) + " \\\\")
    out.append("    \\midrule")

    for i, (t_mode, t_val) in enumerate(tasks):
        tag = task_tag(t_mode, t_val)
        label_tex = task_short_label_tex(t_mode, t_val, kind)
        table = tables_by_tag[tag]
        for j, method in enumerate(methods):
            cells = []
            if j == 0:
                cells.append(f"\\multirow{{{n_methods}}}{{*}}{{{label_tex}}}")
            else:
                cells.append("")
            cells.append(_tex_escape(method))
            for d in datasets:
                cells.append(_to_tex_cell(fmt_num(table['mse'][method].get(d), 3, show_std)))
                cells.append(_to_tex_cell(fmt_num(table['time_s'][method].get(d), 2, show_std)))
            out.append("    " + " & ".join(cells) + " \\\\")
        if i < len(tasks) - 1:
            out.append("    \\midrule")

    out.append("    \\bottomrule")
    out.append("  \\end{tabular}}")
    out.append("\\end{table}")
    return "\n".join(out)


# -----------------------------------------------------------------------------
# Main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(DATASETS_ALL))
    p.add_argument("--seeds",    default="12345,2026,2027,2028,2029")
    p.add_argument("--modes",    default="ddpm,fast200,banded",
                   help="Subset of {ddpm,fast200,banded}. Default: "
                        "'ddpm,fast200,banded' (StrideDiff full-T DDPM + "
                        "200-step DDIM + stridediff frequency-aware "
                        "dynamic-jump sampler).")
    p.add_argument("--tasks",    default="infill:0.1,infill:0.25,infill:0.5,infill:0.75,infill:0.9",
                   help='Tasks to aggregate, comma-separated. Default: the 5 '
                        'imputation ratios from the Diffusion-TS paper.')
    p.add_argument("--seq_length", type=int, default=48,
                   help="Window length that was used in run_cond_pipeline.py. "
                        "Default 48 (paper §4.5). Used to locate the right "
                        "OUTPUT/<ds>_seed<s>_L<seq>/cond/<task>/ folder.")
    p.add_argument("--show_std", action="store_true", default=False,
                   help="Render cells as mean+/-std (default off; matches the paper figure layout).")
    p.add_argument("--layout",   choices=["combined", "per_task"], default="combined",
                   help="'combined' (default) merges all --tasks into ONE table. "
                        "'per_task' emits one table per task (legacy behavior).")
    p.add_argument("--md_out",   default="Experiments/cond_report.md")
    p.add_argument("--tex_out",  default="Experiments/cond_report.tex")
    p.add_argument("--print",    action="store_true", help="Also print Markdown to stdout.")
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tasks = parse_tasks(args.tasks)

    for d in datasets:
        if d not in DATASETS_ALL:
            raise SystemExit(f"unknown dataset: {d}; choices = {DATASETS_ALL}")
    for m in modes:
        if m not in ALL_MODES:
            raise SystemExit(f"unknown mode: {m}; choices = {ALL_MODES}")

    md_parts = ["# StrideDiff Conditional Time-Series Generation Results\n"]
    tex_parts = []
    tables_by_tag: dict = {}
    for t_mode, t_val in tasks:
        tag = task_tag(t_mode, t_val)
        tables_by_tag[tag] = build_task_table(tag, datasets, seeds, modes, args.seq_length)

    if args.layout == "combined":
        md_parts.append(render_markdown_combined(tasks, tables_by_tag, datasets, modes, args.show_std))
        tex_parts.append(render_latex_combined(tasks, tables_by_tag, datasets, modes, args.show_std))
    else:
        for t_mode, t_val in tasks:
            tag = task_tag(t_mode, t_val)
            title = task_title(t_mode, t_val)
            table = tables_by_tag[tag]
            md_parts.append(render_markdown_task(tag, title, table, datasets, modes, args.show_std))
            tex_parts.append(render_latex_task(tag, title, table, datasets, modes, args.show_std))

    md = "\n".join(md_parts) + "\n"
    tex = "\n\n".join(tex_parts) + "\n"

    md_path = ROOT / args.md_out
    tex_path = ROOT / args.tex_out
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    tex_path.write_text(tex)

    print(f"=> wrote {md_path}")
    print(f"=> wrote {tex_path}")
    if args.print:
        print()
        print(md)


if __name__ == "__main__":
    main()
