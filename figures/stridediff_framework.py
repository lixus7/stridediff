"""
StrideDiff methodology framework figure — v01.

Layout:
  * TOP      — overview in the style of the paper's Figure 3
                (denoising timeline → two green snapshots → blue spectra →
                 π[B] / δ[B] / v_φ panels → ∧ / ∨ gates → DDIM sampler →
                 Spectral Guided Jump → output)
  * MIDDLE   — 5-module per-step loop (Gate → Scheduler → Backbone → Solver → Projection)
  * BOTTOM   — 3 visual panels (A/B/C) colour-linked to modules 1 / 2

Run:
    python figures/stridediff_framework_v01.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch,
                                 Rectangle, Circle, Polygon, PathPatch)
from matplotlib.path import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.linewidth": 0.6,
})

# ------------------------------------------------------------------ #
# palette — softer, more muted pastels
# ------------------------------------------------------------------ #
C_GATE   = "#FDE8A7"   # pale amber
C_SCHED  = "#CFE4BA"   # pale mint
C_NET    = "#CFCBE6"   # pale lavender
C_SOLVER = "#F2BEB9"   # pale blush
C_PROJ   = "#F6D7A8"   # pale peach
C_BG     = "#FAFAFA"
C_BGA    = "#FDF7E3"
C_BGB    = "#EDF3E4"
C_BGC    = "#F5E7E5"
C_ARR    = "#333333"
C_ARR_L  = "#7A7A7A"   # lighter arrow for dashed / ancillary
C_TREND  = "#2F6EB5"
C_SEAS   = "#C94C4C"
C_NEUT   = "#606060"

# --- reference-style overview colours (match paper Figure 3) ------ #
C_GREEN_WAVE  = "#3FA05A"   # time-domain wave
C_BLUE_SPEC   = "#3E7BC1"   # frequency-domain spectra
C_PI_BARS     = ["#7DB7D6", "#A998CE", "#DDA297", "#E9C487"]
C_DELTA_BARS  = ["#F0B48A", "#C9D296", "#D2C69F", "#8FB9D0"]
C_DDIM        = "#D8D8D8"   # sampler box

# ------------------------------------------------------------------ #
# primitives
# ------------------------------------------------------------------ #
def shadow_card(ax, x, y, w, h, color, tag=None,
                title=None, subtitle=None, rs=0.18):
    # soft drop shadow
    ax.add_patch(FancyBboxPatch(
        (x + 0.06, y - 0.06), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rs}",
        facecolor="#000000", edgecolor="none",
        alpha=0.12, zorder=1))
    # main card (thicker border)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rs}",
        facecolor=color, edgecolor="#2F2F2F", lw=1.1, zorder=2))
    # tag circle — placed INSIDE the card at the top-left corner
    if tag is not None:
        ax.add_patch(Circle((x + 0.24, y + h - 0.24), 0.18,
                            facecolor="white", edgecolor="#2F2F2F",
                            lw=0.9, zorder=5))
        ax.text(x + 0.24, y + h - 0.24, str(tag),
                ha="center", va="center",
                fontsize=10.5, fontweight="bold", zorder=6)
    if title:
        ax.text(x + w / 2, y + h * 0.58, title,
                ha="center", va="center",
                fontsize=12.2, fontweight="bold", zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.22, subtitle,
                ha="center", va="center",
                fontsize=10.5, color="#303030", zorder=4)


def pill(ax, x, y, w, h, text, fs=11, fc="white", family="serif"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=fc, edgecolor="#2F2F2F", lw=0.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs, family=family, zorder=3)


def hline_arrow(ax, x0, x1, y, label=None, dashed=False, lw=1.8,
                color=C_ARR, lfs=10.5, offset=0.18):
    """Clean horizontal arrow with a clear head."""
    a = FancyArrowPatch((x0, y), (x1, y),
                        arrowstyle="-|>", mutation_scale=16,
                        lw=lw, color=color,
                        linestyle="--" if dashed else "-",
                        zorder=3)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2, y + offset, label, ha="center", va="bottom",
                fontsize=lfs, style="italic", fontweight="bold",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.14", fc="white",
                          ec="none", alpha=0.92))


def ortho_line(ax, pts, dashed=False, lw=0.9, color=C_ARR,
               head=True, head_size=7):
    """Orthogonal polyline with optional arrow head at the last segment."""
    verts = list(pts)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    path = Path(verts, codes)
    ax.add_patch(PathPatch(path, fill=False, lw=lw,
                           edgecolor=color,
                           linestyle="--" if dashed else "-",
                           zorder=3))
    if head:
        # add arrow head on last segment direction
        p_end, p_prev = verts[-1], verts[-2]
        a = FancyArrowPatch(p_prev, p_end,
                            arrowstyle="-|>", mutation_scale=head_size,
                            lw=lw, color=color, zorder=3)
        ax.add_patch(a)


def mini_wave(ax, cx, cy, color, w=1.25, h=0.45, seed=0, freq=2, noise=0.05):
    rng = np.random.default_rng(seed)
    ax.add_patch(Rectangle((cx - w / 2, cy), w, h,
                           fc="white", ec="#888", lw=0.5, zorder=2))
    t = np.linspace(0, 1, 100)
    sig = (0.16 * np.sin(2 * np.pi * freq * t)
           + 0.09 * np.sin(2 * np.pi * (freq * 2.5) * t + rng.random())
           + noise * rng.standard_normal(100))
    ax.plot(cx - w / 2 + 0.05 + t * (w - 0.10),
            cy + h / 2 + sig, color=color, lw=0.9, zorder=3)


# ================================================================== #
# CANVAS
# ================================================================== #
fig, ax = plt.subplots(figsize=(14.2, 11.0))
ax.set_xlim(0, 20)
ax.set_ylim(-0.6, 14.2)
ax.set_aspect("equal")
ax.axis("off")

# -------- title ---------------------------------------------------- #
ax.text(10, 13.85,
        "StrideDiffusion",
        ha="center", va="center", fontsize=16.5, fontweight="bold")

# ================================================================== #
# (1) TOP — OVERVIEW  (reference: paper Figure 3)
# ================================================================== #
# -- container (very light tint so it reads as a single overview panel)
ax.add_patch(FancyBboxPatch(
    (0.15, 7.70), 19.7, 5.75,
    boxstyle="round,pad=0.03,rounding_size=0.30",
    facecolor="#FCFCFC", edgecolor="#CCCCCC", lw=0.6, zorder=0))
ax.text(0.55, 13.25, "Overview",
        ha="left", fontsize=12, fontweight="bold", zorder=3)

# ---------- (1a) Denoising Process timeline ---------------------- #
# Only 3 icons aligned with the 3 green frames below (x_{t-1}, x_t, x_{t-k})
tl_y = 12.96
sz = 0.30
# horizontal arrow spans from just left of frame 1 to just right of frame 3
ax.annotate("", xy=(17.75, tl_y), xytext=(0.90, tl_y),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#333"),
            zorder=2)
# 3 tile icons
frame_centers = [2.075, 4.975, 17.65]
for x in frame_centers:
    ax.add_patch(Rectangle((x - sz / 2, tl_y - sz / 2), sz, sz,
                            fc="white", ec="#666", lw=0.7, zorder=3,
                            hatch="//"))
# thick solid vertical connectors from each timeline tile → its frame top
frame_top_y = 12.35   # snap_y (11.15) + snap_h (1.20)
for x in frame_centers:
    ax.plot([x, x], [tl_y - sz / 2, frame_top_y],
            color="#333", lw=1.8, solid_capstyle="butt", zorder=2)
ax.text(9.5, tl_y + 0.22,
        "Denoising Process",
        ha="center", va="bottom", fontsize=11.5, style="italic")

# ---------- small drawers --------------------------------------- #
def time_wave_box(ax, x, y, w, h, seed, freq, label_below=None,
                  noise=0.18, structure=False, label_x_frac=0.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="white", ec="#777", lw=0.7, zorder=3))
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 220)
    if structure:
        sig = (0.38 * np.sin(2 * np.pi * freq * t)
               + 0.15 * np.sin(2 * np.pi * freq * 2.3 * t + 0.4)
               + 0.03 * rng.standard_normal(220))
    else:
        sig = (0.18 * np.sin(2 * np.pi * freq * t)
               + 0.12 * np.sin(2 * np.pi * (freq * 1.9) * t + rng.random())
               + noise * rng.standard_normal(220))
    ax.plot(x + 0.1 + t * (w - 0.2),
            y + h / 2 + sig * (h * 0.42), color=C_GREEN_WAVE,
            lw=0.95, zorder=4)
    if label_below:
        ax.text(x + w * label_x_frac, y - 0.22, label_below,
                ha="center", va="top", fontsize=10.5, zorder=4)


def spec_box(ax, x, y, w, h, seed, peak_freq):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="white", ec="#777", lw=0.7, zorder=3))
    rng = np.random.default_rng(seed)
    n = 60
    freqs = np.linspace(0, 1, n)
    spec = np.exp(-freqs * 6) * (1 + 0.3 * rng.random(n))
    spec += 0.4 * np.exp(-((freqs - peak_freq) ** 2) * 400)
    spec /= spec.max()
    for i, v in enumerate(spec):
        ax.add_patch(Rectangle(
            (x + 0.1 + i * (w - 0.2) / n,
             y + 0.12),
            (w - 0.2) / n * 0.85,
            v * (h - 0.25),
            fc=C_BLUE_SPEC, ec="none", alpha=0.85, zorder=4))


def metric_panel(ax, x, y, w, h, kind, title, bar_colors=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="white", ec="#666", lw=0.8, zorder=3))
    ax.text(x + w - 0.12, y + h - 0.14, title,
            ha="right", va="top", fontsize=11.5, zorder=5)
    inner_x = x + 0.22
    inner_y = y + 0.18
    inner_w = w - 0.42
    inner_h = h - 0.42
    ax.annotate("", xy=(inner_x, inner_y + inner_h + 0.15),
                xytext=(inner_x, inner_y-0.039),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555"),
                zorder=4)
    ax.annotate("", xy=(inner_x + inner_w + 0.07, inner_y),
                xytext=(inner_x-0.039, inner_y),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555"),
                zorder=4)
    vals = [0.92, 0.32, 0.18, 0.12] if kind == "bars_desc" \
           else [0.30, 0.55, 0.48, 0.35]
    nb = len(vals)
    bw = inner_w / nb * 0.65
    gap = inner_w / nb * 0.35
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle(
            (inner_x + 0.04 + i * (bw + gap), inner_y + 0.01),
            bw, v * (inner_h - 0.05),
            fc=bar_colors[i], ec="#444", lw=0.4, zorder=5))


def phase_panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="white", ec="#666", lw=0.8, zorder=3))
    ax.text(x + w - 0.12, y + h - 0.14, title,
            ha="right", va="top", fontsize=11.5, zorder=5)
    t = np.linspace(0, 1, 200)
    for cc, amp, frq, off in [("#888",        0.15, 2.0, 0.0),
                              (C_SEAS,        0.18, 7.0, 0.5),
                              (C_GREEN_WAVE,  0.17, 4.0, 1.2)]:
        ax.plot(x + 0.18 + t * (w - 0.36),
                y + h / 2 + amp * np.sin(2 * np.pi * frq * t + off),
                color=cc, lw=0.9, zorder=5)


def gate_circle(ax, cx, cy, symbol):
    ax.add_patch(Circle((cx, cy), 0.22, fc="white", ec="#333",
                        lw=0.9, zorder=5))
    ax.text(cx, cy, symbol, ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=6)


# ---------- (1b) snapshots / spectra  (LEFT column) ------------- #
snap_y = 11.15
snap_h = 1.20
time_wave_box(ax, 0.80, snap_y, 2.55, snap_h, seed=11, freq=8,
              label_below=r"$x_{t-1}$", noise=0.22, label_x_frac=0.40)
time_wave_box(ax, 3.70, snap_y, 2.55, snap_h, seed=22, freq=6,
              label_below=r"$x_{t}$",   noise=0.18, label_x_frac=0.40)

spec_y = 9.35
spec_h = 1.00
spec_box(ax, 0.80, spec_y, 2.55, spec_h, seed=5, peak_freq=0.15)
spec_box(ax, 3.70, spec_y, 2.55, spec_h, seed=9, peak_freq=0.22)

# vertical arrows snapshots → spectra
for cx in (0.80 + 2.55 / 2, 3.70 + 2.55 / 2):
    ax.annotate("", xy=(cx, spec_y + spec_h + 0.03),
                xytext=(cx, snap_y - 0.03),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333"),
                zorder=3)

# ---------- (1c) Three metric panels  (MIDDLE column) ----------- #
panel_x = 7.40
panel_w = 2.40
panel_h = 1.20
py_pi    = 11.25
py_delta = 9.85
py_phi   = 8.45
metric_panel(ax, panel_x, py_pi,    panel_w, panel_h,
             "bars_desc", r"$\pi[B]$",   bar_colors=C_PI_BARS)
metric_panel(ax, panel_x, py_delta, panel_w, panel_h,
             "bars_mix",  r"$\delta[B]$", bar_colors=C_DELTA_BARS)
phase_panel(ax, panel_x, py_phi,    panel_w, panel_h,
            r"$v_{\phi}$")

# ---- Orthogonal routing (vertical/horizontal only) --------------- #
# Spectra feed a common junction on the right, which fans out via a
# vertical trunk to π / δ / v_φ through three horizontal branch arrows.
jx       = 6.75
y_pi_c   = py_pi    + panel_h / 2
y_delta_c = py_delta + panel_h / 2
y_phi_c  = py_phi   + panel_h / 2
spec_cy  = spec_y + spec_h / 2
spec1_cx = 0.80 + 2.55 / 2          # 2.075
spec2_cx = 3.70 + 2.55 / 2          # 4.975
spec2_right_x = 3.70 + 2.57         # 6.25

# 1. horizontal feed from spec 2 right edge → junction
ax.plot([spec2_right_x, jx], [spec_cy, spec_cy],
        color="#333", lw=0.9, zorder=3)
# 2. junction dot
ax.plot(jx, spec_cy, marker="o", markersize=5,
        color="#333", zorder=4)
# 3. vertical trunk spanning all three panel y-levels
ax.plot([jx, jx], [y_phi_c, y_pi_c],
        color="#333", lw=0.9, zorder=3)
# 4. three horizontal branch arrows trunk → π / δ / v_φ
for y_panel in (y_pi_c, y_delta_c, y_phi_c):
    ax.annotate("", xy=(panel_x, y_panel), xytext=(jx-0.03, y_panel),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333"),
                zorder=3)

# ---------- (1d) Gates ∧  ∨  + DDIM Sampler (all orthogonal) ----- #
gate_x  = panel_x + panel_w + 0.70   # 10.50
samp_x  = gate_x + 0.55              # 11.05
samp_y_ = 9.55
samp_w, samp_h = 2.77, 1.50

# Place gates vertically so that gate → sampler is a pure horizontal
# arrow (sampler entry points at 28 % / 72 % of its height).
y_wedge = samp_y_ + samp_h * 0.72    # 10.63
y_vee   = samp_y_ + samp_h * 0.28    # 9.97

gate_circle(ax, gate_x, y_wedge, r"$\wedge$")
gate_circle(ax, gate_x, y_vee,   r"$\vee$")

# Right-side junction column between panels and gates
# offset (0.15) chosen so trunk→∧/∨ arrow length (0.33) matches ∧/∨→sampler
jx_R = panel_x + panel_w + 0.15      # 9.95

# 1. vertical trunk covering all panel / gate y-levels
ax.plot([jx_R, jx_R], [y_phi_c, y_pi_c],
        color="#333", lw=0.9, zorder=3)
# 2. horizontal feeds from each panel (π, δ, v_φ) into the trunk
for y in (y_pi_c, y_delta_c, y_phi_c):
    ax.plot([panel_x + panel_w+0.037, jx_R], [y, y],
            color="#333", lw=0.9, zorder=3)
# 3. horizontal branch arrows trunk → ∧ / ∨
for y_gate in (y_wedge, y_vee):
    ax.annotate("", xy=(gate_x - 0.22, y_gate), xytext=(jx_R, y_gate),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333"),
                zorder=3)

# ---------- (1e) DPM-Solver-2 box  (right of gates) -------------- #
ax.add_patch(FancyBboxPatch(
    (samp_x, samp_y_), samp_w, samp_h,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    fc=C_DDIM, ec="#555", lw=0.8, zorder=4))
ax.text(samp_x + samp_w / 2, samp_y_ + samp_h / 2 + 0.12,
        "DPM-Solver-2",
        ha="center", va="center", fontsize=12.2,
        fontweight="bold", zorder=5)
ax.text(samp_x + samp_w / 2, samp_y_ + samp_h / 2 - 0.22,
        "(DDIM fallback when $k{=}1$)",
        ha="center", va="center", fontsize=9.5,
        style="italic", color="#444", zorder=5)

# gates → sampler  (pure horizontal arrows, no gaps)
for y_gate in (y_wedge, y_vee):
    ax.annotate("", xy=(samp_x, y_gate), xytext=(gate_x + 0.22, y_gate),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#333"),
                zorder=3)

# ---------- (1f) Spectral Guided Jump + output wave (far RIGHT) - #
out_x = 16.10
out_y = snap_y
out_w = 3.10
time_wave_box(ax, out_x, out_y, out_w, snap_h, seed=42, freq=2.0,
              label_below=r"$x_{t-k}$", structure=True)

# Spectral Guided Jump label & big arrow (sampler → output wave)
ax.annotate("", xy=(out_x - 0.03, out_y + snap_h / 2),
            xytext=(samp_x + samp_w + 0.02, samp_y_ + samp_h / 2 + 0.30),
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#444",
                            mutation_scale=16,
                            connectionstyle="arc3,rad=-0.12"),
            zorder=3)
ax.text((out_x + samp_x + samp_w) / 2, out_y + snap_h + 0.18,
        "Spectral Guided Jump",
        ha="center", va="bottom", fontsize=11,
        fontweight="bold", zorder=4)

# caption under overview (concise, with each symbol defined)
ax.text(10, 8.17,
        "Given two consecutive snapshots, StrideDiff reads three per-band indicators — "
        r"(1)energy distribution $\pi[B]$ / (2)log-power drift $\delta[B]$ / (3)phase velocity $v_{\phi}$  ",
        ha="center", fontsize=11, style="italic", color="#444")
ax.text(10, 7.87,
        r"combines them through the  $\wedge$ / $\vee$  gates, "
        "and adaptively chooses the next diffusion stride.",
        ha="center", fontsize=11, style="italic", color="#444")

# ================================================================== #
# (2) MIDDLE — per-step loop  (everything on ONE horizontal row)
# ================================================================== #
# container (positioned so gaps with top overview and bottom panels match)
ax.add_patch(FancyBboxPatch(
    (0.3, 4.35), 19.4, 2.95,
    boxstyle="round,pad=0.03,rounding_size=0.25",
    linewidth=0.6, edgecolor="#444", facecolor=C_BG,
    linestyle=(0, (4, 3)), zorder=0))

# ---- content layout (centered within the 19.4-wide container) -----
# 5 modules × 2.95 wide, 4 inter-module arrows × 0.40 gap, 2 entry/exit
# pills × 1.0 wide, 2 outer arrows × 0.40 gap → total 19.15 → tiny
# horizontal margin on both sides.
mw, mh = 2.95, 1.40
mod_y = 5.17
arr_y = mod_y + mh / 2          # 5.87 — vertical centre of modules

# entry / exit pills (smaller; centred vertically on module row)
pill_w, pill_h = 1.00, 0.72
pill_y = arr_y - pill_h / 2     # 5.51
entry_x = 0.425
exit_x  = 18.575
pill(ax, entry_x, pill_y, pill_w, pill_h, r"$x_t$",     fs=13.5)
pill(ax, exit_x,  pill_y, pill_w, pill_h, r"$x_{t-k}$", fs=13.5)

xsm = [1.825, 5.175, 8.525, 11.875, 15.225]
mods = [
    ("Spectral Gate",        "active-band detection",     C_GATE,   1),
    ("Adaptive Scheduler",   "pick stride $k_t$",         C_SCHED,  2),
    ("Diffusion-TS",         "Trend + Season denoiser",   C_NET,    3),
    ("DPM-Solver-2",         "2nd-order multistep",       C_SOLVER, 4),
    ("Spectral Projection",  "soft-filter update",        C_PROJ,   5),
]
for x, (t, s, c, tag) in zip(xsm, mods):
    shadow_card(ax, x, mod_y, mw, mh, c,
                title=t, subtitle=s, tag=tag)

# horizontal arrows — each 0.40 long, touching module / pill edges tightly
hline_arrow(ax, 1.425, 1.825, arr_y)                              # x_t → M1
hline_arrow(ax, 4.775, 5.175, arr_y, label=r"$\mathcal{A}_t$")    # M1 → M2
hline_arrow(ax, 8.125, 8.525, arr_y, label=r"$k_t$")              # M2 → M3
hline_arrow(ax, 11.475, 11.875, arr_y, label=r"$\hat\epsilon_t$") # M3 → M4
hline_arrow(ax, 14.825, 15.225, arr_y, label=r"$x_{\mathrm{cand}}$")  # M4 → M5
hline_arrow(ax, 18.175, 18.575, arr_y)                            # M5 → x_{t-k}


# history buffer — a small floating annotation BELOW module 4 (DPM-Solver-2)
# M4 sits at x = 11.875..14.825 → centre 13.35
hist_y = 4.50
pill(ax, 12.10, hist_y, 2.50, 0.40,
     r"history  $\hat\epsilon_{t_{\mathrm{prev}}}$",
     fs=10, fc="#F9EAE8")
# dashed connectors (Solver.bottom ↔ history.top)
# --- 调节参数（改这些值即可）-----------------------------------
hist_down_x        = 12.60          # 下行箭头 x
hist_up_x          = 14.10          # 上行箭头 x
hist_down_start_y  = mod_y  -0.02          # 下行箭头 起点 y（默认 = M4 底部 5.30）
hist_down_end_y    = hist_y + 0.41  # 下行箭头 终点 y（= history 顶部 4.90）
hist_up_start_y    = hist_y + 0.42  # 上行箭头 起点 y（= history 顶部 4.90）
hist_up_end_y      = mod_y          # 上行箭头 终点 y（= M4 底部 5.30）
# ---------------------------------------------------------------
ax.annotate("", xy=(hist_down_x, hist_down_end_y),
            xytext=(hist_down_x, hist_down_start_y),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_ARR_L,
                            linestyle="--", mutation_scale=8,
                            shrinkA=0, shrinkB=0),
            zorder=3)
ax.annotate("", xy=(hist_up_x, hist_up_end_y),
            xytext=(hist_up_x, hist_up_start_y),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_ARR_L,
                            linestyle="--", mutation_scale=8,
                            shrinkA=0, shrinkB=0),
            zorder=3)

# feedback loop — orthogonal U-shape along the TOP of the pipeline box,
# from exit pill (top centre) back to entry pill (top centre)
entry_cx = entry_x + pill_w / 2     # 0.925
exit_cx  = exit_x  + pill_w / 2     # 19.075
pill_top = pill_y + pill_h          # 6.23
fb_top = 6.98
ortho_line(ax,
           [(exit_cx,  pill_top),
            (exit_cx,  fb_top),
            (entry_cx, fb_top),
            (entry_cx, pill_top)],
           dashed=True, lw=1.1, color=C_ARR_L, head=True, head_size=9)
ax.text(10.0, fb_top + 0.27,
        r"repeat until  $t = 0$",
        ha="center", fontsize=11, style="italic", color="#444",
        bbox=dict(boxstyle="round,pad=0.18",
                  fc=C_BG, ec="none", alpha=0.95))

# ================================================================== #
# (3) BOTTOM — three illustrative panels
# ================================================================== #
p_w, p_h = 6.40, 3.15
p_y = 0.80
gap = 0.15

# ---- Panel A : Band partition visual  (feeds module ①) ----------- #
px = 0.3
ax.add_patch(FancyBboxPatch(
    (px, p_y), p_w, p_h,
    boxstyle="round,pad=0.02,rounding_size=0.22",
    facecolor=C_BGA, edgecolor="#444", lw=0.8, zorder=1))
# left-edge accent stripe (matches module color)
ax.add_patch(Rectangle((px, p_y + 0.08), 0.12, p_h - 0.16,
                       fc=C_GATE, ec="none", zorder=2))
# module-correspondence badge — moved down, below the title line
ax.add_patch(Circle((px + 0.32, p_y + p_h - 0.55), 0.20,
                    fc=C_GATE, ec="#2F2F2F", lw=0.9, zorder=3))
ax.text(px + 0.32, p_y + p_h - 0.55, "1",
        ha="center", va="center", fontsize=11,
        fontweight="bold", zorder=4)
ax.text(px + p_w / 2, p_y + p_h - 0.26,
        "A.  Temporal Band Partition",
        ha="center", fontsize=12.5, fontweight="bold", zorder=3)
ax.text(px + p_w / 2, p_y + p_h - 0.58,
        r"rFFT bins partitioned into 4 coarse bands",
        ha="center", fontsize=10, style="italic", color="#555", zorder=3)
ax.text(px + p_w / 2, p_y + p_h - 0.86,
        r"$\mathcal{B}_0{=}[0]\ \ \ \mathcal{B}_1{=}[1{:}2]\ \ \ "
        r"\mathcal{B}_2{=}[3{:}5]\ \ \ \mathcal{B}_3{=}[6{:}F{-}1]$",
        ha="center", fontsize=10, style="italic", color="#555", zorder=3)

# waveform
w_ax_x, w_ax_y, w_ax_w, w_ax_h = px + 0.40, p_y + 1.05, p_w - 0.80, 1.05
ax.add_patch(Rectangle((w_ax_x, w_ax_y), w_ax_w, w_ax_h,
                       fc="white", ec="#666", lw=0.5, zorder=2))
t = np.linspace(0, 1, 260)
sig = (0.38 * np.sin(2 * np.pi * 1.0 * t)
       + 0.22 * np.sin(2 * np.pi * 3.0 * t + 0.4)
       + 0.12 * np.sin(2 * np.pi * 7.0 * t + 1.1)
       + 0.05 * np.random.default_rng(0).standard_normal(260))
ax.plot(w_ax_x + t * w_ax_w, w_ax_y + w_ax_h / 2 + sig * 0.82,
        color="#333", lw=0.9, zorder=3)

# colored band strip under the waveform axes
strip_y = p_y + 0.60
cell_w = w_ax_w / 13
bands = [
    (0, 0,  C_TREND,   r"$\mathcal{B}_0$ trend"),
    (1, 2,  C_SCHED,   r"$\mathcal{B}_1$ low"),
    (3, 5,  C_PROJ,    r"$\mathcal{B}_2$ mid"),
    (6, 12, C_SOLVER,  r"$\mathcal{B}_3$ high"),
]
for lo, hi, c, _ in bands:
    xlo = w_ax_x + lo * cell_w
    xhi = w_ax_x + (hi + 1) * cell_w
    ax.add_patch(Rectangle((xlo, strip_y), xhi - xlo, 0.22,
                           fc=c, ec="#444", lw=0.5, alpha=0.9, zorder=3))
# bin tick labels
for i in range(13):
    ax.text(w_ax_x + (i + 0.5) * cell_w, strip_y - 0.05,
            str(i), ha="center", va="top", fontsize=8,
            color="#555", zorder=3)
# legend row
for i, (_, _, c, name) in enumerate(bands):
    ax.add_patch(Rectangle((w_ax_x + i * 1.52, p_y + 0.18),
                           0.22, 0.16, fc=c, ec="#444", lw=0.4, zorder=3))
    ax.text(w_ax_x + i * 1.52 + 0.30, p_y + 0.26, name,
            ha="left", va="center", fontsize=9, zorder=3)

# ---- Panel B : Band-activity heatmap  (output of ① → drives ②) --- #
px = 0.3 + p_w + gap
ax.add_patch(FancyBboxPatch(
    (px, p_y), p_w, p_h,
    boxstyle="round,pad=0.02,rounding_size=0.22",
    facecolor=C_BGB, edgecolor="#444", lw=0.8, zorder=1))
# left-edge accent stripe: gradient from Gate → Scheduler
ax.add_patch(Rectangle((px, p_y + 0.08), 0.12, (p_h - 0.16) / 2,
                       fc=C_SCHED, ec="none", zorder=2))
ax.add_patch(Rectangle((px, p_y + 0.08 + (p_h - 0.16) / 2),
                       0.12, (p_h - 0.16) / 2,
                       fc=C_GATE, ec="none", zorder=2))
# double badge: 1 → 2  (moved DOWN below title, still on the left)
ax.add_patch(Circle((px + 0.32, p_y + p_h - 0.55), 0.20,
                    fc=C_GATE, ec="#2F2F2F", lw=0.9, zorder=3))
ax.text(px + 0.32, p_y + p_h - 0.55, "1",
        ha="center", va="center", fontsize=11,
        fontweight="bold", zorder=4)
ax.text(px + 0.597, p_y + p_h - 0.55, r"$\to$",
        ha="center", va="center", fontsize=12.5, zorder=4)
ax.add_patch(Circle((px + 0.865, p_y + p_h - 0.55), 0.20,
                    fc=C_SCHED, ec="#2F2F2F", lw=0.9, zorder=3))
ax.text(px + 0.84, p_y + p_h - 0.55, "2",
        ha="center", va="center", fontsize=11,
        fontweight="bold", zorder=4)
# centered title
ax.text(px + p_w / 2, p_y + p_h - 0.26,
        "B.  Band-Activity Gate over the Trajectory",
        ha="center", fontsize=12.5, fontweight="bold", zorder=3)
# subtitle moved DOWN
ax.text(px + p_w / 2, p_y + p_h - 0.97,
        r"$b\in\mathcal{A}_t \Leftrightarrow$  "
        r"energy  $\wedge$  ( log-power drift  $\vee$  phase velocity )",
        ha="center", fontsize=10, style="italic", color="#555", zorder=3)

rng = np.random.default_rng(3)
n_t, n_b = 24, 4
heat = np.zeros((n_b, n_t))
for b in range(n_b):
    center = (b + 0.5) / n_b
    for j in range(n_t):
        phase = j / (n_t - 1)
        heat[b, j] = np.exp(-((phase - center) ** 2) * 15) \
                     + 0.12 * rng.random()
heat = heat / heat.max()

hx, hy = px + 0.55, p_y + 0.50
hw, hh = p_w - 1.0, 1.30
for b in range(n_b):
    for j in range(n_t):
        c = [C_TREND, C_SCHED, C_PROJ, C_SOLVER][b]
        alpha = float(np.clip(heat[b, j], 0.08, 1.0))
        ax.add_patch(Rectangle(
            (hx + j * (hw / n_t),
             hy + (n_b - 1 - b) * (hh / n_b)),
            hw / n_t * 0.96,
            hh / n_b * 0.88,
            fc=c, ec="none", alpha=alpha, zorder=3))
# axes labels
ax.plot([hx, hx + hw], [hy] * 2, color="#555", lw=0.6, zorder=3)
ax.plot([hx] * 2, [hy, hy + hh], color="#555", lw=0.6, zorder=3)
# diffusion-step label aligned with Panel A's band-legend row (y = p_y + 0.26)
ax.text(hx + hw / 2, p_y + 0.26,
        r"diffusion step   $t$  :   $T \to 0$",
        ha="center", va="center", fontsize=9.5, zorder=3)
for b, name in enumerate([r"$\mathcal{B}_3$", r"$\mathcal{B}_2$",
                           r"$\mathcal{B}_1$", r"$\mathcal{B}_0$"]):
    ax.text(hx - 0.12, hy + b * (hh / n_b) + (hh / n_b) / 2,
            name, ha="right", va="center", fontsize=9.5, zorder=3)
ax.text(hx + 0.55, hy + hh + 0.10,
        "band", ha="left", va="bottom",
        fontsize=10.5, fontweight="bold", zorder=3)

# ---- Panel C : Adaptive stride  (output of module ②) ------------- #
px = 0.3 + 2 * (p_w + gap)
ax.add_patch(FancyBboxPatch(
    (px, p_y), p_w, p_h,
    boxstyle="round,pad=0.02,rounding_size=0.22",
    facecolor=C_BGC, edgecolor="#444", lw=0.8, zorder=1))
# left-edge accent stripe (matches Scheduler)
ax.add_patch(Rectangle((px, p_y + 0.08), 0.12, p_h - 0.16,
                       fc=C_SCHED, ec="none", zorder=2))
# badge ② — moved DOWN below title line
ax.add_patch(Circle((px + 0.32, p_y + p_h - 0.55), 0.20,
                    fc=C_SCHED, ec="#2F2F2F", lw=0.9, zorder=3))
ax.text(px + 0.32, p_y + p_h - 0.55, "2",
        ha="center", va="center", fontsize=11,
        fontweight="bold", zorder=4)
ax.text(px + p_w / 2, p_y + p_h - 0.26,
        r"C.  Adaptive Jump Stride  $k_t$",
        ha="center", fontsize=12.5, fontweight="bold", zorder=3)
# subtitle moved DOWN
ax.text(px + p_w / 2, p_y + p_h - 0.97,
        "big when quiet  ·  medium for low-freq  ·  micro in the final phase",
        ha="center", fontsize=10, style="italic", color="#555", zorder=3)

sx, sy = px + 0.60, p_y + 0.50
sw, sh = p_w - 1.10, 1.30
ax.add_patch(Rectangle((sx, sy), sw, sh,
                       fc="white", ec="#555", lw=0.6, zorder=2))

n = 140
tt = np.linspace(0, 1, n)
k_vals = np.where(
    tt < 0.28,          30 + 2 * np.sin(tt * 30),
    np.where(tt < 0.58, 20 + 2 * np.sin(tt * 20),
             np.where(tt < 0.82, 6 + 3 * np.sin(tt * 25),
                      1 + 0.5 * np.sin(tt * 40))))
xx = sx + tt * sw
yy = sy + 0.06 + (k_vals / 34) * (sh - 0.15)
ax.fill_between(xx, sy + 0.06, yy, color="#808080", alpha=0.10, zorder=3)
ax.plot(xx, yy, color="#333", lw=1.3, zorder=4)

# phase shading
phases = [(0.00, 0.28, C_SCHED,  "big"),
          (0.28, 0.58, C_PROJ,   "med"),
          (0.58, 0.82, C_GATE,   "mixed"),
          (0.82, 1.00, C_SOLVER, "small")]
for a0, b0, c, lbl in phases:
    ax.add_patch(Rectangle((sx + a0 * sw, sy),
                           (b0 - a0) * sw, sh,
                           fc=c, ec="none", alpha=0.18, zorder=3))
    ax.text(sx + (a0 + b0) / 2 * sw, sy + sh + 0.10,
            lbl, ha="center", va="bottom",
            fontsize=9.5, style="italic", zorder=4)

# diffusion-step label aligned with Panel A's band-legend row (y = p_y + 0.26)
ax.text(sx + sw / 2, p_y + 0.26,
        r"diffusion step   $t$  :   $T \to 0$",
        ha="center", va="center", fontsize=9.5, zorder=3)
ax.text(sx - 0.10, sy + sh / 2, r"stride  $k_t$",
        ha="right", va="center", rotation=90, fontsize=9.5, zorder=3)

# ================================================================== #
# (4) TAKEAWAY STRIP
# ================================================================== #
strip_y = 0.15
strip_h = 0.45
ax.add_patch(FancyBboxPatch(
    (0.3, strip_y), 19.4, strip_h,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor="#F4F4F4", edgecolor="#888", lw=0.5, zorder=1))
notes = [
    ([C_GATE],           "Dynamic phase gate",
     "high-freq sensitivity ramps up in late sampling"),
    ([C_GATE, C_SCHED],  r"Big-signal-first  ($\tau_E{=}0.5$)",
     "suppresses pseudo-active mid/high bands"),
    ([C_SOLVER],         "2nd-order multistep",
     r"$\sim 3\times$ fewer NFEs at matched C-FID"),
    ([C_PROJ],           r"Soft projection  ($\gamma{=}0.2$)",
     "avoids spectral cliffs / divergence"),
]
slot_w = 19.4 / 4
for i, (colors, ttl, desc) in enumerate(notes):
    sx_i = 0.3 + i * slot_w + 0.25
    if len(colors) == 1:
        # single-colour swatch
        ax.add_patch(Rectangle((sx_i, strip_y + 0.14), 0.25, 0.18,
                               fc=colors[0], ec="#2F2F2F", lw=0.4, zorder=2))
    else:
        # split swatch — left half / right half, tiny divider
        half = 0.25 / len(colors)
        for k, c in enumerate(colors):
            ax.add_patch(Rectangle((sx_i + k * half, strip_y + 0.14),
                                   half, 0.18,
                                   fc=c, ec="none", zorder=2))
        # outline around the full swatch
        ax.add_patch(Rectangle((sx_i, strip_y + 0.14), 0.25, 0.18,
                               fc="none", ec="#2F2F2F", lw=0.4, zorder=3))
    ax.text(sx_i + 0.40, strip_y + 0.34, ttl,
            fontsize=10, fontweight="bold", va="center", zorder=3)
    ax.text(sx_i + 0.40, strip_y + 0.12, desc,
            fontsize=9, color="#444", va="center", zorder=3)

# ================================================================== #
# LEGEND
# ================================================================== #
# leg_y = -0.45
# leg_items = [
#     (C_GATE,   "1  Spectral Gate"),
#     (C_SCHED,  "2  Adaptive Scheduler"),
#     (C_NET,    "3  Diffusion-TS Backbone"),
#     (C_SOLVER, "4  DPM-Solver-2"),
#     (C_PROJ,   "5  Spectral Projection"),
# ]
# lx = 1.4
# for c, t in leg_items:
#     ax.add_patch(Rectangle((lx, leg_y), 0.32, 0.20,
#                            facecolor=c, edgecolor="#2F2F2F", lw=0.5))
#     ax.text(lx + 0.42, leg_y + 0.10, t, fontsize=9, va="center")
#     lx += 3.6

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "stridediff_framework_v15.pdf")
png_path = os.path.join(out_dir, "stridediff_framework_v15.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=230, bbox_inches="tight")
print(f"Saved:\n  {pdf_path}\n  {png_path}")
