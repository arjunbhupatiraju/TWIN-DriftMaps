"""Paste-ready methods text and a one-page summary figure for a TWIN check."""
from __future__ import annotations

import json
from pathlib import Path


def methods_text(rep: dict, alpha=0.05):
    n_null, n_twin = rep.get("n_null", 200), rep.get("n_twin", 100)
    d = rep.get("drift", {})
    curve = rep.get("detection_curve") or rep.get("power_curve") or {}
    mdr = rep.get("min_detectable_r")
    sessions = rep.get("sessions")
    lines = [
        f"Encoding of {rep['label']} was assessed with TWIN. The statistic ({rep.get('stat', 'pooled correlation')}) "
        f"was compared against a calibrated null of {n_null} pseudo-sessions drawn from the task's generative process, "
        "which preserves the label's temporal structure but not its alignment with neural activity.",
    ]
    if sessions:
        lines.append(f"Session-level p-values from {sessions} sessions were combined with Fisher's method "
                     f"(combined p = {rep['p_calibrated_combined']:.2g}; a trial-shuffle test gave {rep['p_shuffle_combined']:.2g}).")
        lines.append(f"On drift-matched twin recordings with no task code, the shuffle test was significant in "
                     f"{rep['fp_shuffle_on_twins']:.0%} of runs and the calibrated test in {rep['fp_calibrated_on_twins']:.0%} (nominal {alpha:.0%}).")
    else:
        lines.append(f"The calibrated p-value was {rep['p_calibrated']:.2g} (trial-shuffle test: {rep['p_shuffle']:.2g}). "
                     f"On {n_twin} twin recordings with the same drift and inter-neuron correlation but no task code "
                     f"(multivariate phase randomization), the shuffle test was significant in {rep['drift_only_rate']:.0%} "
                     f"and the calibrated test in {rep['calibrated_rate_on_twins']:.0%} of runs (nominal {alpha:.0%}).")
        if d:
            lines.append(f"Fitted drift: {d['slow_frac']:.0%} of single-neuron variance slow (timescale about "
                         f"{1 / (1 - d['phi']):.0f} trials), {d['shared']:.0%} of it shared across neurons.")
    if curve:
        pts = ", ".join(f"r = {k:g}: {v:.0%}" for k, v in curve.items())
        lines.append(f"Positive control: codes planted in the recorded neurons were detected at rates {pts}"
                     + (f"; the minimum detectable per-neuron correlation (80% power) was {mdr:g}." if mdr else "."))
    lines.append(f"Verdict: {rep['verdict']}.")
    return " ".join(lines)


def figure(rep: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    if "drift_only_rate" in rep:
        fp = [rep["drift_only_rate"], rep["calibrated_rate_on_twins"]]
    else:
        fp = [rep["fp_shuffle_on_twins"], rep["fp_calibrated_on_twins"]]
    ax[0].bar(["shuffle test", "calibrated test"], fp, color=["#b4553a", "#2d6a8a"])
    ax[0].axhline(0.05, ls="--", c="0.4", lw=1)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("false-positive rate on drift-only twins")
    curve = rep.get("detection_curve") or rep.get("power_curve") or {}
    if curve:
        ks = sorted(curve); ax[1].plot(ks, [curve[k] for k in ks], "o-", c="#2d6a8a")
        ax[1].axhline(0.8, ls="--", c="0.4", lw=1)
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("planted per-neuron correlation"); ax[1].set_ylabel("detection rate")
    fig.suptitle(f"TWIN check: {rep['label']} - {rep['verdict'].split(':')[0]}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save(rep: dict, out_dir, name="twin_check"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / f"{name}_methods.txt").write_text(methods_text(rep) + "\n")
    figure(rep, out / f"{name}.png")
    return out
