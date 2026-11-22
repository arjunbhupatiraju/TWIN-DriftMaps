"""Mechanical decision rules. These exact functions are what the pre-registration refers to.

Module 2, NI LEADS state switches. Claim SUPPORTED only if ALL hold (session-bootstrap 95% CIs):
  L1  NI's switch-triggered change is centered before the behavioral switch:
      onset_NI (relative to behavior) < 0, CI upper bound < 0
  L2  NI changes before LC:     onset_NI - onset_LC < 0,    CI upper bound < 0
  L3  NI changes before pupil:  onset_NI - onset_pupil < 0, CI upper bound < 0
  L4  convergent, model-free:   cross-correlation asymmetry of NI > 0, CI lower bound > 0
  If LC has too few units for the comparison, L2 is replaced by L2' (NI before pupil only)
  and the claim is labelled "beyond pupil only" in the paper.

Module 3, NI FORECASTS lapses beyond behavior + LC + pupil (primary horizon: the easy trial
HORIZON = 5 trials ahead of the pre-stimulus activity; same-trial h = 0 reported as secondary):
  SUPPORTED if held-out AUC(behavior + NI + LC + pupil) - AUC(behavior + LC + pupil) > 0,
  CI lower bound > 0 (leave-one-session-out; session bootstrap).
"""
from __future__ import annotations

import numpy as np

HORIZON = 5

# What the synthetic validation must reproduce.
# The LEAD rule is the specific claim: it must be SUPPORTED only when NI truly leads.
# The LAPSE rule is a usefulness claim: NI must add beyond LC + pupil when it leads, and must
# NOT add when it carries nothing (noise) or nothing LC does not already carry (null). In
# lc_leads, NI is the only clean readout of the CURRENT state, which legitimately predicts
# lapses 5 trials later, so either answer is correct there (None = not scored).
# Consequence for the paper: a lapse SUPPORTED without a lead SUPPORTED is NOT evidence of a lead.
# (Exploration before this rule was fixed: at h = 0 a leading NI added ~nothing beyond a
# concurrent LC readout, so same-trial prediction is reported as secondary only.)
EXPECTED = {
    "ni_leads": dict(lead="SUPPORTED", lapse="SUPPORTED"),
    "lc_leads": dict(lead="NOT SUPPORTED", lapse=None),
    "null": dict(lead="NOT SUPPORTED", lapse="NOT SUPPORTED"),
    "noise": dict(lead="NOT SUPPORTED", lapse="NOT SUPPORTED"),
}


def _below0(d):
    return np.isfinite(d["hi"]) and d["est"] < 0 and d["hi"] < 0


def _above0(d):
    return np.isfinite(d["lo"]) and d["est"] > 0 and d["lo"] > 0


def lead_verdict(ll, have_lc=True):
    c = dict(L1=_below0(ll["onset_NI"]),
             L2=_below0(ll["onset_NI-LC"]) if have_lc else None,
             L3=_below0(ll["onset_NI-pupil"]),
             L4=_above0(ll["asym_NI"]))
    ok = all(v for v in c.values() if v is not None)
    return dict(criteria=c, verdict="SUPPORTED" if ok else "NOT SUPPORTED",
                scope="beyond LC and pupil" if have_lc else "beyond pupil only")


def lapse_verdict(delta):
    d = delta["NI_beyond_LC_pupil"]
    return dict(delta=d, verdict="SUPPORTED" if _above0(d) else "NOT SUPPORTED")
