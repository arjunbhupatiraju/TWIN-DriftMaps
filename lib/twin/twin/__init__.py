"""TWIN: digital twins of neural recordings for checking encoding claims and planning experiments.

    from twin import BlockTask, PseudoSessions, DriftModel, check, check_sessions, identifiability, plan, advice
"""
from .task import BlockTask, ChoiceModel, PseudoSessions, LABELS, ReversalTask, ForagingTask
from .drift import DriftModel, phase_twin
from .stats import pooled_corr, decode_r2
from .check import check, check_sessions, identifiability, plan, advice, plant
from .report import methods_text, save
from .comm import comm_check, comm_r2, simulate_pair

__version__ = "0.6.0"
