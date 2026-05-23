from __future__ import annotations
from .params import PNPParams


def gap_schedule(p: PNPParams, step: int) -> float:
    """
    Piecewise protocol:
        pre-equilibrate -> approach -> hold -> separate -> relax

    Returns the instantaneous gap h(t).
    """
    # 1) Pre-equilibration
    if step < p.n_pre:
        return p.h_gap0

    step2 = step - p.n_pre

    # 2) Approach: h_gap0 -> h_min
    if step2 < p.n_approach:
        frac = step2 / max(1, p.n_approach - 1)
        return p.h_gap0 + (p.h_min - p.h_gap0) * frac

    step2 -= p.n_approach

    # 3) Hold at minimum gap
    if step2 < p.n_hold:
        return p.h_min

    step2 -= p.n_hold

    # 4) Separate: h_min -> h_gap0
    if step2 < p.n_separate:
        frac = step2 / max(1, p.n_separate - 1)
        return p.h_min + (p.h_gap0 - p.h_min) * frac

    step2 -= p.n_separate

    # 5) Final relaxation at initial gap
    if step2 < p.n_relax:
        return p.h_gap0

    # Fallback: if called beyond nominal protocol, stay at relaxed geometry
    return p.h_gap0