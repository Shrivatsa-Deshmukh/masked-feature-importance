"""
Feature importance from a masked NPE: score every subset, then read off the
three estimands.

For each subset S of groups, the value is the mean log-probability the masked
NPE assigns to the true theta when it sees only S:

    v(S) = mean_i log q(theta_i | x_i, S)

On a bounded (box) prior, log q is first corrected for flow mass outside the
box: log q - log Z(x, S), with Z estimated by sampling the flow (see
evaluation.log_support_mass) -- the same leakage correction sbi applies.

v(S) + H(prior) is a variational lower bound on I(theta; x_S) (Gibbs'
inequality holds for any proper density q; the support correction makes q a
proper density on the box itself, which can only tighten the bound). Without
the correction, subsets whose posteriors leak more mass outside the box score
lower for reasons unrelated to information, which biases every difference
below. H(prior) cancels from every
importance below, which are therefore differences of mutual-information
estimates, in nats:

    unique_j   = v(all) - v(all \\ {j})     what j adds given everything else
                                            (retrain ablation's estimand)
    marginal_j = v({j}) - v({})             what j carries on its own
    shapley_j  = exact Shapley value of v over all 2^G subsets

Uncertainty is assessed by repeating the whole run over training seeds, which
captures training-data, initialization and fit variability.
"""

from itertools import combinations
from math import lgamma
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from evaluation import logprob_per_point
from masked_npe import masked_nets, set_fixed_mask, set_mask_mode


ESTIMANDS = ("unique", "marginal", "shapley")


def all_subsets(n: int) -> List[Tuple[int, ...]]:
    return [c for k in range(n + 1) for c in combinations(range(n), k)]


def per_sample_logprob(posterior, theta: torch.Tensor, x: torch.Tensor,
                       present: Sequence[str], batch_size: int = 1024, bounds=None,
                       n_support_samples: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    log q(theta_i | x_i, S) for every evaluation point, corrected for posterior
    mass outside a bounded prior (see evaluation.log_support_mass).

    Returns (corrected log-prob, log Z) per point; log Z = 0 when bounds is None.
    """
    set_fixed_mask(posterior, present)
    try:
        return logprob_per_point(posterior, theta, x, batch_size, bounds, n_support_samples)
    finally:
        set_mask_mode(posterior, "full")


def score_all_subsets(posterior, theta: torch.Tensor, x: torch.Tensor,
                      batch_size: int = 1024, verbose: bool = True, bounds=None,
                      n_support_samples: int = 0
                      ) -> Tuple[List[str], Dict[frozenset, np.ndarray], Dict[frozenset, np.ndarray]]:
    """Per-sample corrected log q, and log Z, for every subset.
    Returns (groups, {subset: [n_eval]}, {subset: log Z [n_eval]})."""
    groups = masked_nets(posterior)[0].groups
    subsets = all_subsets(len(groups))
    lp: Dict[frozenset, np.ndarray] = {}
    log_z: Dict[frozenset, np.ndarray] = {}
    for n_done, s in enumerate(subsets):
        lp[frozenset(s)], log_z[frozenset(s)] = per_sample_logprob(
            posterior, theta, x, [groups[i] for i in s], batch_size, bounds, n_support_samples)
        if verbose and (n_done + 1) % 64 == 0:
            print(f"    scored {n_done + 1}/{len(subsets)} subsets")
    return groups, lp, log_z


def _shapley_weights(n: int) -> List[float]:
    log_fact = [lgamma(i + 1) for i in range(n + 1)]
    return [float(np.exp(log_fact[k] + log_fact[n - k - 1] - log_fact[n])) for k in range(n)]


def importance_from_values(v: Dict[frozenset, float], n: int) -> Dict[str, np.ndarray]:
    """unique / marginal / exact Shapley from subset values (any value function)."""
    full, empty = frozenset(range(n)), frozenset()
    unique = np.array([v[full] - v[full - {j}] for j in range(n)])
    marginal = np.array([v[frozenset([j])] - v[empty] for j in range(n)])
    w = _shapley_weights(n)
    shapley = np.zeros(n)
    for j in range(n):
        rest = [i for i in range(n) if i != j]
        shapley[j] = sum(w[k] * (v[frozenset(c) | {j}] - v[frozenset(c)])
                         for k in range(n) for c in combinations(rest, k))
    return {"unique": unique, "marginal": marginal, "shapley": shapley}


def importance_table(groups: List[str], lp: Dict[frozenset, np.ndarray]) -> pd.DataFrame:
    """One row per (feature, estimand) with its value in nats."""
    point = importance_from_values({k: float(v.mean()) for k, v in lp.items()}, len(groups))
    return pd.DataFrame([{"feature": g, "estimand": e, "value": point[e][j]}
                         for e in ESTIMANDS for j, g in enumerate(groups)])


def subset_table(groups: List[str], lp: Dict[frozenset, np.ndarray],
                 prior_entropy: Optional[float] = None,
                 log_z: Optional[Dict[frozenset, np.ndarray]] = None) -> pd.DataFrame:
    """
    One row per subset: its features, mask and size, the log-prob score v(S)
    (support-corrected) with its standard error, the information lower bound if
    H(prior) is known, and -- when a support correction was applied -- the
    uncorrected score and the mean posterior mass inside the prior (1 = none lost).
    """
    rows = []
    for s, arr in lp.items():
        members = sorted(s)
        row = {"features": "|".join(groups[i] for i in members) or "none",
               "mask": "".join("1" if i in s else "0" for i in range(len(groups))),
               "n_features": len(members), "logprob": float(arr.mean()),
               "logprob_se": float(arr.std(ddof=1) / np.sqrt(len(arr)))}
        if log_z is not None:
            row["logprob_uncorrected"] = float((arr + log_z[s]).mean())
            row["in_prior_mass"] = float(np.exp(log_z[s]).mean())
        if prior_entropy is not None:
            row["info_lower_bound"] = row["logprob"] + prior_entropy
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_features", "mask"], ascending=[True, False])
