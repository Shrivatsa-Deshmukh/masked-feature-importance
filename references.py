"""
Dedicated-retrain references the masked NPE is checked against.

  train_full_npe   an ordinary full-feature NPE (no masking), trained identically.
                   Measures the cost of amortizing over subsets, and is the
                   baseline that retrain-ablation drops are taken from.
  retrain_ablation for each group, a fresh NPE trained with that feature removed
                   entirely, on the SAME training simulations and init seed
                   as the full reference, scored on the same eval sets. Reports the
                   drop in mean log-prob (nats, the masked NPE's unique estimand)
                   and in R^2.
  retrain_single   for each group, a fresh NPE trained on that feature ALONE.
                   Its mean log-prob minus the prior's mean log-density
                   on the same points is the masked NPE's marginal estimand.

Retraining is a reference, not ground truth. Each retrain is a separate network
with its own training noise, and a drop subtracts two independently noisy
numbers, so single-seed retrain importances can be far off (even negative).
Compare against values averaged over several seeds.
"""

import time

import numpy as np
import torch

from evaluation import mean_logprob, r2_scores
from masked_npe import epochs_trained, train_npe, training_curve


def train_full_npe(problem, theta_train, x_train, seed: int, device, excluded=None):
    """Plain NPE on the problem's features (minus `excluded`). Returns (posterior, info)."""
    net = problem.feature_net(excluded=excluded)
    net.compute_normalization_stats(x_train)
    t0 = time.time()
    posterior, summary = train_npe(net, problem.prior, theta_train, x_train, problem.cfg, seed, device)
    return posterior, {"epochs": epochs_trained(summary), "train_time_s": round(time.time() - t0, 1),
                       "curve": training_curve(summary)}


def score(problem, posterior, theta_lp, x_lp, theta_r2=None, x_r2=None):
    """Mean log-prob (support-corrected; always) and per-parameter R^2 (if an R^2 set is given)."""
    out = {"logprob": mean_logprob(posterior, theta_lp, x_lp, bounds=problem.support_bounds,
                                   n_support_samples=problem.cfg.N_SUPPORT_SAMPLES)}
    if theta_r2 is not None:
        r2, nrmse, coverage_90 = r2_scores(posterior, problem.cfg, theta_r2, x_r2)
        out.update(r2=np.asarray(r2), nrmse=np.asarray(nrmse), coverage_90=coverage_90)
    return out


def retrain_ablation(problem, groups, theta_train, x_train, seed, device, full_scores,
                     theta_lp, x_lp, theta_r2=None, x_r2=None, verbose=True):
    """
    Returns {group: {"logprob_drop", "logprob", ["r2", "r2_drop", "nrmse", "coverage_90"],
                     "epochs", "train_time_s", "curve"}}.
    full_scores: score() output for the full-feature NPE.
    """
    out = {}
    for k, g in enumerate(groups):
        posterior, info = train_full_npe(problem, theta_train, x_train, seed, device, excluded=[g])
        s = score(problem, posterior, theta_lp, x_lp, theta_r2, x_r2)
        res = {"logprob": s["logprob"], "logprob_drop": full_scores["logprob"] - s["logprob"], **info}
        if "r2" in s and "r2" in full_scores:
            res.update(r2=s["r2"], r2_drop=full_scores["r2"] - s["r2"],
                       nrmse=s["nrmse"], coverage_90=s["coverage_90"])
        out[g] = res
        if verbose:
            msg = f"    [{k + 1}/{len(groups)}] retrain without {g:20s} dlogp={res['logprob_drop']:+.3f}"
            if "r2_drop" in res:
                msg += f"  dR2={res['r2_drop'].mean():+.4f}"
            print(msg + f"  ({info['train_time_s']}s)", flush=True)
        del posterior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def retrain_single(problem, groups, theta_train, x_train, seed, device, prior_logprob,
                   theta_lp, x_lp, verbose=True):
    """
    Returns {group: {"marginal", "logprob", "epochs", "train_time_s", "curve"}}.
    prior_logprob: mean log prior density on the log-prob eval points (the
    score of a model that sees no features).
    """
    out = {}
    for k, g in enumerate(groups):
        posterior, info = train_full_npe(problem, theta_train, x_train, seed, device,
                                         excluded=[h for h in groups if h != g])
        lp = score(problem, posterior, theta_lp, x_lp)["logprob"]
        out[g] = {"logprob": lp, "marginal": lp - prior_logprob, **info}
        if verbose:
            print(f"    [{k + 1}/{len(groups)}] train on {g:20s} alone  marginal={lp - prior_logprob:+.3f}"
                  f"  ({info['train_time_s']}s)", flush=True)
        del posterior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
