"""
Evaluation of a trained posterior: batched log-probabilities (with the
support correction for a bounded prior), R^2 / NRMSE / coverage from posterior
samples, and agreement between two sets of importance values.
"""

import logging
import signal
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import Config


class PosteriorSamplingTimeout(Exception):
    """Raised when posterior.sample() exceeds POSTERIOR_SAMPLE_TIMEOUT_S."""


POSTERIOR_SAMPLE_TIMEOUT_S = 60


@contextmanager
def _sampling_timeout(seconds: int):
    """
    Abort a single posterior.sample() call that's stuck in sbi's rejection
    sampler (see samplers/rejection/rejection.py: `while num_remaining > 0`
    has no cap and can spin forever if the trained flow's acceptance rate
    against the prior support is ~0% for a given observation). SIGALRM fires
    between the loop's proposal-batch calls, so this reliably interrupts it
    without corrupting torch/CUDA state. Main-thread/Unix only.
    """
    def _on_alarm(signum, frame):
        raise PosteriorSamplingTimeout(f"posterior.sample() exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


# ─── Log-probabilities ───────────────────────────────────────────────────────

def log_support_mass(estimator, x, bounds, n_samples: int,
                     max_rows: int = 262_144) -> np.ndarray:
    """
    log Z_i = log P_q(theta in prior box | x_i) for every row of x, by sampling the flow.

    NPE's flow is unbounded, so with a bounded (box) prior some of its mass can
    fall outside the prior's support. The posterior restricted to the support
    is q / Z, so log q - log Z is the properly normalized density there. This
    is the same leakage correction sbi applies in posterior.log_prob, done in
    batches. Z is estimated as (inside + 1) / (n_samples + 1), which keeps log Z
    finite when no sample lands inside.

    Args:
        bounds: (low, high) tensors of the box, or None for an unbounded prior
            (then Z = 1 and zeros are returned without sampling).
        max_rows: cap on n_samples * points per sampling call (GPU memory).
    """
    if bounds is None or n_samples <= 0:
        return np.zeros(len(x))
    low, high = (b.to(x.device) for b in bounds)
    chunk = max(1, max_rows // n_samples)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            s = estimator.sample((n_samples,), condition=x[i:i + chunk])   # [M, chunk, d]
            inside = ((s >= low) & (s <= high)).all(dim=-1).float()        # [M, chunk]
            z = (inside.sum(dim=0) + 1.0) / (n_samples + 1.0)
            out.append(torch.log(z).double().cpu().numpy())
    return np.concatenate(out)


def logprob_per_point(posterior, theta, x, batch_size: int = 1024, bounds=None,
                      n_support_samples: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    log q(theta_i | x_i) for every evaluation point, batched, corrected for
    posterior mass outside a bounded prior (see log_support_mass). Uses
    whatever mask is currently set on a masked NPE.

    Returns (corrected log-prob, log Z) per point; log Z = 0 when bounds is None.
    """
    from masked_npe import resolve_estimator
    est = resolve_estimator(posterior)
    lps = []
    with torch.no_grad():
        for i in range(0, len(theta), batch_size):
            lp = est.log_prob(theta[i:i + batch_size].unsqueeze(0), condition=x[i:i + batch_size])
            lps.append(lp.reshape(-1).double().cpu().numpy())
    log_z = log_support_mass(est, x, bounds, n_support_samples)
    return np.concatenate(lps) - log_z, log_z


def mean_logprob(posterior, theta, x, bounds=None, n_support_samples: int = 0) -> float:
    """Mean support-corrected log q(theta_i | x_i) over an evaluation set."""
    lp, _ = logprob_per_point(posterior, theta, x, bounds=bounds, n_support_samples=n_support_samples)
    return float(lp.mean())


# ─── R^2 / NRMSE / coverage from posterior samples ──────────────────────────

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute R² score per parameter."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray,
                  prior_min: List[float], prior_max: List[float]) -> np.ndarray:
    """Compute Normalized RMSE per parameter."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    prior_range = np.array(prior_max) - np.array(prior_min)
    return rmse / prior_range


def evaluate_posterior(posterior, config: Config, theta_test, x_test
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    R², NRMSE and credible-interval coverage per parameter on a fixed
    evaluation set, using the posterior mean of N_POSTERIOR_SAMPLES samples.

    Returns r2, nrmse, {credible level: coverage}.
    """
    all_theta_true, all_theta_pred = [], []
    all_coverage = {level: [] for level in config.CREDIBLE_LEVELS}
    n_skipped = 0

    for i in range(len(theta_test)):
        theta_true_np = theta_test[i].cpu().numpy()
        x_obs = x_test[i:i+1]

        try:
            with _sampling_timeout(POSTERIOR_SAMPLE_TIMEOUT_S):
                samples = posterior.sample((config.N_POSTERIOR_SAMPLES,), x=x_obs, show_progress_bars=False)
        except PosteriorSamplingTimeout:
            # Trained flow's acceptance rate against the prior support collapsed
            # to ~0% for this observation (sbi's rejection sampler has no cap —
            # see _sampling_timeout docstring). Drop this eval point rather than
            # stalling the whole run; a handful of skips out of N_EVAL_SAMPLES
            # doesn't meaningfully bias R²/NRMSE.
            n_skipped += 1
            logging.warning(f"  Skipped eval point {i}: posterior sampling timed out after {POSTERIOR_SAMPLE_TIMEOUT_S}s (near-zero acceptance rate)")
            continue

        samples_np = samples.cpu().numpy()

        all_theta_true.append(theta_true_np)
        all_theta_pred.append(np.mean(samples_np, axis=0))

        for level in config.CREDIBLE_LEVELS:
            alpha = (1 - level) / 2
            lower = np.percentile(samples_np, alpha * 100, axis=0)
            upper = np.percentile(samples_np, (1 - alpha) * 100, axis=0)
            in_ci = (theta_true_np >= lower) & (theta_true_np <= upper)
            all_coverage[level].append(in_ci)

    if n_skipped > 0:
        print(f"  WARNING: {n_skipped}/{len(theta_test)} eval points skipped due to posterior sampling timeout")

    theta_true_arr = np.array(all_theta_true)
    theta_pred_arr = np.array(all_theta_pred)
    prior_min, prior_max, _ = config.get_effective_prior_bounds()

    r2 = compute_r2(theta_true_arr, theta_pred_arr)
    nrmse = compute_nrmse(theta_true_arr, theta_pred_arr, prior_min, prior_max)
    coverage_dict = {level: np.mean(all_coverage[level], axis=0) for level in config.CREDIBLE_LEVELS}
    return r2, nrmse, coverage_dict


def r2_scores(posterior, config: Config, theta_test, x_test,
              present: Optional[Sequence[str]] = None):
    """R^2 / NRMSE / ~90% coverage per parameter.

    For a masked NPE, `present` selects the visible groups (None: as currently
    set). Note: evaluate_posterior drops any eval point whose sampling times
    out, and different subsets can drop different points, so an R^2 drop may
    compare slightly different point sets. Timeouts are rare and printed when
    they occur.
    """
    from masked_npe import set_fixed_mask, set_mask_mode
    if present is not None:
        set_fixed_mask(posterior, present)
    try:
        r2, nrmse, coverage = evaluate_posterior(posterior, config, theta_test, x_test)
    finally:
        if present is not None:
            set_mask_mode(posterior, "full")
    level = min(config.CREDIBLE_LEVELS, key=lambda lv: abs(lv - 0.9))
    return r2, nrmse, np.asarray(coverage[level])   # already averaged over eval points


# ─── Agreement ───────────────────────────────────────────────────────────────

def compare_to_reference(estimate: Dict[str, float], reference: Dict[str, float]) -> Dict[str, float]:
    """Agreement between two {feature: value} maps on their shared features."""
    from scipy.stats import pearsonr, spearmanr
    feats = [f for f in estimate if f in reference]
    a = np.array([estimate[f] for f in feats])
    b = np.array([reference[f] for f in feats])
    return {"n": len(feats),
            "spearman": float(spearmanr(a, b).statistic),
            "pearson": float(pearsonr(a, b).statistic),
            "mae": float(np.mean(np.abs(a - b)))}
