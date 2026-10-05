"""
Toy model: a conjugate linear-Gaussian model with planted feature structure.

theta ~ N(0, Sigma_theta) and s = A theta + eps with Gaussian eps, so for any
feature subset S the posterior is Gaussian and every value function reduces to
a determinant:

    Sigma_post(S) = Sigma_theta
                    - Sigma_theta A_S^T (A_S Sigma_theta A_S^T + Sigma_eps,S)^-1
                      A_S Sigma_theta

That makes three definitions of feature importance exactly computable:

    unique    I(theta; s_j | s_-j)   what j adds given everything else
    marginal  I(theta; s_j)          what j carries on its own
    shapley   average marginal contribution over all 2^d subsets

Two value functions are available:

    info(S)  = 0.5 * log(det Sigma_theta / det Sigma_post(S))   -- nats
    r2(S)    = 1 - diag(Sigma_post(S)) / diag(Sigma_theta)

info is what the masked NPE's log-probability scores estimate.

Planted features
----------------
    f1_strong_t1    strong on theta1, contaminated by a latent eta
    f2_strong_t2    strong and clean on theta2
    f3_dup_a        near-duplicates on theta3 sharing a latent zeta,
    f4_dup_b        separated only by `dup_jitter`
    f5_null         zero loading -- exact null
    f6_suppressor   zero loading, but IS eta: useless alone, large given f1
    f7_weak         small genuine loading on theta4
    f8_bigcoef      large coefficient on the low-prior-variance theta4

The duplicate pair separates `unique` from `marginal` by redundancy, and the
suppressor separates them by conditioning: its marginal importance is exactly
zero while its unique importance is large. f8 distinguishes information from
coefficient magnitude, which z-scoring would otherwise hide.

Channels
--------
n_channels > 1 replicates each base feature per channel with independent
noise, mirroring the multi-channel Jansen-Rit features. The latents eta and zeta are
shared across channels, so channel-averaging cannot remove them. Subsets are
taken at base-feature level, all channel copies in or out together.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


FEATURE_NAMES: List[str] = [
    "f1_strong_t1",
    "f2_strong_t2",
    "f3_dup_a",
    "f4_dup_b",
    "f5_null",
    "f6_suppressor",
    "f7_weak",
    "f8_bigcoef",
]

PARAM_NAMES: List[str] = ["theta1", "theta2", "theta3", "theta4"]


@dataclass
class ToyConfig:
    """Generative constants. `python toy.py` checks the defaults
    keep the planted cases separated in truth."""

    # theta4's sd is small so f8 can carry a large coefficient and still
    # contribute little information.
    theta_sd: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.5)

    # Shared latent contaminants, common to every channel.
    eta_sd: float = 1.5     # contaminates f1; f6 observes it directly
    zeta_sd: float = 1.0    # shared by the duplicate pair

    # Independent per-feature, per-channel noise sd.
    jitter_strong: float = 0.10   # f1 and f6, beyond their shared eta
    noise_clean: float = 0.50     # f2
    dup_jitter: float = 0.30      # f3, f4 -- sweep this to vary redundancy
    noise_null: float = 1.00      # f5
    noise_weak: float = 0.50      # f7
    noise_bigcoef: float = 1.50   # f8

    # Loadings.
    weak_loading: float = 0.60    # f7 on theta4
    bigcoef_loading: float = 3.00  # f8 on theta4

    n_channels: int = 1
    n_control: int = 0            # pure-noise pass-through nulls, as in Config

    def __post_init__(self):
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {self.n_channels}")
        if self.n_control < 0:
            raise ValueError(f"n_control must be >= 0, got {self.n_control}")
        if len(self.theta_sd) != len(PARAM_NAMES):
            raise ValueError(
                f"theta_sd must have {len(PARAM_NAMES)} entries, got {len(self.theta_sd)}")


class ToyModel:
    """Linear-Gaussian generative model plus its exact importance ground truth.

    The generative model, per channel c:

        theta ~ N(0, Sigma_theta)
        s^(c) = A theta + B w + C u^(c)

    with w ~ N(0, I) shared across channels (the latent contaminants) and
    u^(c) ~ N(0, I) drawn independently per channel. Hence

        Cov(s^(c), s^(c')) = A Sigma_theta A^T + B B^T + delta_cc' C C^T
        Cov(theta, s^(c))  = Sigma_theta A^T

    which is all the ground-truth machinery needs.
    """

    def __init__(self, config: Optional[ToyConfig] = None):
        self.config = config or ToyConfig()
        cfg = self.config

        self.feature_names = list(FEATURE_NAMES)
        self.param_names = list(PARAM_NAMES)
        self.n_base_features = len(self.feature_names)
        self.n_params = len(self.param_names)
        self.n_channels = cfg.n_channels
        self.n_control = cfg.n_control

        self.Sigma_theta = np.diag(np.asarray(cfg.theta_sd, dtype=np.float64) ** 2)

        # A: loadings [n_base_features, n_params]
        A = np.zeros((self.n_base_features, self.n_params))
        A[0, 0] = 1.0                      # f1_strong_t1
        A[1, 1] = 1.0                      # f2_strong_t2
        A[2, 2] = 1.0                      # f3_dup_a
        A[3, 2] = 1.0                      # f4_dup_b   (identical row -> duplicate)
        # A[4] = 0                         # f5_null
        # A[5] = 0                         # f6_suppressor (carries no theta)
        A[6, 3] = cfg.weak_loading         # f7_weak
        A[7, 3] = cfg.bigcoef_loading      # f8_bigcoef
        self.A = A

        # B: loadings on the shared latents w = (eta, zeta) [n_base_features, 2]
        B = np.zeros((self.n_base_features, 2))
        B[0, 0] = cfg.eta_sd               # f1 contaminated by eta
        B[5, 0] = cfg.eta_sd               # f6 IS eta -> suppressor
        B[2, 1] = cfg.zeta_sd              # f3 and f4 share zeta
        B[3, 1] = cfg.zeta_sd
        self.B = B

        # C: independent per-feature, per-channel noise sd (diagonal)
        self.c_diag = np.array([
            cfg.jitter_strong,   # f1
            cfg.noise_clean,     # f2
            cfg.dup_jitter,      # f3
            cfg.dup_jitter,      # f4
            cfg.noise_null,      # f5
            cfg.jitter_strong,   # f6
            cfg.noise_weak,      # f7
            cfg.noise_bigcoef,   # f8
        ], dtype=np.float64)

        # Cross-feature covariance pieces, reused by every subset query.
        self._sig = self.A @ self.Sigma_theta @ self.A.T   # signal covariance
        self._shared = self.B @ self.B.T                   # shared-latent covariance
        self._indep = np.diag(self.c_diag ** 2)            # per-channel covariance

    # ---------------------------------------------------------------- simulate

    def sample_theta(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw theta from the Gaussian prior. [n, n_params]"""
        gen_device = generator.device if generator is not None else torch.device("cpu")
        sd = torch.tensor(self.config.theta_sd, dtype=torch.float32, device=gen_device)
        z = torch.randn(n, self.n_params, generator=generator, dtype=torch.float32,
                        device=gen_device)
        return z * sd

    def simulate(self, theta: torch.Tensor,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Simulate observations for given theta.

        Returns x of shape [batch, n_base_features * n_channels + n_control, 1],
        channel-major (channel 0's features, then channel 1's, ...) and constant
        along the trailing time axis -- the layout PassThroughFeatureNet reads,
        and the same encoding the Jansen-Rit pipeline uses for its pure-noise
        control channels.
        """
        theta = theta.to(torch.float32)
        n = theta.shape[0]
        if theta.shape[1] != self.n_params:
            raise ValueError(f"theta must have {self.n_params} columns, got {theta.shape[1]}")

        # torch.Generator is device-bound, so draw on the generator's device and
        # move once at the end rather than mixing devices mid-computation.
        gen_device = generator.device if generator is not None else theta.device
        theta = theta.to(gen_device)
        A = torch.tensor(self.A, dtype=torch.float32, device=gen_device)
        B = torch.tensor(self.B, dtype=torch.float32, device=gen_device)
        c = torch.tensor(self.c_diag, dtype=torch.float32, device=gen_device)

        signal = theta @ A.T                                      # [n, nb]
        w = torch.randn(n, B.shape[1], generator=generator, device=gen_device)
        shared = w @ B.T                                          # [n, nb]

        per_channel = []
        for _ in range(self.n_channels):
            u = torch.randn(n, self.n_base_features, generator=generator,
                            device=gen_device)
            per_channel.append(signal + shared + u * c)
        x = torch.cat(per_channel, dim=1)                         # [n, nb * nch]

        if self.n_control:
            # Independent of theta by construction -- the empirical zero, matching
            # Config.N_CONTROL_NOISE in the real pipeline.
            ctrl = torch.randn(n, self.n_control, generator=generator,
                               device=gen_device)
            x = torch.cat([x, ctrl], dim=1)

        return x.unsqueeze(-1)                                    # [n, rows, 1]

    @property
    def control_names(self) -> List[str]:
        """Pass-through control features, appended after the base features."""
        return [f"noise_{i}" for i in range(self.n_control)]

    @property
    def importance_names(self) -> List[str]:
        """Every feature an importance method should report on.

        Controls are carried here so that all methods cover the same list;
        their exact importance is zero under every estimand (see ground_truth).
        """
        return self.feature_names + self.control_names

    # ------------------------------------------------------------ ground truth

    def _obs_cov(self, subset: Sequence[int]) -> np.ndarray:
        """Covariance of the observed vector for a base-feature subset.

        All channel copies of each selected feature are included, so the block
        is [len(subset) * n_channels] square, ordered channel-major.
        """
        idx = np.asarray(subset, dtype=int)
        nch = self.n_channels
        sig = self._sig[np.ix_(idx, idx)] + self._shared[np.ix_(idx, idx)]
        ind = self._indep[np.ix_(idx, idx)]
        # Every (c, c') block shares sig; only the diagonal blocks add ind.
        cov = np.tile(sig, (nch, nch))
        for c in range(nch):
            sl = slice(c * len(idx), (c + 1) * len(idx))
            cov[sl, sl] += ind
        return cov

    def _cross_cov(self, subset: Sequence[int]) -> np.ndarray:
        """Cov(theta, s_S), [n_params, len(subset) * n_channels]."""
        idx = np.asarray(subset, dtype=int)
        block = self.Sigma_theta @ self.A[idx].T          # [n_params, |S|]
        return np.tile(block, (1, self.n_channels))

    def posterior_cov(self, subset: Sequence[int]) -> np.ndarray:
        """Exact posterior covariance given `subset`; constant in the observed
        value, which is why the R^2 below is exact."""
        subset = list(subset)
        if not subset:
            return self.Sigma_theta.copy()
        Sxx = self._obs_cov(subset)
        Sox = self._cross_cov(subset)
        # Solve rather than invert; Sxx is SPD by construction (every feature
        # carries a strictly positive independent noise term).
        gain = np.linalg.solve(Sxx, Sox.T).T               # [n_params, |S|*nch]
        return self.Sigma_theta - gain @ Sox.T

    def log_posterior(self, theta, x, subset: Sequence[int]) -> np.ndarray:
        """Exact log p(theta_i | x_i,S) for every row, in nats.

        Args:
            theta: [n, n_params] parameter values.
            x: [n, n_base_features * n_channels (+ n_control)] raw observations,
                channel-major as ToyModel.simulate returns them; controls are
                ignored (they carry no information about theta).
            subset: base-feature indices conditioned on (empty: the prior).
        """
        theta = np.asarray(theta, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        subset = list(subset)
        if subset:
            cols = [c * self.n_base_features + j for c in range(self.n_channels) for j in subset]
            # Zero-mean prior and observations: the posterior mean is linear in x.
            mean = np.linalg.solve(self._obs_cov(subset), x[:, cols].T).T @ self._cross_cov(subset).T
        else:
            mean = np.zeros_like(theta)
        chol = np.linalg.cholesky(self.posterior_cov(subset))
        resid = np.linalg.solve(chol, (theta - mean).T)            # whitened, [n_params, n]
        return (-0.5 * (resid ** 2).sum(0) - np.log(np.diag(chol)).sum()
                - 0.5 * self.n_params * np.log(2 * np.pi))

    def info(self, subset: Sequence[int]) -> float:
        """Mutual information I(theta; s_S) in nats."""
        post = self.posterior_cov(subset)
        sign_p, logdet_p = np.linalg.slogdet(post)
        sign_0, logdet_0 = np.linalg.slogdet(self.Sigma_theta)
        if sign_p <= 0 or sign_0 <= 0:
            raise np.linalg.LinAlgError("non-positive-definite covariance")
        return 0.5 * (logdet_0 - logdet_p)

    def r2_per_param(self, subset: Sequence[int]) -> np.ndarray:
        """Exact R^2 of the posterior mean, per parameter.

        The posterior mean is the MMSE estimator and its error variance is the
        posterior variance, so R^2_k = 1 - Sigma_post[k,k] / Sigma_theta[k,k].
        """
        post = self.posterior_cov(subset)
        return 1.0 - np.diag(post) / np.diag(self.Sigma_theta)

    def mean_r2(self, subset: Sequence[int]) -> float:
        return float(np.mean(self.r2_per_param(subset)))

    # -- value functions, cached over all 2^d subsets ------------------------

    def _all_subset_values(self, value_fn) -> Dict[frozenset, float]:
        n = self.n_base_features
        out: Dict[frozenset, float] = {}
        for k in range(n + 1):
            for combo in combinations(range(n), k):
                out[frozenset(combo)] = value_fn(list(combo))
        return out

    def ground_truth(self, value: str = "info") -> "Dict[str, np.ndarray]":
        """
        Exact importance of every base feature under one value function.

        Args:
            value: "info" (nats), "r2" (mean R^2 over parameters), or
                "r2:<param>" for a single parameter's R^2, e.g. "r2:theta3".

        Returns dict of [n_base_features + n_control] arrays:
            unique    v(full) - v(full \\ {j})     -- retrain ablation
            marginal  v({j})  - v({})              -- single-feature retraining
            shapley   exact Shapley value over all 2^d subsets
        """
        if value == "info":
            fn = self.info
        elif value == "r2":
            fn = self.mean_r2
        elif value.startswith("r2:"):
            pname = value.split(":", 1)[1]
            if pname not in self.param_names:
                raise ValueError(f"Unknown parameter '{pname}'; have {self.param_names}")
            k = self.param_names.index(pname)
            fn = lambda S, k=k: float(self.r2_per_param(S)[k])
        else:
            raise ValueError(f"Unknown value function '{value}'")

        # Same estimand definitions as the masked NPE's read-out, applied to the
        # exact value function.
        from importance import importance_from_values
        est = importance_from_values(self._all_subset_values(fn), self.n_base_features)
        # A control carries no theta by construction, so all three estimands are
        # exactly zero for it.
        pad = np.zeros(self.n_control)
        return {e: np.concatenate([est[e], pad]) for e in ("unique", "marginal", "shapley")}

    def ground_truth_table(self, value: str = "info"):
        """Ground truth as a tidy DataFrame, one row per base feature."""
        import pandas as pd
        gt = self.ground_truth(value)
        return pd.DataFrame({
            "feature": self.importance_names,
            "unique": gt["unique"],
            "marginal": gt["marginal"],
            "shapley": gt["shapley"],
        })


def _self_check(verbose: bool = True) -> bool:
    """Validate the closed-form ground truth against Monte Carlo and check the
    planted structure. Run as `python toy.py`."""
    import pandas as pd

    ok = True
    model = ToyModel(ToyConfig())
    rng = torch.Generator().manual_seed(0)
    n_mc = 400_000

    theta = model.sample_theta(n_mc, generator=rng)
    x = model.simulate(theta, generator=rng)[:, :, 0].numpy().astype(np.float64)
    th = theta.numpy().astype(np.float64)

    if verbose:
        print("=" * 74)
        print(f"  Ground-truth self-check  (Monte Carlo n={n_mc:,})")
        print("=" * 74)

    # 1. Analytic posterior covariance vs. empirical least-squares residual
    #    covariance, for a few subsets including the awkward ones.
    subsets = [[0], [5], [0, 5], [2, 3], list(range(8)), [1, 6, 7]]
    if verbose:
        print("\n1. posterior_cov vs Monte Carlo (max abs diff over the matrix)")
    for S in subsets:
        cols = [c * model.n_base_features + j
                for c in range(model.n_channels) for j in S]
        Xs = x[:, cols]
        Xd = np.hstack([Xs, np.ones((len(Xs), 1))])
        beta, *_ = np.linalg.lstsq(Xd, th, rcond=None)
        resid = th - Xd @ beta
        emp = np.cov(resid, rowvar=False)
        ana = model.posterior_cov(S)
        diff = np.abs(emp - ana).max()
        tol = 0.02
        good = diff < tol
        ok &= good
        if verbose:
            names = ",".join(model.feature_names[j].split("_")[0] for j in S)
            label = "S={" + names + "}"
            print(f"   {'PASS' if good else 'FAIL'}  {label:<32} max|emp-analytic|={diff:.5f}")

    # 2. R^2 identity: analytic vs empirical variance explained.
    if verbose:
        print("\n2. r2_per_param vs Monte Carlo")
    for S in ([0, 5], list(range(8))):
        cols = [c * model.n_base_features + j
                for c in range(model.n_channels) for j in S]
        Xd = np.hstack([x[:, cols], np.ones((len(x), 1))])
        beta, *_ = np.linalg.lstsq(Xd, th, rcond=None)
        pred = Xd @ beta
        emp_r2 = 1 - ((th - pred) ** 2).sum(0) / ((th - th.mean(0)) ** 2).sum(0)
        ana_r2 = model.r2_per_param(S)
        diff = np.abs(emp_r2 - ana_r2).max()
        good = diff < 0.01
        ok &= good
        if verbose:
            print(f"   {'PASS' if good else 'FAIL'}  |S|={len(S)}  max|emp-analytic|={diff:.5f}")

    # 3. Shapley efficiency: the values must sum to v(full) - v(empty).
    if verbose:
        print("\n3. Shapley efficiency (sum of values == total value)")
    for value in ("info", "r2"):
        gt = model.ground_truth(value)
        total = (model.info(range(8)) - model.info([])) if value == "info" \
            else (model.mean_r2(range(8)) - model.mean_r2([]))
        diff = abs(gt["shapley"].sum() - total)
        good = diff < 1e-9
        ok &= good
        if verbose:
            print(f"   {'PASS' if good else 'FAIL'}  value={value:5s} "
                  f"sum={gt['shapley'].sum():.6f} total={total:.6f} diff={diff:.2e}")

    # 4. The planted structure must actually be planted.
    if verbose:
        print("\n4. Planted structure")
    gt = model.ground_truth("info")
    names = model.feature_names
    i_null, i_supp = names.index("f5_null"), names.index("f6_suppressor")
    i_a, i_b = names.index("f3_dup_a"), names.index("f4_dup_b")
    i_f2, i_f8 = names.index("f2_strong_t2"), names.index("f8_bigcoef")

    checks = [
        ("f5_null has exactly zero unique",      abs(gt["unique"][i_null]) < 1e-9),
        ("f5_null has exactly zero marginal",    abs(gt["marginal"][i_null]) < 1e-9),
        ("f6_suppressor marginal is exactly 0",  abs(gt["marginal"][i_supp]) < 1e-9),
        # f1 has a larger unique value, since removing it also destroys f6's
        # ability to cancel eta; f6's discriminating property is the gap.
        ("f6_suppressor unique is top-3",        gt["unique"][i_supp] >= np.sort(gt["unique"])[-3]),
        ("f6_suppressor: unique >> marginal",    gt["unique"][i_supp] > 1.0 and gt["marginal"][i_supp] < 1e-9),
        ("duplicates: unique << marginal",       gt["unique"][i_a] < 0.2 * gt["marginal"][i_a]),
        ("duplicates are symmetric",             abs(gt["unique"][i_a] - gt["unique"][i_b]) < 1e-9),
        ("f8 big coefficient < f2 information",  gt["marginal"][i_f8] < gt["marginal"][i_f2]),
    ]
    for label, good in checks:
        ok &= good
        if verbose:
            print(f"   {'PASS' if good else 'FAIL'}  {label}")

    if verbose:
        pd.set_option("display.width", 200)
        print("\n" + "=" * 74)
        print("  Exact ground truth")
        print("=" * 74)
        for value in ("info", "r2"):
            t = model.ground_truth_table(value)
            t["rank_unique"] = t["unique"].rank(ascending=False).astype(int)
            t["rank_marginal"] = t["marginal"].rank(ascending=False).astype(int)
            unit = "nats" if value == "info" else "mean R^2"
            print(f"\n  value function = {value}  ({unit})")
            print(t.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        print("\n  Rank correlation between the estimands themselves "
              "(Spearman, n=8):")
        from scipy.stats import spearmanr
        g = model.ground_truth("info")
        for a, b in (("unique", "marginal"), ("unique", "shapley"), ("marginal", "shapley")):
            print(f"    {a:8s} vs {b:8s}  rho={spearmanr(g[a], g[b]).statistic:+.3f}")
        print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")

    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_check() else 1)
