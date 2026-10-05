"""
The two inference problems, behind one interface, so run.py has a single code
path for both:

    problem.cfg                       Config with estimator + simulator settings
    problem.prior                     sbi-compatible prior on the device
    problem.prior_entropy             H(prior) in nats
    problem.support_bounds            (low, high) of a box prior, or None if unbounded
    problem.feature_net(excluded)     fresh FeatureNet (excluded: list of groups)
    problem.simulate_train(seed)      (theta_train, x_train)
    problem.eval_set(n)               (theta_eval, x_eval) on the device, fixed by cfg.EVAL_SEED
    problem.exact                     toy only: {estimand: {feature: nats}}, else None
    problem.exact_info(present)       toy only: I(theta; x_S) for a list of group names
    problem.exact_logprob(theta, x, present)
                                      toy only: exact log p(theta_i | x_i,S) per point

Jansen-Rit: Sobol-sampled theta and neural-mass simulation (one channel, no
observation noise by default; forward model and noise are optional, see config). Toy: the linear-Gaussian model in toy.py,
whose exact importance values are known.
"""

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import torch

from config import Config
from features import FeatureNet, PassThroughFeatureNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JansenRitProblem:
    name = "jr"

    def __init__(self, cfg: Config):
        from sbi.utils import BoxUniform

        self.cfg = cfg
        if cfg.N_CHANNELS > 1 and not cfg.USE_FORWARD_MODEL:
            raise ValueError("N_CHANNELS > 1 requires USE_FORWARD_MODEL=True")
        self.lead_field = None
        if cfg.USE_FORWARD_MODEL:
            from forward_model import LeadFieldModel     # needs MNE-Python
            self.lead_field = LeadFieldModel(cfg.SOURCE_LOCATION, verbose=False)
        lo, hi, self.param_names = cfg.get_effective_prior_bounds()
        self.prior = BoxUniform(low=torch.tensor(lo, device=DEVICE), high=torch.tensor(hi, device=DEVICE))
        self.prior_entropy = float(np.sum(np.log(np.asarray(hi) - np.asarray(lo))))
        # Box of the bounded prior, for the support correction of log-probs.
        self.support_bounds = (torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32))
        self.exact = None

    def feature_net(self, excluded: Optional[List[str]] = None) -> FeatureNet:
        c = self.cfg
        return FeatureNet(input_len=c.SIGNAL_LENGTH, fs=c.FS_OUT,
                          excluded_features=list(c.EXCLUDED_FEATURES) + list(excluded or []),
                          n_channels=c.N_CHANNELS,
                          n_control=c.N_CONTROL_NOISE)

    def _simulate(self, theta, fixed_snr_db=None):
        from jansen_rit import simulate_jansen_rit
        return simulate_jansen_rit(theta.cpu(), self.cfg, use_gpu=torch.cuda.is_available(),
                                   lead_field_model=self.lead_field, fixed_snr_db=fixed_snr_db)

    def simulate_train(self, seed: int):
        from jansen_rit import generate_sobol_parameters
        lo, hi, _ = self.cfg.get_effective_prior_bounds()
        # jansen_rit.py draws the neural noise from torch and the observation
        # noise (pink noise, per-trial SNR) from numpy, so seed both.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        theta = generate_sobol_parameters(self.cfg.NUM_SIMULATIONS, lo, hi, seed=seed)
        return theta, self._simulate(theta)

    def eval_set(self, n: int):
        """Fixed evaluation set: prior draw + simulation, seeded by EVAL_SEED."""
        s = self.cfg.EVAL_SEED
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        np.random.seed(s)
        theta = self.prior.sample((n,))
        x = self._simulate(theta, fixed_snr_db=self.cfg.EVAL_SNR_DB)
        return theta.to(DEVICE), x.to(DEVICE)

    def exact_info(self, present):
        return None

    def exact_logprob(self, theta, x, present):
        return None


class ToyProblem:
    name = "toy"

    def __init__(self, cfg: Config, n_channels: int = 1, dup_jitter: Optional[float] = None):
        from toy import PARAM_NAMES, ToyConfig, ToyModel

        tcfg = ToyConfig(n_channels=n_channels, n_control=cfg.N_CONTROL_NOISE)
        if dup_jitter is not None:
            tcfg = replace(tcfg, dup_jitter=dup_jitter)
        self.model = ToyModel(tcfg)
        # Exclusions naming Jansen-Rit features (e.g. the default signal_mean)
        # don't apply to the toy; any other unknown name still raises.
        excluded = [f for f in cfg.EXCLUDED_FEATURES if f not in FeatureNet.ALL_FEATURE_NAMES]
        self.cfg = replace(cfg, EXCLUDED_FEATURES=excluded, USE_FORWARD_MODEL=False, ADD_NOISE=False,
                           N_CHANNELS=n_channels, PARAM_NAMES=list(PARAM_NAMES))
        sd = np.asarray(tcfg.theta_sd)
        # Nominal +/-2 sd range, used only to normalize NRMSE.
        self.cfg.PRIOR_MIN, self.cfg.PRIOR_MAX = (-2 * sd).tolist(), (2 * sd).tolist()
        self.param_names = list(PARAM_NAMES)
        sd_t = torch.tensor(sd, dtype=torch.float32, device=DEVICE)
        self.prior = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros_like(sd_t), sd_t), 1)
        d = len(sd)
        self.support_bounds = None   # Gaussian prior: unbounded, no support correction
        self.prior_entropy = 0.5 * float(d * np.log(2 * np.pi * np.e) + np.linalg.slogdet(self.model.Sigma_theta)[1])
        gt = self.model.ground_truth("info")
        names = self.model.importance_names
        self.exact: Dict[str, Dict[str, float]] = {e: dict(zip(names, gt[e].tolist())) for e in gt}

    def feature_net(self, excluded: Optional[List[str]] = None) -> PassThroughFeatureNet:
        m = self.model
        return PassThroughFeatureNet(
            base_feature_names=m.feature_names, n_channels=m.n_channels,
            excluded_features=list(self.cfg.EXCLUDED_FEATURES) + list(excluded or []),
            n_control=m.n_control)

    def simulate_train(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        theta = self.model.sample_theta(self.cfg.NUM_SIMULATIONS, generator=g)
        return theta, self.model.simulate(theta, generator=g)

    def eval_set(self, n: int):
        g = torch.Generator().manual_seed(self.cfg.EVAL_SEED)
        theta = self.model.sample_theta(n, generator=g)
        x = self.model.simulate(theta, generator=g)
        return theta.to(DEVICE), x.to(DEVICE)

    def _base_indices(self, present) -> List[int]:
        """Base-feature indices of the named groups; controls carry nothing."""
        return [self.model.feature_names.index(g) for g in present if g in self.model.feature_names]

    def exact_info(self, present) -> float:
        """I(theta; x_S), the population value."""
        return self.model.info(self._base_indices(present))

    def exact_logprob(self, theta, x, present) -> np.ndarray:
        """Exact log p(theta_i | x_i,S) on the given points."""
        return self.model.log_posterior(theta.cpu().numpy(), x[:, :, 0].cpu().numpy(),
                                        self._base_indices(present))


def make_problem(name: str, cfg: Config, **toy_kwargs):
    if name == "jr":
        return JansenRitProblem(cfg)
    if name == "toy":
        return ToyProblem(cfg, **toy_kwargs)
    raise ValueError(f"Unknown simulator '{name}'")
