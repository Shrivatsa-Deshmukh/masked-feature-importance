"""
Configuration: training, masking, evaluation and Jansen-Rit simulation settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Config:
    # ─── Reproducibility ─────────────────────────────────────────────────────
    SEED: int = 0

    # ─── Training data / hyperparameters ─────────────────────────────────────
    NUM_SIMULATIONS: int = 16384
    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 5e-4
    MAX_EPOCHS: int = 500
    STOP_AFTER_EPOCHS: int = 25
    VALIDATION_FRACTION: float = 0.15

    # ─── Neural density estimator (NSF) ──────────────────────────────────────
    HIDDEN_FEATURES: int = 125
    NUM_TRANSFORMS: int = 5

    # ─── Masking (masked_npe.py) ─────────────────────────────────────────────
    # Per training sample: with prob MASK_P_FULL keep every group; else with
    # prob MASK_P_LOO drop exactly one (uniformly chosen); otherwise draw the
    # subset size uniformly on {0..G} and a uniform subset of that size.
    # 0/0 is the plain size-uniform scheme (full set then has prob 1/(G+1)).
    #
    # An importance value's error is the difference in fit quality between the
    # two subsets it compares, so what matters is EQUAL fit across compared
    # subsets, not the best possible full-set fit. Favouring the full set
    # (p_full > 0) improves full-set accuracy but can widen the gap to the
    # leave-one-out subsets and so inflate unique importance.
    MASK_P_FULL: float = 0.0
    MASK_P_LOO: float = 0.0

    # ─── Evaluation ──────────────────────────────────────────────────────────
    N_EVAL_SAMPLES: int = 500          # points for R^2 / coverage (posterior sampling)
    N_EVAL_LOGPROB: int = 2500         # points for subset log-prob scoring (batched)
    N_POSTERIOR_SAMPLES: int = 1000
    CREDIBLE_LEVELS: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95])
    EVAL_SEED: int = 12345             # eval set is independent of the training seed
    # Flow samples per (eval point, subset) used to estimate the posterior mass inside
    # a bounded prior's box, for the support correction of log-probs (Jansen-Rit only;
    # the toy's Gaussian prior is unbounded). 0 disables the correction.
    N_SUPPORT_SAMPLES: int = 100
    R2_SUBSETS: str = "loo"            # "loo" (full + leave-one-out), "full", "none"

    # ─── Forward model (EEG sensor-space projection) ─────────────────────────
    # Off by default. A single source projected to several electrodes gives
    # scaled copies of one signal (feature correlations 0.94-0.999 across
    # channels), and with one channel the projection is a constant gain that
    # the per-feature z-scoring removes. Turning it on requires MNE-Python.
    USE_FORWARD_MODEL: bool = False
    SOURCE_LOCATION: str = 'visual'
    N_CHANNELS: int = 1                # > 1 requires USE_FORWARD_MODEL
    ELECTRODE_NAMES: Optional[List[str]] = None

    # ─── Observation noise (variable SNR) ────────────────────────────────────
    # Off by default: the Jansen-Rit input noise already makes the problem
    # stochastic. Caveat if turned on: the SNR is set relative to the signal's
    # RMS *including* its DC offset (jansen_rit.py), so traces with a large
    # offset get far more noise than their fluctuations warrant, which couples
    # the noise level (and total_log_power) to the DC level. Define the SNR on
    # the mean-removed signal before using this for results.
    ADD_NOISE: bool = False
    NOISE_SNR_MIN: float = 5.0
    NOISE_SNR_MAX: float = 25.0
    NOISE_EXPONENT: float = 1.0
    EVAL_SNR_DB: Optional[float] = None

    # ─── Features ────────────────────────────────────────────────────────────
    # Pure-noise control features (run.py --n_control). Each is one standard-
    # normal number per simulation, drawn independently of theta, so its true
    # importance is exactly 0 under every estimand: a negative control, since a
    # method that gives it clear importance is fitting noise. The simulator
    # appends it as a channel that is constant in time; the feature net passes
    # the value straight through, z-scores it, and treats it as its own mask
    # group (noise_0, noise_1, ...), so it is masked, scored and retrained like
    # any real feature. Each control doubles the number of scored subsets, and
    # it changes the feature set, so write such runs to their own --out folder.
    N_CONTROL_NOISE: int = 0
    # Features removed from every model (references.retrain_ablation removes one
    # more; run.py --exclude adds to this list).
    #
    # signal_mean (the DC level) is excluded by default. Real EEG is recorded
    # AC-coupled / high-pass filtered, so a trace's DC offset is not observable
    # in data. In the simulator it is the model's operating point and is
    # strongly informative (corr ~-0.56 with log_C, ~-0.57 with log_g), so
    # including it would credit the posterior with information no recording
    # could provide and distort every importance ranking it enters. Keep it out
    # unless the study is explicitly about simulator-only quantities.
    EXCLUDED_FEATURES: List[str] = field(default_factory=lambda: ["signal_mean"])

    # ─── Jansen-Rit physics & priors ─────────────────────────────────────────
    DT: float = 0.001
    FS_SIM: int = 1000
    FS_OUT: int = 250
    T_OUTPUT: float = 3.0
    T_DISCARD: float = 2.0
    SIGMA_VALUE: float = 4.5

    PRIOR_MIN: List[float] = field(default_factory=lambda: [np.log(135.0), np.log(120.0), np.log(0.75), np.log(0.5)])
    PRIOR_MAX: List[float] = field(default_factory=lambda: [np.log(270.0), np.log(350.0), np.log(1.25), np.log(2.0)])
    PARAM_NAMES: List[str] = field(default_factory=lambda: ["log_C", "log_mu", "log_kappa", "log_g"])

    @property
    def T_TOTAL(self) -> float:
        return self.T_OUTPUT + self.T_DISCARD

    @property
    def SIGNAL_LENGTH(self) -> int:
        return int(self.T_OUTPUT * self.FS_OUT)

    @property
    def DOWNSAMPLE_FACTOR(self) -> int:
        return self.FS_SIM // self.FS_OUT

    def get_effective_prior_bounds(self) -> Tuple[List[float], List[float], List[str]]:
        return self.PRIOR_MIN.copy(), self.PRIOR_MAX.copy(), self.PARAM_NAMES.copy()
