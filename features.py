"""
Summary-statistic features: computation, per-feature z-scoring with stats from
the raw training signals, feature exclusion and pass-through noise controls.

Outputs are pinned by tests/test_pipeline.py against frozen reference values.

Statistical features (8 when all included):
- skewness, kurtosis
- spectral_slope
- total_log_power, dominant_freq
- Hjorth mobility, complexity
- signal_mean (DC level; excluded by default, see Config.EXCLUDED_FEATURES)
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


class FeatureNet(nn.Module):
    """
    Signal -> z-scored statistical features [batch, n_features].

    Layout is channel-major (channel 0's base features, then channel 1's, ...),
    followed by any pass-through controls, one column each.

    Args:
        input_len: Length of input time series
        fs: Sampling frequency (default: 250 Hz)
        n_fft, hop_length: Welch PSD settings (defaults ~1 Hz resolution)
        excluded_features: Feature (or control) names to drop entirely
        n_channels: Electrodes; each gets its own copy of the base features
        n_control: Pure-noise control channels the simulator appended to x
        base_feature_names: Feature vocabulary (PassThroughFeatureNet supplies the toy's)
    """

    ALL_FEATURE_NAMES = [
        'skewness', 'kurtosis',
        'spectral_slope',
        'total_log_power', 'dominant_freq',
        'hjorth_mobility', 'hjorth_complexity',
        'signal_mean'
    ]

    CONTROL_PREFIX = "noise_"

    def __init__(self, input_len, fs=250, n_fft=256, hop_length=128,
                 excluded_features=None,
                 n_channels: int = 1, n_control: int = 0,
                 base_feature_names: Optional[List[str]] = None):
        super().__init__()

        self.input_len = input_len
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1
        self.n_channels = n_channels
        # Rows of x that carry signal; any pass-through controls follow them.
        self.n_signal_rows = n_channels

        self.excluded_features = set(excluded_features) if excluded_features else set()
        self._all_names = (list(base_feature_names) if base_feature_names is not None
                           else list(self.ALL_FEATURE_NAMES))

        self.n_control_in = n_control
        all_control_names = [f"{self.CONTROL_PREFIX}{i}" for i in range(n_control)]

        valid = self._all_names + all_control_names
        for feat in self.excluded_features:
            if feat not in valid:
                raise ValueError(f"Unknown feature to exclude: '{feat}'. "
                               f"Valid features: {valid}")

        self._included_features = [f for f in self._all_names
                                   if f not in self.excluded_features]
        self.n_base_features = len(self._included_features)
        self._control_idx = [i for i, nm in enumerate(all_control_names)
                             if nm not in self.excluded_features]
        self._included_controls = [all_control_names[i] for i in self._control_idx]
        self.n_control = len(self._control_idx)
        self.n_features = self.n_base_features * self.n_channels + self.n_control

        if 'spectral_slope' in self._included_features:
            freqs = torch.linspace(0, self.fs / 2, self.n_bins)
            idx_1hz = int(1.0 * (self.n_fft / self.fs))
            idx_50hz = int(50.0 * (self.n_fft / self.fs))

            self.slope_idx_start = max(1, idx_1hz)
            self.slope_idx_end = idx_50hz

            slope_freqs = freqs[self.slope_idx_start:self.slope_idx_end]
            self.register_buffer('log_freqs', torch.log(slope_freqs))
            self.register_buffer('log_freqs_mean', torch.mean(torch.log(slope_freqs)))
            self.register_buffer('log_freqs_var', torch.var(torch.log(slope_freqs)))

        # Dummy parameter to satisfy SBI's device check
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.register_buffer('_normalization_set', torch.tensor(False))
        self.register_buffer('feature_means', torch.zeros(self.n_features))
        self.register_buffer('feature_stds', torch.ones(self.n_features))

    @property
    def groups(self) -> List[str]:
        """Maskable units: base features (all channel copies) then controls."""
        return list(self._included_features) + list(self._included_controls)

    def _is_included(self, feature_name):
        """Check if a feature is included (not excluded)."""
        return feature_name not in self.excluded_features

    def compute_normalization_stats(self, signals):
        """
        Compute normalization statistics from training signals.

        Args:
            signals: All training signals [n_samples, input_len] as tensor or numpy array

        This computes all features for each signal and calculates the mean and
        standard deviation for each feature across the full dataset.
        """
        if isinstance(signals, np.ndarray):
            signals = torch.from_numpy(signals).float()
        signals = signals.to(self.dummy_param.device)

        # Batched to bound peak memory over large training sets.
        batch_size = 1024
        all_features = []

        # forward()'s normalization-set check would otherwise block the raw
        # feature computation below, which is how these stats get set.
        self._normalization_set.fill_(True)

        with torch.no_grad():
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i + batch_size]
                features = self._compute_raw_features_multi(batch)
                all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        means = all_features.mean(dim=0)
        stds = all_features.std(dim=0)

        self.feature_means.copy_(means)
        self.feature_stds.copy_(stds)
        self._normalization_set.fill_(True)

    def _base_feature_of(self, index):
        """Base feature name (mask group) for a flat feature index, with the
        channel stripped."""
        span = self.n_base_features * self.n_channels
        if index >= span:
            return self._included_controls[index - span]
        return self._included_features[index % self.n_base_features]

    def compute_welch_psd(self, x):
        """Compute Welch PSD of input signal."""
        x_unfolded = x.unfold(-1, self.n_fft, self.hop_length)
        window = torch.hann_window(self.n_fft, device=x.device).view(1, 1, -1)
        x_windowed = x_unfolded * window
        fft_windows = torch.fft.rfft(x_windowed, dim=-1)
        psd_windows = torch.abs(fft_windows) ** 2
        return psd_windows.mean(dim=1)


    def _compute_raw_features(self, x):
        """
        Compute raw (unnormalized) statistical features from input signal.
        Only computes included features.

        Every divide-by-zero / log-of-zero guard below clamps its denominator
        against `tiny` rather than adding an absolute constant such as
        `std + 1e-6`. Clamping at `tiny` never fires on real data, so every
        feature except total_log_power and signal_mean (scale-dependent by
        construction) is exactly invariant to rescaling, whatever the input
        scale.

        Args:
            x: Input time series [batch_size, input_len]

        Returns:
            Raw statistical features [batch_size, n_features]
        """
        features = []
        # Smallest positive normal of the working dtype (1.2e-38 for float32).
        # Only an exactly-constant signal can reach it, and there the numerator
        # is exactly zero too, so the result is a clean 0 rather than a NaN.
        tiny = torch.finfo(x.dtype).tiny

        # Compute mean and std once, and remove the mean up front.
        mu = x.mean(1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_centered = x - mu

        # Compute PSD once for efficiency, on the *mean-removed* signal. A
        # Jansen-Rit trace carries a real DC offset (median |mean|/std ~0.6,
        # reaching ~17), and compute_welch_psd applies a Hann window, which
        # smears that DC across the low bins rather than confining it to bin
        # 0. Those bins lie inside the 1-50 Hz analysis band, so an uncentred
        # PSD would let the offset masquerade as low-frequency power (pinning
        # many dominant_freq values to the lowest bin). Centring also makes
        # total_log_power a log of the signal's variance rather than of
        # variance-plus-offset.
        psd = self.compute_welch_psd(x_centered)

        # 1. Skewness
        z = x_centered / std.clamp_min(tiny)
        if self._is_included('skewness'):
            skew = (z ** 3).mean(1, True)
            features.append(skew)

        # 2. Kurtosis (excess kurtosis, captures distribution "peakedness"; normal = 0)
        if self._is_included('kurtosis'):
            kurt = (z ** 4).mean(1, True) - 3.0
            features.append(kurt)

        # 3. Spectral slope (1-50Hz)
        if self._is_included('spectral_slope'):
            log_psd = torch.log(psd.clamp_min(tiny)).unsqueeze(1)
            psd_band = log_psd[:, 0, self.slope_idx_start:self.slope_idx_end]
            y_mean = torch.mean(psd_band, dim=1, keepdim=True)
            xy_cov = torch.mean((self.log_freqs - self.log_freqs_mean) * (psd_band - y_mean), dim=1, keepdim=True)
            slope = xy_cov / (self.log_freqs_var + 1e-8)
            features.append(slope)

        # 4. Total log power
        if self._is_included('total_log_power'):
            total_power = psd.sum(dim=1, keepdim=True)
            # Shifts by log(gain**2) under rescaling — a constant offset that
            # feature_means absorbs, which is the intended behaviour here.
            total_log_power = torch.log(total_power.clamp_min(tiny))
            features.append(total_log_power)

        # 5. Dominant frequency (frequency with maximum power in 1-50 Hz range)
        # We restrict to 1-50 Hz to avoid DC artifacts and focus on physiologically
        # relevant frequencies for Jansen-Rit models (which produce alpha-band activity),
        # while staying clear of the anti-alias filter's rolloff near 100 Hz
        if self._is_included('dominant_freq'):
            freqs = torch.linspace(0, self.fs / 2, psd.shape[1], device=psd.device)

            freq_min_hz = 1.0
            freq_max_hz = 50.0
            idx_min = max(1, int(freq_min_hz * (self.n_fft / self.fs)))
            idx_max = min(psd.shape[1], int(freq_max_hz * (self.n_fft / self.fs)))

            psd_band = psd[:, idx_min:idx_max]
            band_freqs = freqs[idx_min:idx_max]
            max_indices_in_band = psd_band.argmax(dim=1)
            dominant_freq = band_freqs[max_indices_in_band].unsqueeze(1)

            # Normalize to [0, 1] range (relative to Nyquist)
            dominant_freq = dominant_freq / (self.fs / 2)
            features.append(dominant_freq)

        # 6-7. Hjorth parameters (mobility and complexity)
        if self._is_included('hjorth_mobility') or self._is_included('hjorth_complexity'):
            dx = torch.diff(x, dim=1)
            ddx = torch.diff(dx, dim=1)

            var_0 = torch.var(x, dim=1, keepdim=True)
            var_1 = torch.var(dx, dim=1, keepdim=True)
            var_2 = torch.var(ddx, dim=1, keepdim=True)

            mobility = torch.sqrt(var_1 / var_0.clamp_min(tiny))

            if self._is_included('hjorth_mobility'):
                features.append(mobility)

            if self._is_included('hjorth_complexity'):
                mobility_dx = torch.sqrt(var_2 / var_1.clamp_min(tiny))
                complexity = mobility_dx / mobility.clamp_min(tiny)
                features.append(complexity)

        # 8. Signal mean (DC level). Excluded by default (Config.EXCLUDED_FEATURES):
        # real EEG is AC-coupled / high-pass filtered, so the DC offset cannot
        # be measured, while in the simulator it is the model's operating point
        # and strongly informative (corr ~-0.56 with log_C, ~-0.57 with log_g).
        # Using it would credit the posterior with information no recording
        # provides. Every other feature here is blind to the offset (skew/kurt
        # subtract it, the PSD is computed mean-removed, Hjorth uses variances),
        # so excluding this one removes the DC from the analysis entirely.
        if self._is_included('signal_mean'):
            features.append(mu)

        return torch.cat(features, dim=1)

    def _compute_raw_features_multi(self, x):
        """
        Compute raw features for single- or multi-channel input.

        For 2D input [batch, input_len] (n_channels==1), delegates directly
        to _compute_raw_features unchanged. For 3D input
        [batch, n_channels, input_len], flattens batch and channel dims
        together so _compute_raw_features's per-signal math (which assumes
        strictly-2D input, e.g. compute_welch_psd's dim=1 window-axis mean)
        never has to handle an extra axis, then reshapes the result back to
        [batch, n_channels * n_base_features] in channel-major order.

        Returns:
            Raw statistical features [batch_size, n_features]
        """
        signal, controls = self._split_controls(x)

        if signal.dim() == 2:
            feats = self._compute_raw_features(signal)
        else:
            batch, n_ch, length = signal.shape
            flat = signal.reshape(batch * n_ch, length)
            f = self._compute_raw_features(flat)  # [batch*n_ch, n_base_features]
            feats = f.reshape(batch, n_ch * f.shape[1])

        if controls is not None:
            feats = torch.cat([feats, controls], dim=1)
        return feats

    def _split_controls(self, x):
        """Separate the real electrode channels from the pass-through control
        channels the simulator appended.

        Controls are constant along time, so one timepoint carries the value.
        An excluded control remains present in x but is dropped here, which is
        what lets a control be ablated exactly like a real feature.

        Returns:
            (signal, controls) where controls is None when there are none.
        """
        if self.n_control_in == 0:
            return x, None
        if x.dim() != 3:
            raise ValueError(
                f"n_control={self.n_control_in} expects x of shape "
                f"[batch, signal rows + n_control, time], got {tuple(x.shape)}"
            )
        expected = self.n_signal_rows + self.n_control_in
        if x.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} rows ({self.n_signal_rows} signal + "
                f"{self.n_control_in} control), got {x.shape[1]}"
            )
        signal = x[:, :self.n_signal_rows, :]
        controls = x[:, self.n_signal_rows:, 0]        # [batch, n_control_in]
        if self.n_control != self.n_control_in:
            controls = controls[:, self._control_idx]  # drop excluded controls
        return signal, controls

    def forward(self, x):
        """z-scored features [batch, n_features]."""
        if not self._normalization_set:
            raise RuntimeError(
                "Normalization statistics have not been set. "
                "Call compute_normalization_stats(signals) with your training data "
                "before running forward pass."
            )
        features = self._compute_raw_features_multi(x)
        return (features - self.feature_means) / self.feature_stds.clamp_min(
            torch.finfo(self.feature_stds.dtype).tiny)


class PassThroughFeatureNet(FeatureNet):
    """
    Toy-pipeline feature net: features arrive precomputed, one row per
    (channel, base feature), constant along time, and are read straight off x.
    z-scoring, exclusion and controls are inherited unchanged from FeatureNet.

    Layout of x (see toy.py):
        [batch, n_base_features * n_channels + n_control, time]
    """

    ALL_FEATURE_NAMES: List[str] = []

    def __init__(self, base_feature_names: List[str], n_channels: int = 1,
                 excluded_features=None, n_control: int = 0):
        super().__init__(
            input_len=1, fs=250, excluded_features=excluded_features,
            n_channels=n_channels, n_control=n_control,
            base_feature_names=base_feature_names,
        )
        self._row_idx = [self._all_names.index(f) for f in self._included_features]
        self.n_rows_per_channel = len(self._all_names)
        # One row per (channel, base feature) rather than one per electrode.
        self.n_signal_rows = self.n_rows_per_channel * n_channels

    def _compute_raw_features_multi(self, x):
        """Read the toy features off x: no statistics, just selection.

        Mirrors the base class's contract exactly -- returns
        [batch, n_base_features * n_channels (+ n_control)] in channel-major
        order -- so normalization and masking are unaffected.
        """
        signal, controls = self._split_controls(x)

        if signal.dim() != 3:
            raise ValueError(
                f"PassThroughFeatureNet expects x of shape [batch, "
                f"{self.n_signal_rows} (+controls), time], "
                f"got {tuple(x.shape)}"
            )
        if signal.shape[1] != self.n_signal_rows:
            raise ValueError(
                f"Expected {self.n_signal_rows} feature rows "
                f"({self.n_rows_per_channel} features x {self.n_channels} channels), "
                f"got {signal.shape[1]}"
            )

        batch = signal.shape[0]
        # Constant along time by construction, so timepoint 0 carries the value.
        flat = signal[:, :, 0].view(batch, self.n_channels, self.n_rows_per_channel)
        feats = flat[:, :, self._row_idx].reshape(batch, self.n_channels * self.n_base_features)

        if controls is not None:
            feats = torch.cat([feats, controls], dim=1)
        return feats
