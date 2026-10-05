"""
Jansen-Rit neural mass model simulator, observation noise, and parameter sampling.
"""

from typing import List, Optional

import numpy as np
import torch
from scipy.stats import qmc
from scipy.signal import decimate
from scipy.fft import rfft, irfft, rfftfreq

from config import Config

_PINK_NOISE_CHUNK_SIZE = 131072


def generate_pink_noise(n_samples: int, n_timepoints: int,
                        exponent: float = 1.0,
                        random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate pink (1/f) noise using spectral filtering method.

    Pink noise has a power spectral density proportional to 1/f^exponent,
    which is characteristic of real EEG background activity.

    Internally chunked to bound peak memory (see _PINK_NOISE_CHUNK_SIZE
    above) — output is identical in shape/content to computing it all at once.

    Args:
        n_samples: Number of independent noise traces (batch size)
        n_timepoints: Length of each noise trace
        exponent: Spectral exponent (1.0 = pink noise, 0.0 = white, 2.0 = brown)
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, n_timepoints) with pink noise
    """
    if random_state is not None:
        np.random.seed(random_state)

    if n_samples <= _PINK_NOISE_CHUNK_SIZE:
        return _generate_pink_noise_chunk(n_samples, n_timepoints, exponent)

    # Write chunks directly into a pre-allocated output array rather than
    # collecting a list and concatenating, which would momentarily hold both
    # the full list of chunks AND the concatenated result in memory at once.
    result = np.empty((n_samples, n_timepoints), dtype=np.float32)
    start = 0
    while start < n_samples:
        chunk_n = min(_PINK_NOISE_CHUNK_SIZE, n_samples - start)
        result[start:start + chunk_n] = _generate_pink_noise_chunk(chunk_n, n_timepoints, exponent)
        start += chunk_n
    return result


def _generate_pink_noise_chunk(n_samples: int, n_timepoints: int, exponent: float) -> np.ndarray:
    """Generate one chunk of pink noise (see generate_pink_noise)."""
    white_noise = np.random.randn(n_samples, n_timepoints)
    fft_white = rfft(white_noise, axis=1)

    freqs = rfftfreq(n_timepoints)
    freqs[0] = freqs[1]  # avoid division by zero at DC

    # exponent/2 because we're filtering amplitude (this FFT), not power
    pink_filter = 1.0 / (freqs ** (exponent / 2.0))
    pink_filter = pink_filter / pink_filter.max()

    fft_pink = fft_white * pink_filter
    pink_noise = np.asarray(irfft(fft_pink, n=n_timepoints, axis=1))
    pink_noise = pink_noise / (pink_noise.std(axis=1, keepdims=True) + 1e-8)

    return pink_noise.astype(np.float32)


def sigmoid(v, v0=6.0, r=0.56, v_max=5.0):
    """Sigmoid activation function for Jansen-Rit model."""
    return v_max / (1.0 + torch.exp(r * (v0 - v)))


# Rows x channels simulated per chunk in simulate_jansen_rit (see its
# docstring); bounds peak memory.
_SIMULATION_CHUNK_BASE = 131072


def simulate_jansen_rit(theta_batch: torch.Tensor, config: Optional[Config] = None,
                   use_gpu: bool = False, lead_field_model = None,
                   fixed_snr_db: Optional[float] = None) -> torch.Tensor:
    """
    Jansen-Rit neural mass model simulator with optional forward model.

    Processes theta_batch in chunks of at most
    `_SIMULATION_CHUNK_BASE // N_CHANNELS` rows, so that each chunk's total
    (rows * channels) work, and therefore its peak CPU memory during
    downsampling (scipy.signal.decimate, the dominant cost) and noise
    generation, stays bounded.
    Results are concatenated along the batch dimension; output is identical
    in shape/content to computing it all in one pass.

    Args:
        theta_batch: Parameters in log space [log_C, log_mu, log_kappa, log_g]
        config: Configuration object
        use_gpu: Whether to run simulation on GPU
        lead_field_model: Optional LeadFieldModel for realistic EEG projection
        fixed_snr_db: If provided, use this fixed SNR instead of variable SNR range

    Returns:
        Simulated EEG-like signals
    """
    if config is None:
        config = Config()

    n_channels = getattr(config, 'N_CHANNELS', 1)
    chunk_size = max(1, _SIMULATION_CHUNK_BASE // n_channels)

    if theta_batch.shape[0] <= chunk_size:
        return _simulate_jansen_rit_chunk(theta_batch, config, use_gpu, lead_field_model, fixed_snr_db)

    chunks = []
    for start in range(0, theta_batch.shape[0], chunk_size):
        chunks.append(_simulate_jansen_rit_chunk(
            theta_batch[start:start + chunk_size], config, use_gpu, lead_field_model, fixed_snr_db
        ))
    return torch.cat(chunks, dim=0)


def _simulate_jansen_rit_chunk(theta_batch: torch.Tensor, config: Config,
                   use_gpu: bool, lead_field_model, fixed_snr_db: Optional[float]) -> torch.Tensor:
    """Simulate one chunk of theta_batch — see simulate_jansen_rit."""
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    theta_batch = theta_batch.to(device)
    batch_size = theta_batch.shape[0]

    # Unpack and exponentiate log-space parameters (4 inferred params)
    C = torch.exp(theta_batch[:, 0].unsqueeze(1))
    mu = torch.exp(theta_batch[:, 1].unsqueeze(1))
    kappa = torch.exp(theta_batch[:, 2].unsqueeze(1))
    g = torch.exp(theta_batch[:, 3].unsqueeze(1))
    sigma = torch.full((batch_size, 1), config.SIGMA_VALUE, device=device)

    A = 3.25
    B = 22.0 * g

    a = 100.0 * kappa
    b = 50.0 * kappa

    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    v0, v_max, r_sig = 6.0, 5.0, 0.56

    steps = int(config.T_TOTAL / config.DT)
    discard_steps = int(config.T_DISCARD / config.DT)
    # Store at full simulation rate, then downsample
    output_steps_full = int(config.T_OUTPUT / config.DT)

    a_sq = a ** 2
    b_sq = b ** 2
    Aa = A * a
    Bb = B * b

    y = torch.zeros(batch_size, 6, device=device)
    output_trace = torch.zeros(batch_size, output_steps_full, device=device)

    noise = torch.randn(batch_size, steps, device=device)
    p_input = mu + (noise * sigma)

    # Squeeze tensors for efficient computation
    C1_s, C2_s, C3_s, C4_s = C1.squeeze(), C2.squeeze(), C3.squeeze(), C4.squeeze()
    Aa_s, Bb_s = Aa.squeeze(), Bb.squeeze()
    a_s, b_s = a.squeeze(), b.squeeze()
    a_sq_s, b_sq_s = a_sq.squeeze(), b_sq.squeeze()

    out_idx = 0
    for t in range(steps):
        sig_y1_y2 = sigmoid(y[:, 1] - y[:, 2], v0, r_sig, v_max)
        sig_C1_y0 = sigmoid(C1_s * y[:, 0], v0, r_sig, v_max)
        sig_C3_y0 = sigmoid(C3_s * y[:, 0], v0, r_sig, v_max)

        dy0 = y[:, 3]
        dy3 = Aa_s * sig_y1_y2 - 2 * a_s * y[:, 3] - a_sq_s * y[:, 0]
        dy1 = y[:, 4]
        dy4 = Aa_s * (p_input[:, t] + C2_s * sig_C1_y0) - 2 * a_s * y[:, 4] - a_sq_s * y[:, 1]
        dy2 = y[:, 5]
        dy5 = Bb_s * (C4_s * sig_C3_y0) - 2 * b_s * y[:, 5] - b_sq_s * y[:, 2]

        y[:, 0] += config.DT * dy0
        y[:, 1] += config.DT * dy1
        y[:, 2] += config.DT * dy2
        y[:, 3] += config.DT * dy3
        y[:, 4] += config.DT * dy4
        y[:, 5] += config.DT * dy5

        if t >= discard_steps:
            output_trace[:, out_idx] = y[:, 1] - y[:, 2]
            out_idx += 1

    # Downsample from FS_SIM to FS_OUT; scipy.signal.decimate applies an
    # 8th-order Chebyshev type I filter before decimation to prevent aliasing.
    downsample_factor = config.DOWNSAMPLE_FACTOR
    output_np = output_trace.cpu().numpy()
    downsampled_np = decimate(output_np, downsample_factor, axis=1, ftype='iir', zero_phase=True)
    source_signal = torch.from_numpy(downsampled_np.astype(np.float32))

    if lead_field_model is not None:
        if getattr(config, 'N_CHANNELS', 1) > 1:
            channel_names = lead_field_model.resolve_channels(config)
            eeg = lead_field_model.apply_multi(source_signal, channel_names)  # [batch, n_channels, time]
        else:
            eeg = lead_field_model.apply(source_signal)  # [batch, time]
    else:
        eeg = source_signal

    if config.ADD_NOISE:
        # Operate uniformly in 3D [batch, n_channels, time], squeezing back at
        # the end if the input was 2D — avoids two parallel broadcast paths.
        was_2d = eeg.dim() == 2
        if was_2d:
            eeg = eeg.unsqueeze(1)  # [batch, 1, time]
        n_channels, time_len = eeg.shape[1], eeg.shape[2]

        # Generate pink noise independently per channel
        pink_noise = generate_pink_noise(
            n_samples=batch_size * n_channels,
            n_timepoints=time_len,
            exponent=config.NOISE_EXPONENT
        )
        pink_noise = torch.from_numpy(pink_noise).reshape(batch_size, n_channels, time_len)

        # Calculate signal power (RMS) per sample per channel. Note: this RMS
        # includes the DC offset, so a trace with a large offset gets noise far
        # larger than its fluctuations (see the caveat at Config.ADD_NOISE).
        signal_rms = torch.sqrt(torch.mean(eeg ** 2, dim=-1, keepdim=True))  # [batch, n_channels, 1]

        # Fixed SNR is for evaluation; variable SNR (training mode) teaches the
        # model to handle noise. One SNR value is drawn per trial and shared
        # across that trial's channels.
        if fixed_snr_db is not None:
            snr_db = np.full((batch_size, 1, 1), fixed_snr_db)
        else:
            snr_db = np.random.uniform(
                config.NOISE_SNR_MIN,
                config.NOISE_SNR_MAX,
                size=(batch_size, 1, 1)
            )

        # Convert SNR from dB to linear scale
        # SNR_db = 10 * log10(P_signal / P_noise)
        snr_linear = 10 ** (snr_db / 10.0)
        snr_linear = torch.from_numpy(snr_linear.astype(np.float32))

        # Scale noise to achieve target SNR per sample (broadcasts over channels)
        # noise_rms = signal_rms / sqrt(snr_linear)
        noise_scale = signal_rms / torch.sqrt(snr_linear)
        scaled_noise = pink_noise * noise_scale
        eeg = eeg + scaled_noise

        if was_2d:
            eeg = eeg.squeeze(1)

    # ─── Control (null) channels ─────────────────────────────────────────────
    # Appended AFTER the signal is complete, as constant-along-time channels
    # carrying one standard-normal scalar each. Drawn with no reference to
    # theta, so they are independent of it by construction rather than by
    # approximation -- which is the whole point: a feature computed FROM the
    # signal could not be a valid null, because the signal depends on theta.
    # Drawn once here and frozen into x, so the value is deterministic for a
    # given sample at every epoch and at evaluation. The embedding net slices
    # these off and passes them through as features (see _split_controls).
    n_ctrl = getattr(config, 'N_CONTROL_NOISE', 0)
    if n_ctrl > 0:
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(1)
        time_len = eeg.shape[2]
        ctrl = torch.randn(eeg.shape[0], n_ctrl, 1,
                           dtype=eeg.dtype, device=eeg.device).expand(-1, -1, time_len)
        eeg = torch.cat([eeg, ctrl], dim=1)

    return eeg


def generate_sobol_parameters(n_samples: int, prior_min: List[float],
                              prior_max: List[float], seed: int = 42) -> torch.Tensor:
    """Generate parameters using Sobol sequence for better coverage."""
    sampler = qmc.Sobol(d=len(prior_min), scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_samples)))
    sample_unit = sampler.random_base2(m=m)

    prior_min_np = np.array(prior_min)
    prior_max_np = np.array(prior_max)
    scale = prior_max_np - prior_min_np

    theta_sobol = prior_min_np + sample_unit * scale
    return torch.tensor(theta_sobol[:n_samples], dtype=torch.float32)
