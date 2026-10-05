"""
Lead-field forward model for EEG simulation.

Projects a single neural source to EEG sensor space using MNE-Python's
fsaverage template head model (no individual MRI required):
- apply():       one auto-selected electrode (highest sensitivity to the source)
- apply_multi(): several electrodes with signed gains (the real topography)

Forward solutions are cached to disk (~/.mne/lead_field_cache).
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
import torch

# MNE imports
import mne
from mne.datasets import fetch_fsaverage

from config import Config


# Default cache directory
CACHE_DIR = Path.home() / '.mne' / 'lead_field_cache'

# Visual cortex source location (MNI coordinates)
NAMED_REGIONS = {
    'visual': [0, -85, 5],  # Primary visual cortex (V1, calcarine sulcus)
}


class LeadFieldModel:
    """
    Lead-field forward model for a single source.

    This class computes the forward solution using MNE-Python's fsaverage
    template and provides a simple interface for projecting source signals
    to EEG sensor space.

    Args:
        source_location: Source location ('visual' or MNI coordinates [x, y, z] in mm)
        subjects_dir: FreeSurfer subjects directory (default: MNE sample data)
        verbose: Whether to print progress messages

    Attributes:
        lead_field_value: Scalar gain value for the selected electrode
        selected_electrode: Name of the auto-selected electrode
        source_vertex: Index of the source vertex in the source space
        source_hemi: Hemisphere of the source ('lh' or 'rh')

    Example:
        >>> lf = LeadFieldModel('visual')
        >>> source_signal = torch.randn(32, 3000)  # Batch of 32, 3s at 1kHz
        >>> eeg = lf.apply(source_signal)
    """

    def __init__(
        self,
        source_location: Union[str, List[float]] = 'visual',
        subjects_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.source_location = source_location

        if isinstance(source_location, str):
            if source_location not in NAMED_REGIONS:
                raise ValueError(
                    f"Unknown region '{source_location}'. "
                    f"Available: {list(NAMED_REGIONS.keys())}"
                )
            self.mni_coords = np.array(NAMED_REGIONS[source_location])
            self._location_name = source_location
        else:
            self.mni_coords = np.array(source_location)
            self._location_name = f"mni_{self.mni_coords[0]}_{self.mni_coords[1]}_{self.mni_coords[2]}"

        cache_key = self._get_cache_key()
        cached = self._load_from_cache(cache_key)

        if cached is not None:
            self._load_cached_data(cached)
            if self.verbose:
                print(f"[LeadFieldModel] Loaded from cache: {self._location_name}")
                print(f"  Electrode: {self.selected_electrode}, Gain: {self.lead_field_value:.6f}")
        else:
            if self.verbose:
                print(f"[LeadFieldModel] Computing forward model for {self._location_name}...")
            self._compute_forward_model(subjects_dir)
            self._save_to_cache(cache_key)
            if self.verbose:
                print(f"  Electrode: {self.selected_electrode}, Gain: {self.lead_field_value:.6f}")

    def _get_cache_key(self) -> str:
        """Generate unique cache key based on source location."""
        loc_str = f"{self.mni_coords[0]:.1f}_{self.mni_coords[1]:.1f}_{self.mni_coords[2]:.1f}"
        return hashlib.md5(loc_str.encode()).hexdigest()[:12]

    def _load_from_cache(self, cache_key: str) -> Optional[dict]:
        """Load cached forward model data."""
        cache_path = CACHE_DIR / f"leadfield_{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str):
        """Save forward model data to cache."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"leadfield_{cache_key}.pkl"

        data = {
            'lead_field_value': self.lead_field_value,
            'selected_electrode': self.selected_electrode,
            'selected_electrode_idx': self.selected_electrode_idx,
            'source_vertex': self.source_vertex,
            'source_hemi': self.source_hemi,
            'mni_coords': self.mni_coords,
            'all_electrode_names': self.all_electrode_names,
            'full_leadfield_column': self.full_leadfield_column,
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_cached_data(self, data: dict):
        """Load data from cache dictionary."""
        self.lead_field_value = data['lead_field_value']
        self.selected_electrode = data['selected_electrode']
        self.selected_electrode_idx = data['selected_electrode_idx']
        self.source_vertex = data['source_vertex']
        self.source_hemi = data['source_hemi']
        self.all_electrode_names = data['all_electrode_names']
        self.full_leadfield_column = data['full_leadfield_column']

    def _compute_forward_model(self, subjects_dir: Optional[str]):
        """Compute forward model using MNE-Python."""
        mne_verbose = False

        fs_dir = fetch_fsaverage(verbose=mne_verbose)
        if subjects_dir is None:
            subjects_dir = os.path.dirname(fs_dir)

        subject = 'fsaverage'

        # BEM solution ships pre-computed for fsaverage.
        bem_path = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        bem = mne.read_bem_solution(bem_path, verbose=mne_verbose)

        # ico4 trades some spatial resolution for a much faster forward solve.
        src = mne.setup_source_space(
            subject, spacing='ico4', subjects_dir=subjects_dir,
            add_dist=False, verbose=mne_verbose
        )

        self.source_vertex, self.source_hemi = self._find_closest_vertex(
            src, self.mni_coords, subjects_dir, subject
        )

        montage = mne.channels.make_standard_montage('standard_1020')
        ch_names = montage.ch_names
        info = mne.create_info(ch_names, sfreq=1000, ch_types='eeg')
        info.set_montage(montage)

        # Note: Forward model uses reference at infinity by default.
        # For single-source/single-electrode analysis, this is acceptable
        # since we only care about the relative sensitivity pattern.

        trans_path = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
        if os.path.exists(trans_path):
            trans = mne.read_trans(trans_path, verbose=mne_verbose)
        else:
            trans = 'fsaverage'  # MNE falls back to its built-in fsaverage transform

        fwd = mne.make_forward_solution(
            info, trans=trans, src=src, bem=bem,
            eeg=True, meg=False, mindist=5.0,
            verbose=mne_verbose
        )

        # Fix dipole orientation normal to cortex, rather than leaving 3 free
        # orientation components per source.
        fwd = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, verbose=mne_verbose
        )

        leadfield = fwd['sol']['data']  # (n_channels, n_sources)

        # The forward solution may have fewer vertices than the source space
        # (e.g. some excluded by mindist), so look up by hemisphere's vertno.
        if self.source_hemi == 'lh':
            vertno = fwd['src'][0]['vertno']
        else:
            vertno = fwd['src'][1]['vertno']

        try:
            vert_idx_in_hemi = np.where(vertno == self.source_vertex)[0][0]
            if self.source_hemi == 'rh':
                vert_idx_in_hemi += fwd['src'][0]['nuse']
        except IndexError:
            # Vertex not included in the final forward solution (e.g. excluded by
            # mindist); fall back to the nearest included vertex, compared in true
            # MNI space via MNE's surface-RAS -> MNI transform (mne.vertex_to_mni).
            if self.verbose:
                print(f"  Source vertex {self.source_vertex} not in forward solution, finding nearest...")
            all_mni = []
            for hi, hemi in enumerate(['lh', 'rh']):
                hemi_src = fwd['src'][hi]
                mni_pos = mne.vertex_to_mni(hemi_src['vertno'], hemis=hi,
                                             subject=subject, subjects_dir=subjects_dir)
                all_mni.append(mni_pos)
            all_mni = np.vstack(all_mni)

            dists = np.linalg.norm(all_mni - self.mni_coords, axis=1)
            vert_idx_in_hemi = np.argmin(dists)

        self.full_leadfield_column = leadfield[:, vert_idx_in_hemi]
        self.all_electrode_names = [ch['ch_name'] for ch in fwd['info']['chs']]

        # Auto-select the electrode with highest sensitivity to this source.
        self.selected_electrode_idx = np.argmax(np.abs(self.full_leadfield_column))
        self.selected_electrode = self.all_electrode_names[self.selected_electrode_idx]
        # Use absolute value to ensure consistent polarity across sources
        self.lead_field_value = np.abs(self.full_leadfield_column[self.selected_electrode_idx])

    def _find_closest_vertex(
        self,
        src,
        mni_coords: np.ndarray,
        subjects_dir: str,
        subject: str
    ) -> Tuple[int, str]:
        """
        Find the closest usable source-space vertex to the given MNI coordinates.

        """
        min_dist = float('inf')
        best_vertex = 0
        best_hemi = 'lh'

        for hi, hemi in enumerate(['lh', 'rh']):
            vertno = src[hi]['vertno']
            mni_pos = mne.vertex_to_mni(vertno, hemis=hi, subject=subject,
                                         subjects_dir=subjects_dir)
            dists = np.linalg.norm(mni_pos - mni_coords, axis=1)
            hemi_min_idx = np.argmin(dists)
            hemi_min_dist = dists[hemi_min_idx]

            if hemi_min_dist < min_dist:
                min_dist = hemi_min_dist
                best_vertex = int(vertno[hemi_min_idx])
                best_hemi = hemi

        if self.verbose:
            print(f"  Found vertex {best_vertex} in {best_hemi} at distance {min_dist:.1f} mm (true MNI space)")
            if min_dist > 10:
                print("  Warning: Distance > 10mm — nearest available source vertex is unusually far from target")

        return best_vertex, best_hemi

    def apply(self, source_signal: torch.Tensor) -> torch.Tensor:
        """
        Apply forward projection to source signal.

        Args:
            source_signal: Source time series [batch_size, time] or [time]

        Returns:
            Projected EEG signal with same shape as input
        """
        return source_signal * self.lead_field_value

    def get_electrode_leadfield(self, electrode_name: str) -> float:
        """Get lead field value for a specific electrode."""
        if electrode_name not in self.all_electrode_names:
            raise ValueError(f"Unknown electrode '{electrode_name}'. "
                           f"Available: {self.all_electrode_names}")
        idx = self.all_electrode_names.index(electrode_name)
        return self.full_leadfield_column[idx]

    def get_top_electrodes(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N electrodes by absolute sensitivity."""
        abs_gains = np.abs(self.full_leadfield_column)
        top_idx = np.argsort(abs_gains)[::-1][:n]
        return [(self.all_electrode_names[i], self.full_leadfield_column[i])
                for i in top_idx]

    def resolve_channels(self, config: Config) -> List[str]:
        """
        Resolve which electrode names to use for (possibly multi-channel)
        projection, based on config.N_CHANNELS / config.ELECTRODE_NAMES.
        """
        n_channels = getattr(config, 'N_CHANNELS', 1)
        electrode_names = getattr(config, 'ELECTRODE_NAMES', None)

        if n_channels == 1 and electrode_names is None:
            return [self.selected_electrode]
        if electrode_names is not None:
            if len(electrode_names) != n_channels:
                raise ValueError(
                    f"ELECTRODE_NAMES has {len(electrode_names)} entries "
                    f"but N_CHANNELS={n_channels}"
                )
            return electrode_names
        return [name for name, _ in self.get_top_electrodes(n_channels)]

    def apply_multi(self, source_signal: torch.Tensor, electrode_names: List[str]) -> torch.Tensor:
        """
        Project a single source signal to multiple electrodes using SIGNED
        gains, preserving the real topographic pattern (sign flips and
        relative magnitudes across the scalp) — unlike apply(), which uses a
        single abs()-valued gain for one auto-selected electrode.

        Args:
            source_signal: Source time series [batch_size, time]
            electrode_names: Names of the electrodes to project to

        Returns:
            Projected EEG signal [batch_size, n_channels, time]
        """
        gains = torch.tensor(
            [self.get_electrode_leadfield(name) for name in electrode_names],
            dtype=source_signal.dtype, device=source_signal.device
        )
        return source_signal.unsqueeze(1) * gains.view(1, -1, 1)

    def __repr__(self) -> str:
        return (f"LeadFieldModel(source={self._location_name}, "
                f"electrode={self.selected_electrode}, "
                f"gain={self.lead_field_value:.6f})")


def clear_cache():
    """Clear all cached forward models."""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")


if __name__ == '__main__':
    print("Testing LeadFieldModel...")

    lf = LeadFieldModel('visual')
    print(f"\n{lf}")
    print(f"Top 5 electrodes: {lf.get_top_electrodes(5)}")

    source = torch.randn(4, 3000)
    eeg = lf.apply(source)
    print(f"\nSource shape: {source.shape}")
    print(f"EEG shape: {eeg.shape}")
    print(f"Gain applied: source max = {source.abs().max():.3f}, eeg max = {eeg.abs().max():.3f}")
