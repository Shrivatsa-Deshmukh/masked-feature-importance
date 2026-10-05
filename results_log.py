"""
Results logging. (Not named logging.py: that would shadow the stdlib module
evaluation.py imports.)

Each experiment gets its own folder, results/<name>/, keyed on the training seed:

  config.json             every setting of the experiment (identical for all seeds)
  seeds.csv               one row per seed: fit and agreement metrics
  importance.csv          one row per (seed, method, feature, estimand)
  subset_scores.csv       one row per (seed, feature subset): log-prob score
  parameter_recovery.csv  one row per (seed, model, left-out feature): R^2, NRMSE, coverage
  training_curves.csv     one row per (seed, model, epoch): training and validation loss

Writing a seed replaces any earlier rows for that seed, so a rerun never
duplicates results. A folder only ever holds one configuration: writing a run
whose settings differ from config.json raises an error (use another folder).

Writes are crash-safe (write to temp, atomic rename) and safe under concurrent
writers (exclusive file lock; POSIX only).
"""

import contextlib
import dataclasses
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
TABLES = ("seeds", "importance", "subset_scores", "parameter_recovery", "training_curves")

# Free-text cells such as "NA" or "None" must round-trip as text, not NaN.
READ_KWARGS = dict(keep_default_na=False, na_values=[""], float_precision="round_trip")


def flatten(value: Any) -> Any:
    """Make a value CSV-cell-safe: scalars pass through, containers become JSON."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def config_dict(cfg) -> Dict[str, Any]:
    """The experiment's settings: every Config field except the per-run SEED."""
    return {k: v for k, v in dataclasses.asdict(cfg).items() if k != "SEED"}


def _same(a: Any, b: Any) -> bool:
    """Equality that tolerates float round-off and int/float spelling."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


@contextlib.contextmanager
def _locked(folder: Path):
    with open(folder / ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_seed(name: str, cfg, seed: int, tables: Dict[str, List[Dict[str, Any]]]) -> Path:
    """Store one seed's rows in results/<name>/, replacing that seed's earlier rows."""
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise ValueError(f"Unknown tables {sorted(unknown)}; have {TABLES}")
    folder = RESULTS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    with _locked(folder):
        settings = config_dict(cfg)
        cfg_path = folder / "config.json"
        if cfg_path.exists():
            stored = json.loads(cfg_path.read_text())
            diff = sorted(k for k in set(stored) | set(settings)
                          if not _same(stored.get(k), settings.get(k)))
            if diff:
                raise ValueError(f"{folder} holds results with different settings ({', '.join(diff)}); "
                                 "write to another folder with --out")
        else:
            cfg_path.write_text(json.dumps(settings, indent=2) + "\n")
        for table, rows in tables.items():
            if not rows:
                continue
            path = folder / f"{table}.csv"
            new = pd.DataFrame([{"seed": seed, **{k: flatten(v) for k, v in r.items()}} for r in rows])
            if path.exists():
                old = pd.read_csv(path, **READ_KWARGS)
                new = pd.concat([old[old["seed"] != seed], new], ignore_index=True)
            new = new.sort_values("seed", kind="stable")
            _atomic_write_csv(new, path)
    return folder
