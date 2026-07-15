"""Configuration loading and lightweight path helpers.

The YAML import is lazy so that ``import ocrbench.config`` works even before
PyYAML is installed; only :func:`load_config` requires it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_CONFIG_NAME = "config.yaml"


def _find_default_config() -> Path:
    """Locate config.yaml: env override, cwd, then repo root next to this pkg."""
    env = os.environ.get("OCRBENCH_CONFIG")
    if env:
        return Path(env)

    cwd_candidate = Path.cwd() / DEFAULT_CONFIG_NAME
    if cwd_candidate.exists():
        return cwd_candidate

    # repo root is the parent of the ocrbench package directory
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / DEFAULT_CONFIG_NAME


def load_config(path: str | os.PathLike | None = None) -> "Config":
    """Load and return the benchmark configuration."""
    import yaml  # lazy import

    cfg_path = Path(path) if path is not None else _find_default_config()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data, base_dir=cfg_path.resolve().parent)


class Config:
    """Thin wrapper over the parsed YAML dict with path resolution helpers.

    Access raw values with ``cfg["section"]["key"]`` (dict semantics) or through
    the convenience accessors below.
    """

    def __init__(self, data: Dict[str, Any], base_dir: Path | None = None):
        self.data = data
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    # -- dict passthrough -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    # -- path helpers -----------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a value from the ``paths`` section against the config dir."""
        raw = self.data.get("paths", {})[key]
        p = Path(raw)
        return p if p.is_absolute() else (self.base_dir / p)

    def images_dir(self) -> Path:
        return self.path("images_dir")

    def raw_dir(self) -> Path:
        return self.path("raw_dir")

    def ground_truth_dir(self) -> Path:
        return self.path("ground_truth_dir")

    def results_dir(self) -> Path:
        return self.path("results_dir")

    def input_pdfs(self) -> Path:
        return self.path("input_pdfs")

    # -- typed sub-sections ----------------------------------------------
    @property
    def dpi(self) -> int:
        return int(self.data.get("render", {}).get("dpi", 300))

    @property
    def manifests(self) -> Dict[str, List[int]]:
        return {str(k): list(v) for k, v in self.data.get("manifests", {}).items()}

    @property
    def boilerplate(self) -> List[str]:
        return list(self.data.get("boilerplate", []))

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self.data.get("scorecard", {}).get("weights", {}))

    def cost_per_page(self, engine: str) -> float:
        costs = self.data.get("costs", {})
        return float(costs.get(f"{engine}_per_page", 0.0))

    # -- Doc AI settings ----------------------------------------------------
    # Environment variables win over config.yaml. This keeps the real project
    # id / processor id out of version control (config.yaml only ever ships
    # placeholders); production values are supplied via the environment on
    # whatever machine actually runs the docai engine.
    _DOCAI_ENV_VARS = {
        "project_id": "OCRBENCH_DOCAI_PROJECT_ID",
        "region": "OCRBENCH_DOCAI_REGION",
        "processor_id": "OCRBENCH_DOCAI_PROCESSOR_ID",
    }

    def docai_settings(self) -> Dict[str, str]:
        """Resolve project_id/region/processor_id: env var first, config.yaml fallback."""
        dcfg = self.data.get("docai", {})
        resolved = {
            key: os.environ.get(env_var) or str(dcfg.get(key, ""))
            for key, env_var in self._DOCAI_ENV_VARS.items()
        }
        # Google Cloud regional endpoints expect lowercase ("us", "eu"),
        # regardless of how the value was supplied.
        resolved["region"] = resolved["region"].lower()
        return resolved
