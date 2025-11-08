"""YAML tabanlı ayar yükleyicisi.

Örnek:
    from src.config.settings import get_settings

    settings = get_settings()
    print(settings.runtime.symbol)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataSourceConfig:
    """Veri kaynağı yapılandırması."""

    type: str
    path: Optional[str] = None
    delay_seconds: float = 0.0


@dataclass
class RuntimeConfig:
    """Çalışma zamanı parametrelerini kapsar."""

    symbol: str
    quote: str
    timeframe: str
    max_latency_seconds: int
    reconnect_backoff_seconds: int
    fee_bps: int
    slippage_bps: int
    min_hold_bars: int
    cooldown_after_stop_minutes: int
    data_source: Optional[DataSourceConfig] = None


@dataclass
class MetricsPenaltyConfig:
    winrate: float
    profit_factor: float
    sharpe: float
    roi: float
    mdd: float
    alpha_floor: float
    alpha_cap: float
    increase_factor: float
    decrease_factor: float


@dataclass
class MetricsRewardConfig:
    pnl_scale: float
    vola_lambda: float
    vola_window: int


@dataclass
class MetricsWindowsConfig:
    winrate: int
    profit_factor: int
    sharpe: int
    mdd: int
    roi_days: int


@dataclass
class MetricsTargetsConfig:
    winrate: float
    profit_factor: float
    sharpe: float
    roi: float
    mdd: float


@dataclass
class MetricsConfig:
    windows: MetricsWindowsConfig
    targets: MetricsTargetsConfig
    reward: MetricsRewardConfig
    penalties: MetricsPenaltyConfig


@dataclass
class SizingConfig:
    base: float
    beta0: float
    beta_sharpe: float
    beta_mdd: float
    kappa_max: float


@dataclass
class PfWrFloor:
    profit_factor: float
    winrate: float
    window: int


@dataclass
class SafetyConfig:
    drawdown_soft: float
    drawdown_hard: float
    sharpe_floor: float
    pf_wr_floor: PfWrFloor
    roi_floor: float
    cooldown_after_stop_minutes: int


@dataclass
class BanditConfig:
    algo: str
    base_exploration: float
    mild_penalty: float
    severe_penalty: float
    recovery_rate: float
    max_exploration: float
    min_exploration: float


@dataclass
class Settings:
    runtime: RuntimeConfig
    metrics: MetricsConfig
    sizing: SizingConfig
    safety: SafetyConfig
    bandit: BanditConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _parse_settings(data: Dict[str, Any]) -> Settings:
    runtime_raw = dict(data["runtime"])
    data_source_raw = runtime_raw.pop("data_source", None)
    data_source = None
    if data_source_raw is not None:
        data_source = DataSourceConfig(**data_source_raw)
    runtime = RuntimeConfig(**runtime_raw, data_source=data_source)
    metrics = MetricsConfig(
        windows=MetricsWindowsConfig(**data["metrics"]["windows"]),
        targets=MetricsTargetsConfig(**data["metrics"]["targets"]),
        reward=MetricsRewardConfig(**data["metrics"]["reward"]),
        penalties=MetricsPenaltyConfig(**data["metrics"]["penalties"]),
    )
    sizing = SizingConfig(**data["sizing"])
    pf_wr_floor = PfWrFloor(**data["safety"]["pf_wr_floor"])
    safety = SafetyConfig(
        pf_wr_floor=pf_wr_floor,
        drawdown_soft=data["safety"]["drawdown_soft"],
        drawdown_hard=data["safety"]["drawdown_hard"],
        sharpe_floor=data["safety"]["sharpe_floor"],
        roi_floor=data["safety"]["roi_floor"],
        cooldown_after_stop_minutes=data["safety"]["cooldown_after_stop_minutes"],
    )
    bandit = BanditConfig(**data["bandit"])
    return Settings(runtime=runtime, metrics=metrics, sizing=sizing, safety=safety, bandit=bandit)


@lru_cache(maxsize=1)
def get_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Ayarları dosyadan yükle ve bellekte sakla."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Ayar dosyası bulunamadı: {config_path}")
    raw = _load_yaml(config_path)
    return _parse_settings(raw)
