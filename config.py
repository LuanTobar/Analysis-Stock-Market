"""
Configuración de la aplicación - Centralización de parámetros.

CONCEPTO: No magic numbers - todo configurado en un solo lugar.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass(frozen=True)
class MarketConfig:
    """Configuración de parámetros de mercado."""
    trading_days_per_year: int = 252
    default_risk_free_rate: float = 0.02


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuración para optimización de carteras."""
    default_n_simulations: int = 10000
    min_weight: float = 0.0
    max_weight: float = 1.0
    weight_sum_tolerance: float = 1e-6
    max_iterations: int = 1000
    optimization_method: str = 'SLSQP'


@dataclass
class CacheConfig:
    """Configuración de caché."""
    enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path.home() / '.portfolio_analyzer' / 'cache')
    ttl_days: int = 1
    
    def __post_init__(self):
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ApplicationConfig:
    """Configuración principal de la aplicación."""
    market: MarketConfig = field(default_factory=MarketConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    @classmethod
    def testing(cls) -> 'ApplicationConfig':
        """Config para tests (sin caché, simulaciones rápidas)."""
        return cls(
            cache=CacheConfig(enabled=False),
            optimization=OptimizationConfig(default_n_simulations=100)
        )


_default_config: Optional[ApplicationConfig] = None


def get_config() -> ApplicationConfig:
    """Obtiene la configuración de la aplicación."""
    global _default_config
    if _default_config is None:
        _default_config = ApplicationConfig()
    return _default_config


def set_config(config: ApplicationConfig) -> None:
    """Establece una configuración personalizada."""
    global _default_config
    _default_config = config
