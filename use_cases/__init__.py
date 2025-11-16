"""
Use Cases Layer - Casos de uso de la aplicación.

Esta capa orquesta el flujo de trabajo de la aplicación,
coordinando entidades de dominio y servicios de infraestructura.
"""

from .analyze_asset import (
    AnalyzeAssetUseCase,
    AnalyzeAssetRequest,
    AnalyzeAssetResponse
)

from .optimize_portfolio import (
    OptimizePortfolioUseCase,
    OptimizePortfolioRequest,
    OptimizePortfolioResponse,
    OptimizationStrategy
)

from .calculate_efficient_frontier import (
    CalculateEfficientFrontierUseCase,
    CalculateEfficientFrontierRequest,
    CalculateEfficientFrontierResponse,
    PortfolioPoint
)

from .interfaces import (
    IMarketDataSource,
    IAssetRepository,
    IPortfolioAnalyzer,
    IReportGenerator
)

__all__ = [
    # Analyze Asset
    'AnalyzeAssetUseCase',
    'AnalyzeAssetRequest',
    'AnalyzeAssetResponse',
    # Optimize Portfolio
    'OptimizePortfolioUseCase',
    'OptimizePortfolioRequest',
    'OptimizePortfolioResponse',
    'OptimizationStrategy',
    # Efficient Frontier
    'CalculateEfficientFrontierUseCase',
    'CalculateEfficientFrontierRequest',
    'CalculateEfficientFrontierResponse',
    'PortfolioPoint',
    # Interfaces
    'IMarketDataSource',
    'IAssetRepository',
    'IPortfolioAnalyzer',
    'IReportGenerator',
]