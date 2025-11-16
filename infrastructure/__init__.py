"""
Infrastructure Layer - Implementaciones concretas de interfaces.

Esta capa contiene:
- Data Sources: Acceso a APIs externas (yfinance)
- Repositories: Construcción de entidades de dominio
- Caché: Optimización de descargas
- Configuración: Settings de la aplicación
"""

from .data_source import YFinanceDataSource
from .repositories import YFinanceAssetRepository

__all__ = [
    'YFinanceDataSource',
    'YFinanceAssetRepository',
]