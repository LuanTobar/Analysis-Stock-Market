"""
Asset Repository - Implementa IAssetRepository.

CONCEPTO - Repository Pattern:
-----------------------------
El Repository es responsable de:
1. Obtener datos (del Data Source)
2. Transformarlos al formato correcto
3. Construir entidades de dominio (Asset)
4. Retornarlas listas para usar

Es el puente entre Infrastructure y Domain.
"""

from typing import List
import logging


from domain import Ticker, DateRange, Asset
from domain.exceptions import DataFetchError
from use_cases.interfaces import IAssetRepository, IMarketDataSource
from infrastructure.data_source.yfinance_source import YFinanceDataSource


logger = logging.getLogger(__name__)


class YFinanceAssetRepository(IAssetRepository):
    """
    Implementación de IAssetRepository usando YFinance.
    
    RESPONSABILIDAD:
    - Coordinar descarga de datos (Data Source)
    - Construir entidades Asset
    - Validar que los Assets estén completos
    
    NO ES RESPONSABLE DE:
    - Lógica de negocio (eso es Domain)
    - Detalles de yfinance (eso es Data Source)
    """
    
    def __init__(self, data_source: IMarketDataSource = None):
        """
        Constructor con Dependency Injection.
        
        Args:
            data_source: Fuente de datos. Si es None, usa YFinance por defecto.
        
        CONCEPTO: Dependency Injection permite inyectar mock en tests.
        """
        self._data_source = data_source or YFinanceDataSource()
    
    def get_asset(self, ticker: Ticker, date_range: DateRange) -> Asset:
        """
        Obtiene un Asset completamente construido y validado.
        
        MAPEO CON TU CÓDIGO:
        Antes tenías que hacer:
        1. Descargar datos manualmente (yf.download)
        2. Limpiar MultiIndex
        3. Eliminar NaN
        4. Calcular métricas manualmente
        
        Ahora:
        asset = repository.get_asset(ticker, date_range)
        # Y el Asset ya está listo con todos sus métodos
        
        Flujo interno:
        1. Data Source descarga y limpia datos
        2. Repository construye Asset con esos datos
        3. Asset se valida a sí mismo en __post_init__
        4. Se retorna Asset listo para usar
        """
        logger.info(f"Getting asset for {ticker}")
        
        try:
            # 1. Obtener datos del Data Source
            data = self._data_source.fetch_historical_data(ticker, date_range)
            
            # 2. Construir entidad Asset
            # El Asset valida los datos en su __post_init__
            asset = Asset(
                ticker=ticker,
                data=data,
                name=str(ticker)  # Podríamos obtener el nombre real de yfinance
            )
            
            logger.info(f"Successfully created asset for {ticker} with {len(data)} data points")
            return asset
            
        except DataFetchError:
            # Re-lanzar errores de dominio
            raise
        except Exception as e:
            # Convertir errores técnicos en errores de dominio
            logger.error(f"Failed to create asset for {ticker}: {e}")
            raise DataFetchError(
                message=f"Failed to get asset for {ticker}",
                details={"ticker": str(ticker), "error": str(e)}
            )
    
    def get_multiple_assets(
        self, 
        tickers: List[Ticker], 
        date_range: DateRange
    ) -> List[Asset]:
        """
        Obtiene múltiples Assets.
        
        MAPEO CON TU CÓDIGO:
        Tu código (líneas 584-598):
            activos = {"IBB": "IBB", "MNR": "MNR", ...}
            datos_historicos = {}
            for nombre, ticker in activos.items():
                datos_historicos[nombre] = yf.download(ticker, ...)
        
        Ahora:
            assets = repository.get_multiple_assets(tickers, date_range)
        
        Ventajas:
        - Descarga en paralelo (más rápido)
        - Manejo de errores por ticker
        - Si un ticker falla, continúa con los demás
        - Assets ya construidos y validados
        """
        logger.info(f"Getting {len(tickers)} assets")
        
        try:
            # Descargar datos de todos los tickers
            # El Data Source maneja la descarga en paralelo
            data_dict = self._data_source.fetch_multiple(tickers, date_range)
            
            # Construir Assets
            assets = []
            for ticker, data in data_dict.items():
                try:
                    asset = Asset(
                        ticker=ticker,
                        data=data,
                        name=str(ticker)
                    )
                    assets.append(asset)
                    logger.info(f"Created asset for {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to create asset for {ticker}: {e}")
                    # Continuar con los demás
                    continue
            
            if not assets:
                raise DataFetchError(
                    message="Failed to create any assets",
                    details={"tickers": [str(t) for t in tickers]}
                )
            
            logger.info(f"Successfully created {len(assets)}/{len(tickers)} assets")
            return assets
            
        except DataFetchError:
            raise
        except Exception as e:
            logger.error(f"Failed to get multiple assets: {e}")
            raise DataFetchError(
                message="Failed to get multiple assets",
                details={"error": str(e)}
            )


# EJEMPLO DE USO:
"""
from infrastructure.repositories import YFinanceAssetRepository
from domain import Ticker, DateRange

# Crear repository
repository = YFinanceAssetRepository()

# Obtener un asset
ticker = Ticker("AAPL")
date_range = DateRange("2020-01-01", "2024-11-30")
asset = repository.get_asset(ticker, date_range)

# El asset ya está listo para usar
print(asset.annualized_return())
print(asset.annualized_volatility())
print(asset.sharpe_ratio())

# Obtener múltiples assets
tickers = [Ticker("IBB"), Ticker("MGPHF"), Ticker("SMH")]
assets = repository.get_multiple_assets(tickers, date_range)

# Crear portfolio con los assets
from domain import Portfolio
portfolio = Portfolio(assets=assets)
print(portfolio.get_summary())
"""


# COMPARACIÓN CON TU CÓDIGO:
"""
# ❌ ANTES (múltiples pasos manuales):

# 1. Descargar
ibb_data = yf.download("IBB", start="2020-01-01", end="2024-11-30")

# 2. Limpiar MultiIndex
ibb_data.columns = ['_'.join(col) for col in ibb_data.columns]

# 3. Eliminar NaN
ibb_data = ibb_data.dropna(subset=['Adj Close_IBB'])

# 4. Calcular métricas manualmente
ibb_data['Daily Return'] = ibb_data['Adj Close_IBB'].pct_change()
daily_return_mean = ibb_data['Daily Return'].mean()
annual_return = (1 + daily_return_mean)**252 - 1

# 5. Repetir para cada activo... 😫


# ✅ AHORA (un solo paso):

asset = repository.get_asset(Ticker("IBB"), date_range)

# Todo ya está hecho:
# - Descarga
# - Limpieza
# - Validación
# - Construcción de entidad
# - Métricas disponibles

print(asset.annualized_return())  # Ya calculado
print(asset.sharpe_ratio())       # Ya calculado

# Ventajas:
# - Un solo paso
# - No repetir código
# - Manejo de errores incluido
# - Testeable (mock data source)
# - Reutilizable
"""


# PRINCIPIOS CLEAN CODE APLICADOS:
# =================================
#
# 1. REPOSITORY PATTERN: Abstrae cómo se obtienen los Assets
#
# 2. DEPENDENCY INJECTION: Recibe data_source en constructor
#
# 3. SINGLE RESPONSIBILITY: Solo construye Assets, no descarga ni calcula
#
# 4. ERROR HANDLING: Convierte errores técnicos en errores de dominio
#
# 5. LOGGING: Operaciones importantes loggeadas
#
# 6. SEPARATION OF CONCERNS: 
#    - Data Source → Descarga
#    - Repository → Construcción
#    - Asset → Cálculos
#
# 7. COMPOSITION OVER INHERITANCE: Usa Data Source, no hereda