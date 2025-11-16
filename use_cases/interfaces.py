"""
Interfaces (Puertos) - Abstracciones para Dependency Inversion.

CONCEPTO CLEAN CODE - Dependency Inversion Principle (DIP):
----------------------------------------------------------
"Depende de abstracciones, no de concreciones"

En vez de que los Use Cases dependan directamente de yfinance,
dependen de INTERFACES (contratos). Luego, Infrastructure implementa esas interfaces.

Ventajas:
1. Podemos cambiar yfinance por otro proveedor sin tocar Use Cases
2. Podemos hacer tests sin internet (mocks)
3. El core de la aplicación no depende de librerías externas

Ejemplo:
    ❌ MAL: UseCase importa yfinance directamente
    ✅ BIEN: UseCase depende de IMarketDataSource (interfaz)
             YFinanceSource implementa IMarketDataSource
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
from datetime import datetime


from domain import Ticker, DateRange, Asset


class IMarketDataSource(ABC):
    """
    Interfaz para fuentes de datos de mercado.
    
    CONCEPTO: Interface Segregation Principle - interfaz pequeña y específica.
    Solo define las operaciones que REALMENTE necesitamos.
    
    Cualquier implementación (yfinance, Alpha Vantage, archivo CSV)
    debe cumplir este contrato.
    """
    
    @abstractmethod
    def fetch_historical_data(
        self, 
        ticker: Ticker, 
        date_range: DateRange
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de un activo.
        
        Args:
            ticker: Símbolo del activo
            date_range: Rango de fechas
            
        Returns:
            DataFrame con columnas: 
            - Index: DatetimeIndex
            - 'Adj Close': float
            - 'Volume': float (opcional)
            
        Raises:
            DataFetchError: Si falla la descarga
            
        CONCEPTO: El contrato especifica QUÉ debe retornar,
        no CÓMO debe obtenerlo. Eso es responsabilidad de la implementación.
        """
        pass
    
    @abstractmethod
    def fetch_multiple(
        self, 
        tickers: List[Ticker], 
        date_range: DateRange
    ) -> Dict[Ticker, pd.DataFrame]:
        """
        Descarga datos de múltiples activos en paralelo.
        
        Args:
            tickers: Lista de símbolos a descargar
            date_range: Rango de fechas común
            
        Returns:
            Dict con ticker como key y DataFrame como value
            
        CONCEPTO: Batch operation - más eficiente que llamar
        fetch_historical_data() en un loop.
        """
        pass


class IAssetRepository(ABC):
    """
    Interfaz para repositorio de activos.
    
    CONCEPTO: Repository Pattern - abstrae el acceso a datos.
    La fuente puede ser API, base de datos, caché, archivo...
    
    El Use Case no sabe ni le importa DE DÓNDE vienen los Assets,
    solo los pide al repositorio.
    """
    
    @abstractmethod
    def get_asset(self, ticker: Ticker, date_range: DateRange) -> Asset:
        """
        Obtiene un Asset completamente construido.
        
        Returns:
            Asset entity lista para usar
            
        CONCEPTO: El repositorio se encarga de:
        1. Obtener los datos (del source)
        2. Transformarlos al formato correcto
        3. Construir la entidad Asset
        4. Validar que esté en buen estado
        
        El Use Case solo recibe un Asset listo.
        """
        pass
    
    @abstractmethod
    def get_multiple_assets(
        self, 
        tickers: List[Ticker], 
        date_range: DateRange
    ) -> List[Asset]:
        """
        Obtiene múltiples Assets.
        
        Returns:
            Lista de Assets listos para usar
        """
        pass


class IPortfolioAnalyzer(ABC):
    """
    Interfaz para analizadores de Portfolio.
    
    CONCEPTO: Strategy Pattern - diferentes algoritmos de análisis
    pueden implementar esta interfaz.
    
    Podríamos tener:
    - ModernPortfolioTheoryAnalyzer
    - BlackLittermanAnalyzer
    - RiskParityAnalyzer
    
    Todos implementando la misma interfaz.
    """
    
    @abstractmethod
    def calculate_efficient_frontier(
        self, 
        assets: List[Asset],
        n_portfolios: int = 10000
    ) -> pd.DataFrame:
        """
        Calcula la frontera eficiente mediante simulación.
        
        Args:
            assets: Lista de activos a incluir
            n_portfolios: Número de carteras a simular
            
        Returns:
            DataFrame con columnas:
            - 'return': Retorno de la cartera
            - 'volatility': Volatilidad de la cartera
            - 'sharpe': Sharpe ratio
            - 'weights': Dict con pesos
        """
        pass


class IReportGenerator(ABC):
    """
    Interfaz para generadores de reportes.
    
    CONCEPTO: Open/Closed Principle - podemos tener múltiples
    implementaciones (PDF, HTML, Excel) sin modificar Use Cases.
    """
    
    @abstractmethod
    def generate_asset_report(self, asset: Asset, output_path: str) -> str:
        """
        Genera reporte de un activo individual.
        
        Returns:
            Path al archivo generado
        """
        pass
    
    @abstractmethod
    def generate_portfolio_report(
        self, 
        portfolio: 'Portfolio',
        output_path: str
    ) -> str:
        """
        Genera reporte completo de cartera.
        
        Returns:
            Path al archivo generado
        """
        pass


# EJEMPLO DE USO EN USE CASE:
"""
class AnalyzeAssetUseCase:
    def __init__(
        self, 
        repository: IAssetRepository,  # ← Depende de interfaz
        report_generator: IReportGenerator
    ):
        self._repository = repository
        self._report_generator = report_generator
    
    def execute(self, ticker: Ticker, date_range: DateRange):
        # Usar la interfaz, no saber la implementación
        asset = self._repository.get_asset(ticker, date_range)
        report = self._report_generator.generate_asset_report(asset)
        return report

# En producción:
use_case = AnalyzeAssetUseCase(
    repository=YFinanceRepository(),  # ← Implementación concreta
    report_generator=PDFReportGenerator()
)

# En tests:
use_case = AnalyzeAssetUseCase(
    repository=MockRepository(),  # ← Mock para testing
    report_generator=MockReportGenerator()
)
"""


# RESUMEN DE CONCEPTOS:
# ====================
#
# 1. ABSTRACTION: Las interfaces definen QUÉ, no CÓMO
#
# 2. DEPENDENCY INVERSION: Use Cases dependen de interfaces,
#    no de implementaciones concretas (yfinance)
#
# 3. TESTABILITY: Podemos inyectar mocks en tests
#
# 4. FLEXIBILITY: Cambiar de proveedor es cambiar la implementación,
#    no tocar Use Cases
#
# 5. SEPARATION: El core (Use Cases) no sabe nada de yfinance,
#    pandas, matplotlib, etc.