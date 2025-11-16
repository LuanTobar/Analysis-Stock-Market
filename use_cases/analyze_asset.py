"""
Use Case: Analizar Activo Individual.

CONCEPTO CLEAN CODE - Use Case (Interactor):
-------------------------------------------
Un Use Case representa UN flujo de trabajo específico de la aplicación.
Orquesta las entidades del dominio y servicios de infraestructura
para cumplir un objetivo de negocio.

Responsabilidades:
1. Coordinar el flujo de trabajo
2. Manejar errores de aplicación
3. NO contener lógica de negocio (esa está en Domain)
4. NO saber de detalles técnicos (esos están en Infrastructure)

Analogía:
- Domain = Actores (saben actuar)
- Use Case = Director (coordina a los actores)
- Infrastructure = Utilería (provee recursos)
"""

from dataclasses import dataclass
from typing import Optional, Dict

import sys
sys.path.append(r'c:\Users\Dentaldata1\Documents\Lucho Brainstorming\Analysis-Stock-Market-main\Analysis-Stock-Market-main')

from domain import Ticker, DateRange, Asset
from domain.exceptions import InsufficientDataError, DataFetchError
from use_cases.interfaces import IAssetRepository


@dataclass
class AnalyzeAssetRequest:
    """
    DTO (Data Transfer Object) para la request del Use Case.
    
    CONCEPTO: Encapsular parámetros de entrada en un objeto.
    
    Ventajas:
    - Validación centralizada
    - Fácil de extender (agregar campos sin romper firmas)
    - Documentación clara de qué necesita el Use Case
    """
    ticker_symbol: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    risk_free_rate: float = 0.0
    
    def to_domain(self) -> tuple[Ticker, DateRange]:
        """
        Convierte DTO a objetos de dominio.
        
        CONCEPTO: Anti-Corruption Layer - convierte datos externos
        al lenguaje del dominio.
        """
        ticker = Ticker(self.ticker_symbol)
        date_range = DateRange(self.start_date, self.end_date)
        return ticker, date_range


@dataclass
class AnalyzeAssetResponse:
    """
    DTO para la respuesta del Use Case.
    
    CONCEPTO: El Use Case retorna datos estructurados,
    no entidades de dominio directamente.
    
    Esto permite que el dominio evolucione sin afectar
    a los consumidores del Use Case.
    """
    success: bool
    ticker: str
    statistics: Optional[Dict] = None
    error_message: Optional[str] = None
    
    @classmethod
    def success_response(cls, asset: Asset, risk_free_rate: float) -> 'AnalyzeAssetResponse':
        """
        Factory method para crear respuesta exitosa.
        
        CONCEPTO: Named Constructor - hace explícito el propósito.
        """
        stats = asset.get_summary_statistics()
        stats['sharpe_ratio_custom'] = asset.sharpe_ratio(risk_free_rate)
        
        return cls(
            success=True,
            ticker=str(asset.ticker),
            statistics=stats,
            error_message=None
        )
    
    @classmethod
    def error_response(cls, ticker: str, error: Exception) -> 'AnalyzeAssetResponse':
        """Factory method para crear respuesta de error."""
        return cls(
            success=False,
            ticker=ticker,
            statistics=None,
            error_message=str(error)
        )


class AnalyzeAssetUseCase:
    """
    Caso de uso: Analizar un activo financiero individual.
    
    Flujo:
    1. Validar entrada
    2. Obtener datos del activo (via Repository)
    3. Calcular métricas (delegado al Asset)
    4. Retornar resultados estructurados
    
    CONCEPTO: Single Responsibility - este Use Case solo hace
    análisis de activo individual. Nada más.
    """
    
    def __init__(self, asset_repository: IAssetRepository):
        """
        Constructor con Dependency Injection.
        
        CONCEPTO: Dependency Inversion - recibimos una INTERFAZ,
        no una implementación concreta.
        
        Esto permite:
        - Testing con mocks
        - Cambiar implementación sin tocar este código
        - Claridad sobre qué dependencias necesita
        """
        self._repository = asset_repository
    
    def execute(self, request: AnalyzeAssetRequest) -> AnalyzeAssetResponse:
        """
        Ejecuta el caso de uso.
        
        CONCEPTO: Command Pattern - un método que hace UNA cosa completa.
        
        Args:
            request: DTO con parámetros de entrada
            
        Returns:
            Response: DTO con resultados o error
            
        IMPORTANTE: Este método NO lanza excepciones.
        Captura errores y los convierte en respuestas estructuradas.
        """
        try:
            # 1. Convertir request a objetos de dominio
            ticker, date_range = request.to_domain()
            
            # 2. Obtener Asset del repositorio
            # NOTA: El repositorio se encarga de descargar datos,
            # validarlos, y construir el Asset
            asset = self._repository.get_asset(ticker, date_range)
            
            # 3. El Asset ya tiene todos los métodos para calcular métricas
            # No necesitamos lógica adicional aquí
            
            # 4. Retornar respuesta exitosa
            return AnalyzeAssetResponse.success_response(
                asset=asset,
                risk_free_rate=request.risk_free_rate
            )
            
        except (InsufficientDataError, DataFetchError) as e:
            # Errores de dominio/negocio
            return AnalyzeAssetResponse.error_response(
                ticker=request.ticker_symbol,
                error=e
            )
        
        except Exception as e:
            # Errores inesperados
            return AnalyzeAssetResponse.error_response(
                ticker=request.ticker_symbol,
                error=Exception(f"Unexpected error: {str(e)}")
            )


# EJEMPLO DE USO:
"""
# Setup (normalmente en main.py o dependency container)
from infrastructure.repositories import YFinanceAssetRepository

repository = YFinanceAssetRepository()
use_case = AnalyzeAssetUseCase(repository)

# Ejecutar
request = AnalyzeAssetRequest(
    ticker_symbol="AAPL",
    start_date="2020-01-01",
    end_date="2024-11-30",
    risk_free_rate=0.02
)

response = use_case.execute(request)

if response.success:
    print(f"Análisis de {response.ticker}:")
    print(f"Retorno anualizado: {response.statistics['annualized_return']}")
    print(f"Volatilidad: {response.statistics['annualized_volatility']}")
    print(f"Sharpe Ratio: {response.statistics['sharpe_ratio_custom']:.3f}")
else:
    print(f"Error: {response.error_message}")
"""


# COMPARACIÓN CON TU CÓDIGO ORIGINAL:
"""
# ❌ ANTES (procedural, mezclado):
ticker = "IBB"
start_date = "2020-01-01"
end_date = "2024-11-30"

ibb_data = yf.download(ticker, start=start_date, end=end_date)  # Descarga
ibb_data.columns = ['_'.join(col) for col in ibb_data.columns]  # Limpieza
ibb_data = ibb_data.dropna(subset=['Adj Close_IBB'])  # Validación
ibb_data['Daily Return'] = ibb_data['Adj Close_IBB'].pct_change()  # Cálculo
daily_return_mean = ibb_data['Daily Return'].mean()  # Cálculo
annual_return = (1 + daily_return_mean)**252 - 1  # Cálculo
print(f"Rendimiento anual: {annual_return:.2%}")  # Presentación

# TODO mezclado, difícil de testear, no reutilizable


# ✅ AHORA (Clean Architecture):
request = AnalyzeAssetRequest(
    ticker_symbol="IBB",
    start_date="2020-01-01",
    end_date="2024-11-30"
)

response = use_case.execute(request)

# Separación clara:
# - Repository descarga y valida
# - Asset calcula métricas
# - Use Case orquesta
# - Presentation muestra resultados

# Beneficios:
# - Testeable sin internet
# - Reutilizable
# - Fácil de mantener
# - Claro qué hace cada parte
"""


# PRINCIPIOS CLEAN CODE APLICADOS:
# =================================
#
# 1. SINGLE RESPONSIBILITY: 
#    - Use Case solo orquesta
#    - No descarga datos (eso es Repository)
#    - No calcula métricas (eso es Asset)
#
# 2. DEPENDENCY INVERSION:
#    - Depende de IAssetRepository (interfaz)
#    - No de YFinanceRepository (implementación)
#
# 3. FAIL GRACEFULLY:
#    - No lanza excepciones
#    - Retorna respuestas estructuradas con errores
#
# 4. MEANINGFUL NAMES:
#    - execute() hace lo que dice
#    - success_response() es claro
#
# 5. DTO PATTERN:
#    - Request/Response encapsulan datos
#    - Separan interfaz de implementación
#
# 6. TESTABILITY:
#    - Podemos inyectar mock repository
#    - Tests sin internet, rápidos