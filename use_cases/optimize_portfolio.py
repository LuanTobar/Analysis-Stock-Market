"""
Use Case: Optimizar Cartera de Inversión.

Este Use Case implementa el flujo completo de optimización de portfolio
que hacías en tu código original con scipy.optimize.

DIFERENCIA CLAVE:
- Antes: Todo el código en un script mezclado
- Ahora: Orquestación clara que delega responsabilidades
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


from domain import Ticker, DateRange, Portfolio
from domain.exceptions import OptimizationError, PortfolioValidationError
from use_cases.interfaces import IAssetRepository


class OptimizationStrategy(Enum):
    """
    Estrategias de optimización disponibles.
    
    CONCEPTO: Enum para valores predefinidos.
    Previene errores de tipeo y documenta opciones.
    """
    MINIMUM_VARIANCE = "minimum_variance"
    TARGET_VOLATILITY = "target_volatility"
    MAXIMUM_SHARPE = "maximum_sharpe"


@dataclass
class OptimizePortfolioRequest:
    """
    Request DTO para optimización de cartera.
    """
    ticker_symbols: List[str]
    start_date: str
    end_date: str
    strategy: OptimizationStrategy
    target_volatility: Optional[float] = None  # Solo para TARGET_VOLATILITY
    risk_free_rate: float = 0.0
    
    def validate(self) -> None:
        """
        Validación de negocio del request.
        
        CONCEPTO: Fail Fast - validar antes de procesar.
        """
        if not self.ticker_symbols:
            raise ValueError("Must provide at least one ticker")
        
        if len(self.ticker_symbols) < 2:
            raise ValueError("Portfolio optimization requires at least 2 assets")
        
        if self.strategy == OptimizationStrategy.TARGET_VOLATILITY:
            if self.target_volatility is None:
                raise ValueError("target_volatility required for TARGET_VOLATILITY strategy")
            if self.target_volatility <= 0:
                raise ValueError("target_volatility must be positive")
    
    def to_domain(self) -> tuple[List[Ticker], DateRange]:
        """Convierte a objetos de dominio."""
        tickers = [Ticker(symbol) for symbol in self.ticker_symbols]
        date_range = DateRange(self.start_date, self.end_date)
        return tickers, date_range


@dataclass
class OptimizePortfolioResponse:
    """
    Response DTO con resultados de optimización.
    """
    success: bool
    original_portfolio: Optional[Dict] = None
    optimized_portfolio: Optional[Dict] = None
    improvement: Optional[Dict] = None
    error_message: Optional[str] = None
    
    @classmethod
    def success_response(
        cls,
        original: Portfolio,
        optimized: Portfolio,
        risk_free_rate: float
    ) -> 'OptimizePortfolioResponse':
        """
        Crea respuesta exitosa con comparación.
        
        CONCEPTO: El response incluye tanto el resultado como
        información contextual (comparación antes/después).
        """
        original_summary = original.get_summary()
        optimized_summary = optimized.get_summary()
        
        # Calcular mejora
        original_sharpe = original.sharpe_ratio(risk_free_rate)
        optimized_sharpe = optimized.sharpe_ratio(risk_free_rate)
        
        original_vol = original.annualized_volatility().value
        optimized_vol = optimized.annualized_volatility().value
        
        improvement = {
            'sharpe_improvement': optimized_sharpe - original_sharpe,
            'sharpe_improvement_pct': ((optimized_sharpe / original_sharpe - 1) * 100) 
                if original_sharpe != 0 else 0,
            'volatility_reduction': original_vol - optimized_vol,
            'volatility_reduction_pct': ((optimized_vol / original_vol - 1) * 100)
                if original_vol != 0 else 0
        }
        
        return cls(
            success=True,
            original_portfolio=original_summary,
            optimized_portfolio=optimized_summary,
            improvement=improvement,
            error_message=None
        )
    
    @classmethod
    def error_response(cls, error: Exception) -> 'OptimizePortfolioResponse':
        """Crea respuesta de error."""
        return cls(
            success=False,
            original_portfolio=None,
            optimized_portfolio=None,
            improvement=None,
            error_message=str(error)
        )


class OptimizePortfolioUseCase:
    """
    Caso de uso: Optimizar asignación de pesos en cartera.
    
    Flujo:
    1. Validar request
    2. Obtener assets del repositorio
    3. Crear portfolio inicial (pesos iguales)
    4. Ejecutar optimización según estrategia
    5. Retornar comparación antes/después
    
    MAPEO CON TU CÓDIGO ORIGINAL:
    Tu código hacía esto en ~100 líneas dispersas.
    Ahora está organizado, testeable y reutilizable.
    """
    
    def __init__(self, asset_repository: IAssetRepository):
        """
        Dependency Injection del repositorio.
        
        CONCEPTO: El Use Case NO sabe de yfinance.
        Solo sabe que hay "algo" que le provee Assets.
        """
        self._repository = asset_repository
    
    def execute(self, request: OptimizePortfolioRequest) -> OptimizePortfolioResponse:
        """
        Ejecuta la optimización de cartera.
        
        Este método reemplaza todo el código que tenías disperso
        para optimización de cartera.
        """
        try:
            # 1. Validar request
            request.validate()
            
            # 2. Convertir a objetos de dominio
            tickers, date_range = request.to_domain()
            
            # 3. Obtener assets
            # El repositorio se encarga de descargar todos los datos
            assets = self._repository.get_multiple_assets(tickers, date_range)
            
            if not assets:
                raise ValueError("No assets could be retrieved")
            
            # 4. Crear portfolio inicial con pesos iguales
            # MAPEO: Esto reemplaza tu código:
            # pesos_iniciales = np.ones(len(df_precios.columns)) / len(df_precios.columns)
            original_portfolio = Portfolio(
                assets=assets,
                name="Original Portfolio (Equal Weights)"
            )
            
            # 5. Ejecutar optimización según estrategia
            optimized_portfolio = self._optimize_by_strategy(
                portfolio=original_portfolio,
                strategy=request.strategy,
                target_volatility=request.target_volatility
            )
            
            # 6. Retornar comparación
            return OptimizePortfolioResponse.success_response(
                original=original_portfolio,
                optimized=optimized_portfolio,
                risk_free_rate=request.risk_free_rate
            )
            
        except (OptimizationError, PortfolioValidationError, ValueError) as e:
            return OptimizePortfolioResponse.error_response(e)
        
        except Exception as e:
            return OptimizePortfolioResponse.error_response(
                Exception(f"Unexpected error: {str(e)}")
            )
    
    def _optimize_by_strategy(
        self,
        portfolio: Portfolio,
        strategy: OptimizationStrategy,
        target_volatility: Optional[float]
    ) -> Portfolio:
        """
        Ejecuta la optimización según la estrategia elegida.
        
        CONCEPTO: Strategy Pattern - diferentes algoritmos
        encapsulados en métodos separados.
        
        MAPEO CON TU CÓDIGO:
        - MINIMUM_VARIANCE → Tu código de minimizar riesgo
        - TARGET_VOLATILITY → Tu código con nivel_deseado = 0.016
        """
        if strategy == OptimizationStrategy.MINIMUM_VARIANCE:
            # MAPEO: Reemplaza tu código:
            # resultado = sco.minimize(riesgo_cartera, pesos_iniciales, ...)
            return portfolio.optimize_weights_minimum_variance()
        
        elif strategy == OptimizationStrategy.TARGET_VOLATILITY:
            if target_volatility is None:
                raise ValueError("target_volatility required")
            
            # MAPEO: Reemplaza tu código:
            # nivel_deseado = 0.016
            # resultado = minimize(objetivo_riesgo, ...)
            return portfolio.optimize_weights_target_volatility(target_volatility)
        
        elif strategy == OptimizationStrategy.MAXIMUM_SHARPE:
            # Esto lo podríamos implementar en el futuro
            raise NotImplementedError("Maximum Sharpe optimization not yet implemented")
        
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")


# EJEMPLO DE USO:
"""
# Setup
from infrastructure.repositories import YFinanceAssetRepository

repository = YFinanceAssetRepository()
use_case = OptimizePortfolioUseCase(repository)

# Ejecutar optimización de mínima varianza
request = OptimizePortfolioRequest(
    ticker_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    start_date="2020-01-01",
    end_date="2024-11-30",
    strategy=OptimizationStrategy.MINIMUM_VARIANCE
)

response = use_case.execute(request)

if response.success:
    print("🎯 OPTIMIZACIÓN EXITOSA\n")
    
    print("📊 CARTERA ORIGINAL (Pesos Iguales):")
    print(f"  Retorno anualizado: {response.original_portfolio['annualized_return']}")
    print(f"  Volatilidad: {response.original_portfolio['annualized_volatility']}")
    print(f"  Sharpe Ratio: {response.original_portfolio['sharpe_ratio']:.3f}")
    print(f"  Pesos: {response.original_portfolio['weights']}\n")
    
    print("✨ CARTERA OPTIMIZADA:")
    print(f"  Retorno anualizado: {response.optimized_portfolio['annualized_return']}")
    print(f"  Volatilidad: {response.optimized_portfolio['annualized_volatility']}")
    print(f"  Sharpe Ratio: {response.optimized_portfolio['sharpe_ratio']:.3f}")
    print(f"  Pesos: {response.optimized_portfolio['weights']}\n")
    
    print("📈 MEJORA:")
    print(f"  Sharpe +{response.improvement['sharpe_improvement']:.3f} "
          f"({response.improvement['sharpe_improvement_pct']:.1f}%)")
    print(f"  Volatilidad {response.improvement['volatility_reduction']:.2f}% "
          f"({response.improvement['volatility_reduction_pct']:.1f}%)")
else:
    print(f"❌ Error: {response.error_message}")


# Optimizar con volatilidad objetivo
request_target = OptimizePortfolioRequest(
    ticker_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    start_date="2020-01-01",
    end_date="2024-11-30",
    strategy=OptimizationStrategy.TARGET_VOLATILITY,
    target_volatility=0.016  # 1.6% diario
)

response = use_case.execute(request_target)
"""


# COMPARACIÓN CON TU CÓDIGO ORIGINAL:
"""
# ❌ ANTES (tu código):
import scipy.optimize as sco

# Descargar datos manualmente
activos = {"IBB": "IBB", "MNR": "MNR", ...}
datos_historicos = {}
for nombre, ticker in activos.items():
    datos_historicos[nombre] = yf.download(ticker, start="2020-01-01", ...)

# Crear DataFrame manualmente
df_precios = pd.concat(datos_historicos.values(), axis=1)
df_precios.columns = datos_historicos.keys()

# Calcular covarianza manualmente
covarianza = df_precios.pct_change().cov()

# Definir funciones objetivo
def riesgo_cartera(pesos, covarianza):
    return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

def restriccion_pesos(pesos):
    return np.sum(pesos) - 1

# Optimizar
pesos_iniciales = np.ones(len(df_precios.columns)) / len(df_precios.columns)
restricciones = ({'type': 'eq', 'fun': restriccion_pesos})
resultado = sco.minimize(riesgo_cartera, pesos_iniciales, ...)

# Imprimir resultados manualmente
print(f"Pesos optimizados: {resultado.x}")

# TODO mezclado, difícil de mantener, no reutilizable


# ✅ AHORA (Clean Architecture):
request = OptimizePortfolioRequest(
    ticker_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    start_date="2020-01-01",
    end_date="2024-11-30",
    strategy=OptimizationStrategy.MINIMUM_VARIANCE
)

response = use_case.execute(request)

# Separación clara de responsabilidades:
# - Repository: Descarga datos
# - Asset: Calcula retornos individuales
# - Portfolio: Maneja matriz de covarianza y optimización
# - Use Case: Orquesta el flujo
# - Response: Estructura los resultados

# Beneficios:
# - Testeable
# - Reutilizable
# - Mantenible
# - Extensible (nuevas estrategias fáciles de agregar)
"""


# PRINCIPIOS CLEAN CODE APLICADOS:
# =================================
#
# 1. STRATEGY PATTERN: Diferentes optimizaciones sin if/else grandes
#
# 2. DTO PATTERN: Request/Response separan interfaz de implementación
#
# 3. SINGLE RESPONSIBILITY: Use Case solo orquesta, no optimiza
#
# 4. DEPENDENCY INVERSION: Depende de IAssetRepository (interfaz)
#
# 5. MEANINGFUL NAMES: execute(), _optimize_by_strategy() claros
#
# 6. ERROR HANDLING: No lanza excepciones, retorna respuestas estructuradas
#
# 7. VALIDATION: validate() centralizada y explícita