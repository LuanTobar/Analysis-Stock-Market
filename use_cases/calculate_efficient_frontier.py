"""
Use Case: Calcular Frontera Eficiente.

Este Use Case implementa la simulación de 10,000 carteras aleatorias
que hacías en tu código original para graficar la frontera eficiente.

MAPEO CON TU CÓDIGO:
Tu código hacía la simulación en ~40 líneas dentro del script.
Ahora está encapsulado, testeable y reutilizable.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


from domain import Ticker, DateRange, Portfolio, Asset
from use_cases.interfaces import IAssetRepository


@dataclass
class CalculateEfficientFrontierRequest:
    """Request DTO para cálculo de frontera eficiente."""
    ticker_symbols: List[str]
    start_date: str
    end_date: str
    n_simulations: int = 10000
    risk_free_rate: float = 0.0
    
    def validate(self) -> None:
        """Validación del request."""
        if not self.ticker_symbols:
            raise ValueError("Must provide at least one ticker")
        
        if len(self.ticker_symbols) < 2:
            raise ValueError("Efficient frontier requires at least 2 assets")
        
        if self.n_simulations < 100:
            raise ValueError("n_simulations must be at least 100")
        
        if self.n_simulations > 100000:
            raise ValueError("n_simulations too high (max 100,000)")
    
    def to_domain(self) -> tuple[List[Ticker], DateRange]:
        """Convierte a objetos de dominio."""
        tickers = [Ticker(symbol) for symbol in self.ticker_symbols]
        date_range = DateRange(self.start_date, self.end_date)
        return tickers, date_range


@dataclass
class PortfolioPoint:
    """
    Representa un punto en la frontera eficiente.
    
    CONCEPTO: Value Object para un punto de la simulación.
    Encapsula los datos de una cartera simulada.
    """
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass
class CalculateEfficientFrontierResponse:
    """
    Response con todos los puntos de la frontera eficiente.
    """
    success: bool
    frontier_points: Optional[List[PortfolioPoint]] = None
    optimal_portfolios: Optional[Dict[str, PortfolioPoint]] = None
    statistics: Optional[Dict] = None
    error_message: Optional[str] = None
    
    @classmethod
    def success_response(
        cls,
        frontier_points: List[PortfolioPoint],
        min_volatility: PortfolioPoint,
        max_sharpe: PortfolioPoint,
        n_simulations: int
    ) -> 'CalculateEfficientFrontierResponse':
        """Crea respuesta exitosa con puntos óptimos identificados."""
        
        # Calcular estadísticas de la frontera
        returns = [p.expected_return for p in frontier_points]
        volatilities = [p.volatility for p in frontier_points]
        sharpe_ratios = [p.sharpe_ratio for p in frontier_points]
        
        statistics = {
            'n_simulations': n_simulations,
            'n_assets': len(min_volatility.weights),
            'return_range': (min(returns), max(returns)),
            'volatility_range': (min(volatilities), max(volatilities)),
            'sharpe_range': (min(sharpe_ratios), max(sharpe_ratios))
        }
        
        return cls(
            success=True,
            frontier_points=frontier_points,
            optimal_portfolios={
                'minimum_volatility': min_volatility,
                'maximum_sharpe': max_sharpe
            },
            statistics=statistics,
            error_message=None
        )
    
    @classmethod
    def error_response(cls, error: Exception) -> 'CalculateEfficientFrontierResponse':
        """Crea respuesta de error."""
        return cls(
            success=False,
            frontier_points=None,
            optimal_portfolios=None,
            statistics=None,
            error_message=str(error)
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte los puntos a DataFrame para fácil análisis/visualización.
        
        CONCEPTO: Convenience method para compatibilidad con pandas.
        """
        if not self.frontier_points:
            return pd.DataFrame()
        
        data = []
        for point in self.frontier_points:
            row = {
                'return': point.expected_return,
                'volatility': point.volatility,
                'sharpe': point.sharpe_ratio
            }
            # Agregar pesos como columnas
            for ticker, weight in point.weights.items():
                row[f'weight_{ticker}'] = weight
            data.append(row)
        
        return pd.DataFrame(data)


class CalculateEfficientFrontierUseCase:
    """
    Caso de uso: Calcular frontera eficiente mediante simulación Monte Carlo.
    
    Flujo:
    1. Validar request
    2. Obtener assets
    3. Calcular matriz de covarianza y retornos esperados
    4. Simular N carteras con pesos aleatorios
    5. Calcular métricas de cada cartera
    6. Identificar carteras óptimas
    7. Retornar resultados estructurados
    
    MAPEO CON TU CÓDIGO ORIGINAL:
    Esto reemplaza tu código de simulación de ~40 líneas:
    
    for _ in range(num_simulaciones):
        pesos = np.random.random(len(df_precios.columns))
        pesos /= np.sum(pesos)
        rendimiento_simulado = np.sum(pesos * rendimientos_esperados)
        riesgo_simulado = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
        ...
    """
    
    def __init__(self, asset_repository: IAssetRepository):
        """Dependency Injection del repositorio."""
        self._repository = asset_repository
    
    def execute(
        self,
        request: CalculateEfficientFrontierRequest
    ) -> CalculateEfficientFrontierResponse:
        """
        Ejecuta la simulación de la frontera eficiente.
        
        NOTA: Este método hace EXACTAMENTE lo mismo que tu código original,
        pero está organizado, documentado y testeable.
        """
        try:
            # 1. Validar request
            request.validate()
            
            # 2. Convertir a objetos de dominio
            tickers, date_range = request.to_domain()
            
            # 3. Obtener assets
            assets = self._repository.get_multiple_assets(tickers, date_range)
            
            if not assets:
                raise ValueError("No assets could be retrieved")
            
            # 4. Simular carteras
            frontier_points = self._simulate_portfolios(
                assets=assets,
                n_simulations=request.n_simulations,
                risk_free_rate=request.risk_free_rate
            )
            
            # 5. Identificar carteras óptimas
            min_vol_point = min(frontier_points, key=lambda p: p.volatility)
            max_sharpe_point = max(frontier_points, key=lambda p: p.sharpe_ratio)
            
            # 6. Retornar respuesta
            return CalculateEfficientFrontierResponse.success_response(
                frontier_points=frontier_points,
                min_volatility=min_vol_point,
                max_sharpe=max_sharpe_point,
                n_simulations=request.n_simulations
            )
            
        except Exception as e:
            return CalculateEfficientFrontierResponse.error_response(e)
    
    def _simulate_portfolios(
        self,
        assets: List[Asset],
        n_simulations: int,
        risk_free_rate: float
    ) -> List[PortfolioPoint]:
        """
        Ejecuta la simulación Monte Carlo.
        
        MAPEO DIRECTO CON TU CÓDIGO:
        Esto es exactamente lo que hacías en el loop:
        
        for _ in range(num_simulaciones):
            pesos = np.random.random(len(df_precios.columns))
            pesos /= np.sum(pesos)
            ...
        """
        # Crear portfolio base para obtener matrices
        base_portfolio = Portfolio(assets=assets)
        
        # Obtener matriz de covarianza y retornos esperados
        # MAPEO: Esto reemplaza tu código:
        # rendimientos_esperados = df_precios.pct_change().mean()
        # matriz_covarianza = df_precios.pct_change().cov()
        cov_matrix = base_portfolio.covariance_matrix.values
        expected_returns = np.array([
            asset.mean_daily_return() for asset in assets
        ])
        
        ticker_symbols = [str(asset.ticker) for asset in assets]
        n_assets = len(assets)
        
        frontier_points = []
        
        # Simulación Monte Carlo
        # MAPEO: for _ in range(num_simulaciones):
        for _ in range(n_simulations):
            # Generar pesos aleatorios
            # MAPEO: pesos = np.random.random(len(df_precios.columns))
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)  # Normalizar
            
            # Calcular retorno esperado de la cartera
            # MAPEO: rendimiento_simulado = np.sum(pesos * rendimientos_esperados)
            portfolio_return = float(np.sum(weights * expected_returns))
            
            # Calcular volatilidad de la cartera
            # MAPEO: riesgo_simulado = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = float(np.sqrt(portfolio_variance))
            
            # Calcular Sharpe ratio
            # MAPEO: Esto no lo calculabas explícitamente en tu código original
            if portfolio_volatility > 0:
                sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
            else:
                sharpe = 0.0
            
            # Crear punto de la frontera
            # MAPEO: Guardabas en listas separadas (rendimientos_simulados, riesgos_simulados)
            point = PortfolioPoint(
                weights={ticker: float(w) for ticker, w in zip(ticker_symbols, weights)},
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe
            )
            
            frontier_points.append(point)
        
        return frontier_points


# EJEMPLO DE USO:
"""
# Setup
from infrastructure.repositories import YFinanceAssetRepository

repository = YFinanceAssetRepository()
use_case = CalculateEfficientFrontierUseCase(repository)

# Ejecutar simulación
request = CalculateEfficientFrontierRequest(
    ticker_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    start_date="2020-01-01",
    end_date="2024-11-30",
    n_simulations=10000
)

response = use_case.execute(request)

if response.success:
    print(f"✅ Simulación completada: {response.statistics['n_simulations']} carteras")
    print(f"\n📊 Rangos:")
    print(f"  Retorno: {response.statistics['return_range']}")
    print(f"  Volatilidad: {response.statistics['volatility_range']}")
    print(f"  Sharpe: {response.statistics['sharpe_range']}")
    
    print(f"\n🎯 CARTERA DE MÍNIMA VOLATILIDAD:")
    min_vol = response.optimal_portfolios['minimum_volatility']
    print(f"  Volatilidad: {min_vol.volatility:.4f}")
    print(f"  Retorno: {min_vol.expected_return:.4f}")
    print(f"  Sharpe: {min_vol.sharpe_ratio:.3f}")
    print(f"  Pesos: {min_vol.weights}")
    
    print(f"\n⭐ CARTERA DE MÁXIMO SHARPE:")
    max_sharpe = response.optimal_portfolios['maximum_sharpe']
    print(f"  Sharpe: {max_sharpe.sharpe_ratio:.3f}")
    print(f"  Retorno: {max_sharpe.expected_return:.4f}")
    print(f"  Volatilidad: {max_sharpe.volatility:.4f}")
    print(f"  Pesos: {max_sharpe.weights}")
    
    # Convertir a DataFrame para análisis
    df = response.to_dataframe()
    print(f"\n📈 DataFrame shape: {df.shape}")
    print(df.head())
    
    # Graficar (esto iría en Presentation Layer)
    import matplotlib.pyplot as plt
    plt.scatter(df['volatility'], df['return'], c=df['sharpe'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatilidad')
    plt.ylabel('Retorno Esperado')
    plt.title('Frontera Eficiente')
    plt.show()
    
else:
    print(f"❌ Error: {response.error_message}")
"""


# COMPARACIÓN CON TU CÓDIGO ORIGINAL:
"""
# ❌ ANTES (tu código - disperso, ~40 líneas):
num_simulaciones = 10000
rendimientos_simulados = []
riesgos_simulados = []

for _ in range(num_simulaciones):
    pesos = np.random.random(len(df_precios.columns))
    pesos /= np.sum(pesos)
    
    rendimiento_simulado = np.sum(pesos * rendimientos_esperados)
    riesgo_simulado = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
    
    rendimientos_simulados.append(rendimiento_simulado)
    riesgos_simulados.append(riesgo_simulado)

rendimientos_simulados = np.array(rendimientos_simulados)
riesgos_simulados = np.array(riesgos_simulados)

plt.scatter(riesgos_simulados, rendimientos_simulados, ...)
plt.show()

# Problemas:
# - Difícil de testear (no puedes mockear yfinance)
# - No reutilizable
# - Mezcla lógica y presentación
# - No identifica carteras óptimas automáticamente


# ✅ AHORA (Clean Architecture):
request = CalculateEfficientFrontierRequest(
    ticker_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    start_date="2020-01-01",
    end_date="2024-11-30",
    n_simulations=10000
)

response = use_case.execute(request)

# Ventajas:
# - Testeable (podemos inyectar mock repository)
# - Reutilizable (es un Use Case, no un script)
# - Separación clara (lógica vs presentación)
# - Identifica automáticamente carteras óptimas
# - Retorna datos estructurados listos para usar
# - Validación de entrada
# - Manejo de errores robusto
"""


# PRINCIPIOS CLEAN CODE APLICADOS:
# =================================
#
# 1. ENCAPSULATION: Lógica de simulación encapsulada en método privado
#
# 2. VALUE OBJECTS: PortfolioPoint encapsula datos de un punto
#
# 3. DTO PATTERN: Request/Response para interfaz limpia
#
# 4. SINGLE RESPONSIBILITY: Use Case orquesta, no implementa matemáticas
#
# 5. CONVENIENCE METHODS: to_dataframe() para compatibilidad
#
# 6. MEANINGFUL NAMES: _simulate_portfolios() claro y descriptivo
#
# 7. VALIDATION: Centralizada en request.validate()