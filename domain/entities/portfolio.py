"""
Entidad Portfolio - Representa una cartera de inversión.

CONCEPTO CLEAN CODE - Aggregate Root:
-------------------------------------
Portfolio es un "Aggregate Root" en DDD (Domain-Driven Design).
Es responsable de mantener la consistencia de todos los activos
que contiene y sus pesos.

Reglas de negocio invariantes:
- La suma de pesos debe ser siempre 1.0 (100%)
- No puede haber pesos negativos
- Debe tener al menos 1 activo
"""

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from domain.entities.asset import Asset
from domain.value_objects import Ticker, Percentage
from domain.exceptions import (
    PortfolioValidationError, 
    InvalidWeightError,
    OptimizationError
)


@dataclass
class Portfolio:
    """
    Entidad que representa una cartera de inversión con múltiples activos.
    
    RESPONSABILIDAD ÚNICA:
    - Gestionar la colección de activos y sus pesos
    - Calcular métricas agregadas de la cartera
    - Optimizar la asignación de pesos
    - Mantener las invariantes de negocio
    
    Attributes:
        assets: Lista de activos en la cartera
        weights: Dict con pesos de cada activo (ticker -> peso)
        name: Nombre de la cartera (opcional)
    """
    
    assets: List[Asset]
    weights: Dict[Ticker, float] = field(default_factory=dict)
    name: str = "Portfolio"
    
    TOLERANCE: float = field(default=1e-6, init=False)  # Para comparaciones float
    
    def __post_init__(self) -> None:
        """
        Validación y normalización inicial de la cartera.
        
        CONCEPTO: Constructor honesto - garantiza invariantes desde la creación.
        """
        if not self.assets:
            raise PortfolioValidationError(
                message="Portfolio must contain at least one asset"
            )
        
        # Si no se proporcionaron pesos, asignar pesos iguales
        if not self.weights:
            equal_weight = 1.0 / len(self.assets)
            self.weights = {asset.ticker: equal_weight for asset in self.assets}
        
        # Validar la cartera
        self._validate_portfolio()
    
    def _validate_portfolio(self) -> None:
        """
        Valida las invariantes de negocio de la cartera.
        
        CONCEPTO: Métodos privados para encapsular lógica interna.
        El underscore indica "esto es interno, no lo llames desde fuera".
        """
        # Validar que todos los activos tengan peso asignado
        asset_tickers = {asset.ticker for asset in self.assets}
        weight_tickers = set(self.weights.keys())
        
        if asset_tickers != weight_tickers:
            missing = asset_tickers - weight_tickers
            extra = weight_tickers - asset_tickers
            raise PortfolioValidationError(
                message="Mismatch between assets and weights",
                details={"missing_weights": list(missing), "extra_weights": list(extra)}
            )
        
        # Validar que no haya pesos negativos
        for ticker, weight in self.weights.items():
            if weight < 0:
                raise InvalidWeightError(
                    message=f"Weight cannot be negative",
                    details={"ticker": str(ticker), "weight": weight}
                )
        
        # Validar que los pesos sumen 1.0 (con tolerancia para errores float)
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > self.TOLERANCE:
            raise InvalidWeightError(
                message=f"Weights must sum to 1.0, got {total_weight:.6f}",
                details={"total_weight": total_weight, "weights": dict(self.weights)}
            )
    
    @property
    def returns_matrix(self) -> pd.DataFrame:
        """
        Matriz de retornos diarios de todos los activos.
        
        CONCEPTO: Property para lazy computation.
        Solo se calcula cuando se necesita.
        
        Returns:
            DataFrame con retornos diarios (fechas x activos)
        """
        returns_dict = {
            str(asset.ticker): asset.daily_returns 
            for asset in self.assets
        }
        return pd.DataFrame(returns_dict)
    
    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """
        Matriz de covarianza de los retornos.
        
        CONCEPTO: Encapsular cálculos complejos detrás de una propiedad simple.
        El usuario no necesita saber la fórmula de covarianza.
        """
        return self.returns_matrix.cov()
    
    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Matriz de correlación entre activos."""
        return self.returns_matrix.corr()
    
    def expected_return(self) -> float:
        """
        Retorno esperado de la cartera (media ponderada).
        
        Formula: Σ(peso_i * retorno_i)
        
        Returns:
            Retorno diario esperado de la cartera
        """
        weighted_returns = [
            self.weights[asset.ticker] * asset.mean_daily_return()
            for asset in self.assets
        ]
        return sum(weighted_returns)
    
    def portfolio_volatility(self) -> float:
        """
        Volatilidad de la cartera considerando correlaciones.
        
        Formula: sqrt(w^T * Σ * w)
        donde w es el vector de pesos y Σ la matriz de covarianza
        
        CONCEPTO: Método que encapsula álgebra lineal compleja.
        El usuario solo pide "dame la volatilidad" sin saber cómo se calcula.
        """
        # Ordenar pesos según orden de columnas en covariance matrix
        weight_vector = np.array([
            self.weights[Ticker(col)] 
            for col in self.covariance_matrix.columns
        ])
        
        # Cálculo matricial: w^T * Σ * w
        variance = np.dot(weight_vector, np.dot(self.covariance_matrix.values, weight_vector))
        return float(np.sqrt(variance))
    
    def annualized_return(self) -> Percentage:
        """Retorno anualizado de la cartera."""
        daily_return = self.expected_return()
        annual = (1 + daily_return) ** 252 - 1
        return Percentage(annual * 100)
    
    def annualized_volatility(self) -> Percentage:
        """Volatilidad anualizada de la cartera."""
        daily_vol = self.portfolio_volatility()
        annual_vol = daily_vol * np.sqrt(252)
        return Percentage(annual_vol * 100)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Ratio de Sharpe de la cartera."""
        ret = self.annualized_return().as_decimal()
        vol = self.annualized_volatility().as_decimal()
        return (ret - risk_free_rate) / vol if vol != 0 else 0.0
    
    def optimize_weights_minimum_variance(self) -> 'Portfolio':
        """
        Optimiza los pesos para minimizar la varianza (riesgo).
        
        CONCEPTO: Factory Method - retorna una NUEVA Portfolio optimizada,
        no modifica la actual (inmutabilidad funcional).
        
        Returns:
            Nueva instancia de Portfolio con pesos optimizados
        
        Raises:
            OptimizationError: Si la optimización falla
        
        Example:
            >>> original = Portfolio(assets)
            >>> optimized = original.optimize_weights_minimum_variance()
            >>> print(optimized.weights)
        """
        def objective(weights: np.ndarray) -> float:
            """Función objetivo: varianza de la cartera."""
            variance = np.dot(weights, np.dot(self.covariance_matrix.values, weights))
            return variance
        
        def constraint_sum_to_one(weights: np.ndarray) -> float:
            """Restricción: los pesos deben sumar 1."""
            return np.sum(weights) - 1.0
        
        # Configuración de la optimización
        n_assets = len(self.assets)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        bounds = [(0.0, 1.0)] * n_assets  # Pesos entre 0 y 1
        constraints = {'type': 'eq', 'fun': constraint_sum_to_one}
        
        try:
            # Ejecutar optimización
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                raise OptimizationError(
                    message="Optimization failed to converge",
                    details={"reason": result.message}
                )
            
            # Crear dict de nuevos pesos
            optimized_weights = {
                asset.ticker: float(weight)
                for asset, weight in zip(self.assets, result.x)
            }
            
            # Retornar nueva Portfolio con pesos optimizados
            return Portfolio(
                assets=self.assets,
                weights=optimized_weights,
                name=f"{self.name} (Min Variance)"
            )
        
        except Exception as e:
            raise OptimizationError(
                message="Optimization process failed",
                details={"error": str(e)}
            )
    
    def optimize_weights_target_volatility(self, target_volatility: float) -> 'Portfolio':
        """
        Optimiza los pesos para alcanzar una volatilidad objetivo.
        
        CONCEPTO: Mismo patrón que el método anterior pero con objetivo diferente.
        Esto es Open/Closed Principle - podemos añadir nuevos métodos de optimización
        sin modificar los existentes.
        
        Args:
            target_volatility: Volatilidad diaria objetivo (ej: 0.016 = 1.6%)
        
        Returns:
            Nueva Portfolio con pesos que minimizan la distancia a la volatilidad objetivo
        """
        def objective(weights: np.ndarray) -> float:
            """Minimizar la diferencia absoluta con el target."""
            variance = np.dot(weights, np.dot(self.covariance_matrix.values, weights))
            volatility = np.sqrt(variance)
            return abs(volatility - target_volatility)
        
        def constraint_sum_to_one(weights: np.ndarray) -> float:
            return np.sum(weights) - 1.0
        
        n_assets = len(self.assets)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        bounds = [(0.0, 1.0)] * n_assets
        constraints = {'type': 'eq', 'fun': constraint_sum_to_one}
        
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                raise OptimizationError(
                    message="Optimization failed",
                    details={"reason": result.message}
                )
            
            optimized_weights = {
                asset.ticker: float(weight)
                for asset, weight in zip(self.assets, result.x)
            }
            
            return Portfolio(
                assets=self.assets,
                weights=optimized_weights,
                name=f"{self.name} (Target Vol {target_volatility:.4f})"
            )
        
        except Exception as e:
            raise OptimizationError(
                message="Optimization failed",
                details={"error": str(e)}
            )
    
    def get_summary(self) -> dict:
        """
        Resumen completo de la cartera.
        
        CONCEPTO: Facade method - proporciona una interfaz simple
        para obtener toda la información relevante.
        """
        return {
            'name': self.name,
            'n_assets': len(self.assets),
            'assets': [str(asset.ticker) for asset in self.assets],
            'weights': {str(k): round(v, 4) for k, v in self.weights.items()},
            'annualized_return': str(self.annualized_return()),
            'annualized_volatility': str(self.annualized_volatility()),
            'sharpe_ratio': round(self.sharpe_ratio(), 3),
            'expected_daily_return': round(self.expected_return(), 6)
        }
    
    def __str__(self) -> str:
        """Representación legible de la cartera."""
        return f"Portfolio({self.name}: {len(self.assets)} assets)"
    
    def __repr__(self) -> str:
        """Representación técnica para debugging."""
        return f"Portfolio(name={self.name!r}, assets={len(self.assets)})"


# RESUMEN DE CONCEPTOS CLEAN CODE EN PORTFOLIO:
# =============================================
#
# 1. AGGREGATE ROOT: Portfolio controla la consistencia de sus assets
#
# 2. INVARIANTES: _validate_portfolio() garantiza reglas de negocio
#
# 3. IMMUTABILITY: optimize_*() retorna NUEVA instancia, no modifica
#
# 4. ENCAPSULATION: Cálculos complejos ocultos detrás de métodos simples
#
# 5. SINGLE RESPONSIBILITY: Solo gestiona cartera, no descarga datos ni grafica
#
# 6. MEANINGFUL NAMES: optimize_weights_minimum_variance() es autoexplicativo
#
# 7. FAIL FAST: Validaciones en __post_init__
#
# 8. DRY: Lógica de optimización compartida, solo cambia objetivo
#
# 9. TESTABILITY: Cada método es independiente y testeable
#
# 10. BUSINESS LANGUAGE: Los métodos hablan el idioma del negocio financiero