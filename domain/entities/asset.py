"""
Entidad Asset - Representa un activo financiero individual.

CONCEPTO CLEAN CODE - Rich Domain Model:
---------------------------------------
La entidad Asset encapsula toda la lógica de negocio relacionada con
un activo financiero. No es un simple contenedor de datos (anemic model),
sino que tiene comportamiento y puede ejecutar sus propias reglas de negocio.

Principios SOLID aplicados:
- SRP: Asset solo se encarga de cálculos sobre SUS propios datos
- OCP: Abierto a extensión (podemos heredar), cerrado a modificación
- DIP: Depende de abstracciones (Ticker, Percentage) no de concreciones
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from domain.value_objects import Ticker, Percentage
from domain.exceptions import InsufficientDataError


@dataclass
class Asset:
    """
    Representa un activo financiero con su historial de precios y métricas.
    
    Esta entidad NO sabe cómo obtener datos de fuentes externas.
    Solo sabe qué hacer CON los datos una vez que los tiene.
    (Principio de Dependency Inversion)
    """
    
    ticker: Ticker
    data: pd.DataFrame  # Debe tener columna 'Adj Close'
    name: Optional[str] = None
    
    TRADING_DAYS_PER_YEAR: int = field(default=252, init=False)
    MIN_DATA_POINTS: int = field(default=30, init=False)
    
    def __post_init__(self) -> None:
        """Validación inmediata del estado del objeto (Fail Fast)."""
        if self.data.empty:
            raise InsufficientDataError(
                message=f"Asset {self.ticker} has no price data",
                details={"ticker": str(self.ticker)}
            )
        
        if 'Adj Close' not in self.data.columns:
            raise ValueError(f"Data must contain 'Adj Close' column")
        
        self.data = self.data.dropna(subset=['Adj Close'])
        
        if self.name is None:
            self.name = str(self.ticker)
    
    @property
    def prices(self) -> pd.Series:
        """Precios ajustados del activo."""
        return self.data['Adj Close']
    
    @property
    def daily_returns(self) -> pd.Series:
        """Retornos diarios calculados."""
        return self.prices.pct_change().dropna()
    
    def mean_daily_return(self) -> float:
        """Retorno promedio diario."""
        self._validate_sufficient_data()
        return float(self.daily_returns.mean())
    
    def daily_volatility(self) -> float:
        """Volatilidad diaria (std deviation)."""
        self._validate_sufficient_data()
        return float(self.daily_returns.std())
    
    def annualized_return(self) -> Percentage:
        """Retorno anualizado."""
        mean_return = self.mean_daily_return()
        annual = (1 + mean_return) ** self.TRADING_DAYS_PER_YEAR - 1
        return Percentage(annual * 100)
    
    def annualized_volatility(self) -> Percentage:
        """Volatilidad anualizada."""
        daily_vol = self.daily_volatility()
        annual_vol = daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return Percentage(annual_vol * 100)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Ratio de Sharpe."""
        ret = self.annualized_return().as_decimal()
        vol = self.annualized_volatility().as_decimal()
        return (ret - risk_free_rate) / vol if vol != 0 else 0.0
    
    def max_drawdown(self) -> Percentage:
        """Máxima caída desde un pico."""
        cum_returns = (1 + self.daily_returns).cumprod() - 1
        running_max = cum_returns.cummax()
        drawdown = cum_returns - running_max
        return Percentage(float(drawdown.min()) * 100)
    
    def get_summary_statistics(self) -> dict:
        """Estadísticas completas del activo."""
        try:
            return {
                'ticker': str(self.ticker),
                'name': self.name,
                'data_points': len(self.data),
                'start_date': self.data.index[0].strftime('%Y-%m-%d'),
                'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
                'annualized_return': str(self.annualized_return()),
                'annualized_volatility': str(self.annualized_volatility()),
                'sharpe_ratio': round(self.sharpe_ratio(), 3),
                'max_drawdown': str(self.max_drawdown())
            }
        except InsufficientDataError:
            return {
                'ticker': str(self.ticker),
                'error': 'Insufficient data'
            }
    
    def _validate_sufficient_data(self) -> None:
        """Validación interna de datos suficientes."""
        if len(self.daily_returns) < self.MIN_DATA_POINTS:
            raise InsufficientDataError(
                message=f"Need at least {self.MIN_DATA_POINTS} days",
                details={"ticker": str(self.ticker)}
            )
    
    def __eq__(self, other: object) -> bool:
        """Dos assets son iguales si tienen el mismo ticker."""
        if not isinstance(other, Asset):
            return NotImplemented
        return self.ticker == other.ticker
    
    def __str__(self) -> str:
        return f"Asset({self.ticker}: {len(self.data)} days)"