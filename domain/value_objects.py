"""
Value Objects del dominio financiero.

CONCEPTO CLEAN CODE - Value Objects:
------------------------------------
Un Value Object es un objeto inmutable que se define por sus atributos, no por su identidad.
Dos Value Objects con los mismos valores son intercambiables.

Principios aplicados:
- Inmutabilidad: Una vez creado, no puede cambiar (frozen=True)
- Validación en construcción: Falla rápido si los datos son inválidos
- Comparación por valor: __eq__ basado en atributos, no en identidad
- Sin efectos secundarios: Métodos puros que no modifican estado

Ejemplo: Ticker("AAPL") == Ticker("AAPL") -> True (mismo valor)
         Pero son objetos diferentes en memoria
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)  # frozen=True hace la clase inmutable
class Ticker:
    """
    Representa el símbolo de un activo financiero.
    
    RESPONSABILIDAD ÚNICA (SRP):
    - Validar que el ticker sea un string válido
    - Normalizar el formato (mayúsculas, sin espacios)
    
    Attributes:
        symbol: Símbolo del ticker (ej: "AAPL", "IBB", "TSLA")
    
    Raises:
        ValueError: Si el símbolo es inválido (vacío, None, caracteres extraños)
    
    Examples:
        >>> ticker = Ticker("aapl")
        >>> ticker.symbol
        'AAPL'
        
        >>> Ticker("")  # Falla en construcción
        ValueError: Ticker symbol cannot be empty
    """
    
    symbol: str
    
    def __post_init__(self) -> None:
        """
        Validación post-inicialización.
        
        CONCEPTO: "Fail Fast" - Detectar errores lo antes posible.
        Si el ticker es inválido, fallamos inmediatamente en construcción,
        no esperamos a que se use más adelante.
        """
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Ticker symbol cannot be empty")
        
        # Normalizar: mayúsculas y sin espacios
        # Usamos object.__setattr__ porque la clase es frozen
        object.__setattr__(self, 'symbol', self.symbol.strip().upper())
        
        # Validación básica de caracteres permitidos
        if not all(c.isalnum() or c in '.-' for c in self.symbol):
            raise ValueError(
                f"Ticker symbol '{self.symbol}' contains invalid characters. "
                "Only alphanumeric and '.-' allowed."
            )
    
    def __str__(self) -> str:
        """Representación en string legible para humanos."""
        return self.symbol
    
    def __repr__(self) -> str:
        """Representación técnica para debugging."""
        return f"Ticker('{self.symbol}')"


@dataclass(frozen=True)
class DateRange:
    """
    Representa un rango de fechas para análisis financiero.
    
    RESPONSABILIDAD ÚNICA (SRP):
    - Validar que el rango de fechas sea coherente
    - Proveer operaciones útiles sobre el rango
    
    CONCEPTO: Encapsular lógica de negocio relacionada con fechas.
    En lugar de tener start_date y end_date flotando como strings,
    tenemos un objeto que entiende el concepto de "rango" y puede validarse.
    """
    
    start: str  # Formato: "YYYY-MM-DD"
    end: str    # Formato: "YYYY-MM-DD"
    
    def __post_init__(self) -> None:
        """Validar que el rango sea coherente."""
        from datetime import datetime
        
        # Validar formato de fechas
        try:
            start_dt = datetime.strptime(self.start, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(
                f"Invalid date format. Expected YYYY-MM-DD. Error: {e}"
            )
        
        # Validar que start <= end
        if start_dt > end_dt:
            raise ValueError(
                f"Start date ({self.start}) must be before or equal to end date ({self.end})"
            )
    
    def days_between(self) -> int:
        """
        Calcula el número de días en el rango.
        
        CONCEPTO: Método puro - no modifica estado, siempre retorna
        el mismo resultado para los mismos inputs.
        """
        from datetime import datetime
        start_dt = datetime.strptime(self.start, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end, "%Y-%m-%d")
        return (end_dt - start_dt).days
    
    def __str__(self) -> str:
        return f"{self.start} to {self.end}"


@dataclass(frozen=True)
class Money:
    """
    Representa un valor monetario con su moneda.
    
    CONCEPTO: Evitar "Primitive Obsession" - usar float para dinero es peligroso.
    Este Value Object encapsula dinero con su moneda y maneja precisión.
    
    Attributes:
        amount: Cantidad de dinero
        currency: Código de moneda (USD, EUR, etc.)
    """
    
    amount: float
    currency: str = "USD"
    
    def __post_init__(self) -> None:
        """Validar valores monetarios."""
        if self.amount < 0:
            raise ValueError("Money amount cannot be negative")
        
        # Normalizar moneda
        object.__setattr__(self, 'currency', self.currency.upper())
        
        # Redondear a 2 decimales para evitar problemas de precisión
        object.__setattr__(self, 'amount', round(self.amount, 2))
    
    def __add__(self, other: 'Money') -> 'Money':
        """
        Suma de valores monetarios.
        
        CONCEPTO: Operaciones intuitivas - los Value Objects se comportan
        como tipos primitivos pero con validación de negocio.
        """
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot add different currencies: {self.currency} and {other.currency}"
            )
        return Money(self.amount + other.amount, self.currency)
    
    def __mul__(self, factor: float) -> 'Money':
        """Multiplicación por un factor."""
        return Money(self.amount * factor, self.currency)
    
    def __str__(self) -> str:
        return f"{self.currency} {self.amount:,.2f}"


@dataclass(frozen=True)
class Percentage:
    """
    Representa un porcentaje con validación.
    
    CONCEPTO: Type Safety - en lugar de usar float 0.05 para 5%,
    usamos Percentage(5.0) y el objeto se encarga de la conversión.
    
    Attributes:
        value: Valor del porcentaje (5.0 para 5%)
    """
    
    value: float
    
    def __post_init__(self) -> None:
        """Validar rango razonable de porcentajes."""
        if not -100 <= self.value <= 1000:
            raise ValueError(
                f"Percentage {self.value}% is outside reasonable range [-100%, 1000%]"
            )
    
    def as_decimal(self) -> float:
        """Convierte a decimal (5% -> 0.05)."""
        return self.value / 100.0
    
    def __str__(self) -> str:
        return f"{self.value:.2f}%"
    
    def __repr__(self) -> str:
        return f"Percentage({self.value})"