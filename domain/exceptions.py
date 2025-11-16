"""
Excepciones del dominio financiero.

CONCEPTO CLEAN CODE - Domain Exceptions:
-----------------------------------------
Las excepciones de dominio comunican errores de negocio, no errores técnicos.

Principios aplicados:
- Nombrado expresivo: El nombre de la excepción explica QUÉ salió mal
- Jerarquía clara: Todas heredan de una base común para facilitar el catch
- Mensajes informativos: Contexto suficiente para debugging
- Separación de concerns: Errores de dominio != errores de infraestructura

Ejemplo de uso:
    try:
        portfolio.add_asset(asset, weight=-0.5)
    except InvalidWeightError as e:
        print(f"Error de negocio: {e}")
        # Aquí sabemos que es un error de REGLA DE NEGOCIO, no técnico
"""

from typing import Optional


class DomainException(Exception):
    """
    Excepción base para todos los errores de dominio.
    
    CONCEPTO: Base exception pattern - permite capturar todas las
    excepciones de dominio con un solo catch si es necesario.
    
    VENTAJA: Separación clara entre errores de negocio y errores técnicos
    (ValueError, TypeError, etc.)
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Args:
            message: Descripción del error
            details: Información adicional contextual (opcional)
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Representación legible del error con detalles."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message


class InvalidTickerError(DomainException):
    """
    Se lanza cuando un ticker es inválido según reglas de negocio.
    
    CONCEPTO: Explicit over implicit - en lugar de ValueError genérico,
    tenemos un error específico que comunica exactamente qué regla de negocio
    se violó.
    
    Example:
        >>> ticker = Ticker("")
        InvalidTickerError: Ticker cannot be empty
    """
    pass


class InvalidWeightError(DomainException):
    """
    Se lanza cuando el peso de un activo en la cartera es inválido.
    
    Reglas de negocio:
    - Peso debe estar entre 0 y 1 (0% a 100%)
    - La suma de todos los pesos debe ser 1 (100%)
    """
    pass


class InvalidDateRangeError(DomainException):
    """
    Se lanza cuando un rango de fechas es inválido.
    
    Reglas de negocio:
    - Fecha de inicio debe ser anterior a fecha fin
    - Fechas deben estar en formato válido
    - El rango debe tener sentido financiero (mínimo 1 día)
    """
    pass


class InsufficientDataError(DomainException):
    """
    Se lanza cuando no hay suficientes datos para realizar un cálculo.
    
    CONCEPTO: Fail Fast - mejor fallar explícitamente que retornar
    resultados incorrectos o NaN silenciosamente.
    
    Example:
        >>> asset.calculate_volatility()  # Solo 2 días de datos
        InsufficientDataError: Need at least 30 days for volatility calculation
    """
    pass


class OptimizationError(DomainException):
    """
    Se lanza cuando la optimización de cartera falla.
    
    Posibles causas:
    - No converge el algoritmo
    - Restricciones inconsistentes
    - Matriz de covarianza singular
    """
    pass


class PortfolioValidationError(DomainException):
    """
    Se lanza cuando una cartera no cumple las reglas de negocio.
    
    Reglas de negocio:
    - Debe tener al menos 1 activo
    - Los pesos deben sumar 1.0 (100%)
    - No puede tener pesos negativos (sin short selling)
    """
    pass


class DataFetchError(DomainException):
    """
    Se lanza cuando falla la descarga de datos financieros.
    
    NOTA: Esta es frontera entre dominio e infraestructura.
    La incluimos en dominio porque "no poder obtener datos" afecta
    las reglas de negocio (no podemos analizar sin datos).
    """
    pass


# Ejemplo de uso en código:
"""
# En lugar de esto (genérico, poco informativo):
if weight < 0:
    raise ValueError("Invalid weight")

# Hacemos esto (específico, comunicativo):
if weight < 0:
    raise InvalidWeightError(
        message=f"Weight must be positive, got {weight}",
        details={"weight": weight, "ticker": ticker.symbol}
    )

# VENTAJAS:
# 1. El código que llama sabe exactamente qué tipo de error ocurrió
# 2. Puede decidir cómo manejarlo según el tipo
# 3. Los mensajes son consistentes y claros
# 4. Facilita el debugging con detalles estructurados
"""