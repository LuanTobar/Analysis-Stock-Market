# 📊 Portfolio Analyzer - Clean Architecture

Análisis y optimización de carteras de inversión implementado con **Clean Architecture**.

Transformación de un script de 900 líneas en una arquitectura empresarial modular, testeable y mantenible.

---

## 🏗️ Arquitectura
```
portfolio_analyzer/
│
├── domain/                    # Capa de Negocio
│   ├── entities/
│   │   ├── asset.py          # Entidad Asset
│   │   └── portfolio.py      # Entidad Portfolio
│   ├── value_objects.py      # Ticker, DateRange, Money, Percentage
│   └── exceptions.py         # Excepciones de dominio
│
├── use_cases/                 # Capa de Aplicación
│   ├── analyze_asset.py      # UC: Analizar activo
│   ├── optimize_portfolio.py # UC: Optimizar cartera
│   ├── calculate_efficient_frontier.py  # UC: Frontera eficiente
│   └── interfaces.py         # Contratos/Puertos
│
├── infrastructure/            # Capa de Infraestructura
│   ├── data_sources/
│   │   └── yfinance_source.py    # Descarga de Yahoo Finance
│   └── repositories/
│       └── asset_repository.py   # Repository Pattern
│
├── presentation/              # Capa de Presentación
│   └── cli/
│       └── main.py           # Command Line Interface
│
├── config.py                 # Configuración
├── main.py                   # Script principal
└── requirements.txt          # Dependencias
```

---

## 🚀 Instalación

### Requisitos
- Python 3.11 o 3.12 (recomendado)
- pip

### Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Uso

### Opción 1: Script Rápido
```bash
python main.py
```

Ejecuta ejemplos predefinidos:
- Análisis de AAPL
- Optimización de cartera (IBB, MNR, SMH, LIT, EEM)

---

### Opción 2: CLI Completa

#### Analizar un activo
```bash
python -m presentation.cli.main analyze AAPL

# Con parámetros
python -m presentation.cli.main analyze TSLA \
    --start 2020-01-01 \
    --end 2024-11-30 \
    --risk-free-rate 0.03
```

#### Optimizar cartera
```bash
python -m presentation.cli.main optimize AAPL,GOOGL,MSFT,TSLA,NVDA

# Con estrategia
python -m presentation.cli.main optimize JPM,JNJ,DIS,META,V \
    --strategy minimum_variance \
    --start 2020-01-01
```

#### Frontera eficiente
```bash
python -m presentation.cli.main frontier AAPL,GOOGL,MSFT \
    --simulations 10000 \
    --plot
```

**Flag `--plot`**: Genera gráfico con matplotlib

---

### Opción 3: Como Librería
```python
from infrastructure import YFinanceAssetRepository
from use_cases import (
    AnalyzeAssetUseCase, 
    AnalyzeAssetRequest,
    OptimizePortfolioUseCase,
    OptimizePortfolioRequest,
    OptimizationStrategy
)

# Analizar activo
repository = YFinanceAssetRepository()
use_case = AnalyzeAssetUseCase(repository)
request = AnalyzeAssetRequest("AAPL", "2020-01-01", "2024-11-30")
response = use_case.execute(request)

print(response.statistics['sharpe_ratio'])

# Optimizar cartera
optimize_uc = OptimizePortfolioUseCase(repository)
request = OptimizePortfolioRequest(
    ticker_symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2020-01-01",
    end_date="2024-11-30",
    strategy=OptimizationStrategy.MINIMUM_VARIANCE
)
response = optimize_uc.execute(request)

print(response.optimized_portfolio['weights'])
```

---

## 📊 Características

### Análisis de Activos
- Retorno anualizado
- Volatilidad anualizada
- Sharpe Ratio
- Maximum Drawdown
- Precios históricos

### Optimización de Cartera
- **Minimum Variance**: Minimiza volatilidad
- **Target Volatility**: Alcanza volatilidad objetivo
- Comparación cartera original vs optimizada
- Cálculo de mejoras (Sharpe, volatilidad)

### Frontera Eficiente
- Simulación Monte Carlo (10,000 carteras)
- Identificación de cartera óptima (máximo Sharpe)
- Identificación de mínima volatilidad
- Visualización con matplotlib

---

## 🎓 Principios Aplicados

### Clean Code
✅ Meaningful Names  
✅ Small Functions  
✅ Single Responsibility  
✅ DRY (Don't Repeat Yourself)  
✅ Error Handling  

### SOLID
✅ **S**ingle Responsibility Principle  
✅ **O**pen/Closed Principle  
✅ **L**iskov Substitution Principle  
✅ **I**nterface Segregation Principle  
✅ **D**ependency Inversion Principle  

### Design Patterns
✅ Repository Pattern  
✅ DTO Pattern  
✅ Strategy Pattern  
✅ Command Pattern  
✅ Factory Method  
✅ Adapter Pattern  

### Clean Architecture
✅ 4 capas independientes  
✅ Dependency Rule  
✅ Testeable sin APIs externas  
✅ Framework independiente  

---

## 🧪 Testing (Próximamente)
```bash
pytest tests/ -v --cov=domain --cov=use_cases
```

---

## 📈 Ejemplos de Salida

### Análisis de Activo
```
============================================================
📊 ANALIZANDO: AAPL
============================================================

✅ Análisis: AAPL

📅 Período: 2020-01-01 a 2024-11-30
   Datos: 1234 días

📊 Retorno anualizado: 32.5%
   Volatilidad: 28.3%
   Sharpe Ratio: 1.123
   Max Drawdown: -25.4%

💰 Precio inicial: $73.41
   Precio final: $189.95
```

### Optimización
```
✅ Optimización completada

📊 ORIGINAL (Pesos iguales):
   Sharpe: 0.856
   Volatilidad: 22.5%
   AAPL: 20.0%
   GOOGL: 20.0%
   MSFT: 20.0%
   TSLA: 20.0%
   NVDA: 20.0%

✨ OPTIMIZADA:
   Sharpe: 1.234
   Volatilidad: 18.2%
   AAPL: 25.3%
   GOOGL: 18.7%
   MSFT: 32.1%
   TSLA: 12.4%
   NVDA: 11.5%

📈 Mejora Sharpe: +0.378 (+44.2%)
```

---

## 🔧 Configuración

Editar `config.py`:
```python
TRADING_DAYS_PER_YEAR = 252
MIN_DATA_POINTS = 30
DEFAULT_RISK_FREE_RATE = 0.02
OPTIMIZATION_MAX_ITER = 1000
```

---

## 📚 Dependencias

- **yfinance**: Descarga datos de Yahoo Finance
- **pandas**: Manipulación de datos
- **numpy**: Cálculos numéricos
- **scipy**: Optimización matemática
- **matplotlib**: Visualización

---


### Agregar nueva estrategia de optimización

1. Editar `use_cases/optimize_portfolio.py`
2. Agregar a `OptimizationStrategy` enum
3. Implementar en `Portfolio.optimize_weights_*()`

---

---

## 🎯 Roadmap

- [ ] Tests unitarios (pytest)
- [ ] CI/CD (GitHub Actions)
- [ ] Web API (FastAPI)
- [ ] Dashboard (Streamlit)
- [ ] Caché de datos (Redis)
- [ ] Base de datos (PostgreSQL)
- [ ] Docker container

---

## 👨‍💻 Autor

**Luan Tobar**
- GitHub: [@LuanTobar](https://github.com/LuanTobar)

---

## 📄 Licencia

MIT License

---


