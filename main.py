#!/usr/bin/env python3
"""
Main - Portfolio Analyzer

Script principal que reemplaza TODO tu código original.
"""

import sys
import os

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from use_cases import (
    AnalyzeAssetUseCase, AnalyzeAssetRequest,
    OptimizePortfolioUseCase, OptimizePortfolioRequest, OptimizationStrategy
)
from infrastructure import YFinanceAssetRepository


def analyze_example():
    """Análisis de activo - Reemplaza líneas 28-114 de tu código."""
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE ACTIVO")
    print("="*60 + "\n")
    
    repository = YFinanceAssetRepository()
    use_case = AnalyzeAssetUseCase(repository)
    
    request = AnalyzeAssetRequest(
        ticker_symbol="META",
        start_date="2020-01-01",
        end_date="2024-11-30"
    )
    
    print("⏳ Analizando...")
    response = use_case.execute(request)
    
    if response.success:
        s = response.statistics
        print(f"✅ {s['ticker']}")
        print(f"   Retorno: {s['annualized_return']}")
        print(f"   Sharpe: {s['sharpe_ratio']:.3f}")
    else:
        print(f"❌ {response.error_message}")


def optimize_example():
    """Optimización - Reemplaza líneas 584-701 de tu código."""
    print("\n" + "="*60)
    print("🎯 OPTIMIZACIÓN DE CARTERA")
    print("="*60 + "\n")
    
    repository = YFinanceAssetRepository()
    use_case = OptimizePortfolioUseCase(repository)
    
    request = OptimizePortfolioRequest(
        ticker_symbols=["JPM", "JNJ", "DIS", "META", "V"],
        start_date="2020-01-01",
        end_date="2024-11-30",
        strategy=OptimizationStrategy.MINIMUM_VARIANCE
    )
    
    print("⏳ Optimizando...")
    response = use_case.execute(request)
    
    if response.success:
        print(f"✅ Optimizado")
        print(f"\n   Original Sharpe: {response.original_portfolio['sharpe_ratio']:.3f}")
        print(f"   Optimizado Sharpe: {response.optimized_portfolio['sharpe_ratio']:.3f}")
        print(f"\n   Pesos optimizados:")
        for t, w in response.optimized_portfolio['weights'].items():
            print(f"   {t}: {w*100:.1f}%")
    else:
        print(f"❌ {response.error_message}")


if __name__ == "__main__":
    print("\n🎯 PORTFOLIO ANALYZER\n")
    
    try:
        analyze_example()
        optimize_example()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60 + "\n")
