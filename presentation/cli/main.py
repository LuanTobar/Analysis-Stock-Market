"""
CLI (Command Line Interface) para Portfolio Analyzer.

CONCEPTO: Presentation Layer - Interactúa con el usuario.
Convierte comandos en Requests y Responses en output legible.
"""

import argparse
import sys
from datetime import datetime

sys.path.append(r'c:\Users\Dentaldata1\Documents\Lucho Brainstorming\Analysis-Stock-Market-main\Analysis-Stock-Market-main')

from use_cases import (
    AnalyzeAssetUseCase, AnalyzeAssetRequest,
    OptimizePortfolioUseCase, OptimizePortfolioRequest, OptimizationStrategy,
    CalculateEfficientFrontierUseCase, CalculateEfficientFrontierRequest
)
from infrastructure import YFinanceAssetRepository


class PortfolioAnalyzerCLI:
    """CLI para Portfolio Analyzer."""
    
    def __init__(self):
        """Inicializa CLI con dependency injection."""
        self.repository = YFinanceAssetRepository()
        self.analyze_use_case = AnalyzeAssetUseCase(self.repository)
        self.optimize_use_case = OptimizePortfolioUseCase(self.repository)
        self.frontier_use_case = CalculateEfficientFrontierUseCase(self.repository)
    
    def analyze_command(self, args):
        """Analiza un activo individual."""
        print(f"\n{'='*60}")
        print(f"📊 ANALIZANDO: {args.ticker}")
        print(f"{'='*60}\n")
        
        request = AnalyzeAssetRequest(
            ticker_symbol=args.ticker,
            start_date=args.start,
            end_date=args.end,
            risk_free_rate=args.risk_free_rate
        )
        
        response = self.analyze_use_case.execute(request)
        
        if response.success:
            self._display_asset_analysis(response)
        else:
            print(f"❌ Error: {response.error_message}")
    
    def optimize_command(self, args):
        """Optimiza una cartera."""
        print(f"\n{'='*60}")
        print(f"🎯 OPTIMIZANDO CARTERA")
        print(f"{'='*60}\n")
        
        tickers = [t.strip() for t in args.tickers.split(',')]
        print(f"Activos: {', '.join(tickers)}\n")
        
        strategy_map = {
            'minimum_variance': OptimizationStrategy.MINIMUM_VARIANCE,
            'target_volatility': OptimizationStrategy.TARGET_VOLATILITY,
        }
        strategy = strategy_map.get(args.strategy, OptimizationStrategy.MINIMUM_VARIANCE)
        
        request = OptimizePortfolioRequest(
            ticker_symbols=tickers,
            start_date=args.start,
            end_date=args.end,
            strategy=strategy,
            target_volatility=args.target_volatility,
            risk_free_rate=args.risk_free_rate
        )
        
        print("⏳ Optimizando...")
        response = self.optimize_use_case.execute(request)
        
        if response.success:
            self._display_optimization(response)
        else:
            print(f"❌ Error: {response.error_message}")
    
    def frontier_command(self, args):
        """Calcula frontera eficiente."""
        print(f"\n{'='*60}")
        print(f"📈 FRONTERA EFICIENTE")
        print(f"{'='*60}\n")
        
        tickers = [t.strip() for t in args.tickers.split(',')]
        print(f"Activos: {', '.join(tickers)}")
        print(f"Simulaciones: {args.simulations:,}\n")
        
        request = CalculateEfficientFrontierRequest(
            ticker_symbols=tickers,
            start_date=args.start,
            end_date=args.end,
            n_simulations=args.simulations,
            risk_free_rate=args.risk_free_rate
        )
        
        print("⏳ Simulando...")
        response = self.frontier_use_case.execute(request)
        
        if response.success:
            self._display_frontier(response, args.plot)
        else:
            print(f"❌ Error: {response.error_message}")
    
    def _display_asset_analysis(self, response):
        """Muestra análisis de activo."""
        stats = response.statistics
        
        print(f"✅ Análisis: {stats['ticker']}\n")
        print(f"📅 Período: {stats['start_date']} a {stats['end_date']}")
        print(f"   Datos: {stats['data_points']} días\n")
        print(f"📊 Retorno anualizado: {stats['annualized_return']}")
        print(f"   Volatilidad: {stats['annualized_volatility']}")
        print(f"   Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {stats['max_drawdown']}\n")
        print(f"💰 Precio inicial: ${stats['initial_price']:.2f}")
        print(f"   Precio final: ${stats['current_price']:.2f}")
        print(f"\n{'='*60}\n")
    
    def _display_optimization(self, response):
        """Muestra optimización."""
        print(f"✅ Optimización completada\n")
        
        print(f"📊 ORIGINAL (Pesos iguales):")
        orig = response.original_portfolio
        print(f"   Sharpe: {orig['sharpe_ratio']:.3f}")
        print(f"   Volatilidad: {orig['annualized_volatility']}")
        for ticker, weight in orig['weights'].items():
            print(f"   {ticker}: {weight*100:.1f}%")
        
        print(f"\n✨ OPTIMIZADA:")
        opt = response.optimized_portfolio
        print(f"   Sharpe: {opt['sharpe_ratio']:.3f}")
        print(f"   Volatilidad: {opt['annualized_volatility']}")
        for ticker, weight in opt['weights'].items():
            print(f"   {ticker}: {weight*100:.1f}%")
        
        print(f"\n📈 Mejora Sharpe: {response.improvement['sharpe_improvement']:+.3f}")
        print(f"\n{'='*60}\n")
    
    def _display_frontier(self, response, should_plot):
        """Muestra frontera eficiente."""
        stats = response.statistics
        
        print(f"✅ Simulación completada\n")
        print(f"📊 {stats['n_simulations']:,} carteras simuladas")
        print(f"   Sharpe range: {stats['sharpe_range'][0]:.2f} a {stats['sharpe_range'][1]:.2f}\n")
        
        min_vol = response.optimal_portfolios['minimum_volatility']
        max_sharpe = response.optimal_portfolios['maximum_sharpe']
        
        print(f"🎯 MÍNIMA VOLATILIDAD:")
        print(f"   Volatilidad: {min_vol.volatility:.4f}")
        print(f"   Sharpe: {min_vol.sharpe_ratio:.3f}")
        for ticker, weight in min_vol.weights.items():
            print(f"   {ticker}: {weight*100:.1f}%")
        
        print(f"\n⭐ MÁXIMO SHARPE:")
        print(f"   Sharpe: {max_sharpe.sharpe_ratio:.3f}")
        print(f"   Volatilidad: {max_sharpe.volatility:.4f}")
        for ticker, weight in max_sharpe.weights.items():
            print(f"   {ticker}: {weight*100:.1f}%")
        
        if should_plot:
            print(f"\n📊 Generando gráfico...")
            self._plot_frontier(response)
        
        print(f"\n{'='*60}\n")
    
    def _plot_frontier(self, response):
        """Grafica frontera eficiente."""
        import matplotlib.pyplot as plt
        
        df = response.to_dataframe()
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            df['volatility'] * 100,
            df['return'] * 100,
            c=df['sharpe'],
            cmap='viridis',
            alpha=0.6,
            s=10
        )
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        min_vol = response.optimal_portfolios['minimum_volatility']
        max_sharpe = response.optimal_portfolios['maximum_sharpe']
        
        plt.scatter(min_vol.volatility * 100, min_vol.expected_return * 100,
                   color='red', s=200, marker='*', label='Min Volatilidad',
                   edgecolors='black', linewidths=2)
        
        plt.scatter(max_sharpe.volatility * 100, max_sharpe.expected_return * 100,
                   color='gold', s=200, marker='*', label='Max Sharpe',
                   edgecolors='black', linewidths=2)
        
        plt.xlabel('Volatilidad (%)')
        plt.ylabel('Retorno (%)')
        plt.title('Frontera Eficiente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def run(self):
        """Ejecuta CLI."""
        parser = argparse.ArgumentParser(description='Portfolio Analyzer')
        subparsers = parser.add_subparsers(dest='command')
        
        # analyze
        analyze_p = subparsers.add_parser('analyze')
        analyze_p.add_argument('ticker')
        analyze_p.add_argument('--start', default='2020-01-01')
        analyze_p.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
        analyze_p.add_argument('--risk-free-rate', type=float, default=0.02)
        
        # optimize
        optimize_p = subparsers.add_parser('optimize')
        optimize_p.add_argument('tickers')
        optimize_p.add_argument('--strategy', choices=['minimum_variance', 'target_volatility'], default='minimum_variance')
        optimize_p.add_argument('--target-volatility', type=float)
        optimize_p.add_argument('--start', default='2020-01-01')
        optimize_p.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
        optimize_p.add_argument('--risk-free-rate', type=float, default=0.02)
        
        # frontier
        frontier_p = subparsers.add_parser('frontier')
        frontier_p.add_argument('tickers')
        frontier_p.add_argument('--simulations', type=int, default=10000)
        frontier_p.add_argument('--start', default='2020-01-01')
        frontier_p.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
        frontier_p.add_argument('--risk-free-rate', type=float, default=0.02)
        frontier_p.add_argument('--plot', action='store_true')
        
        args = parser.parse_args()
        
        if args.command == 'analyze':
            self.analyze_command(args)
        elif args.command == 'optimize':
            self.optimize_command(args)
        elif args.command == 'frontier':
            self.frontier_command(args)
        else:
            parser.print_help()


def main():
    cli = PortfolioAnalyzerCLI()
    cli.run()


if __name__ == '__main__':
    main()