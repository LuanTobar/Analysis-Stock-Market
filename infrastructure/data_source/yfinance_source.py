"""
YFinance Data Source - Descarga datos de Yahoo Finance.

CONCEPTO: Implementa IMarketDataSource.
Esta es la ÚNICA clase que sabe sobre yfinance.
"""

from typing import List, Dict
import pandas as pd
import yfinance as yf
import time
import logging

from domain import Ticker, DateRange
from domain.exceptions import DataFetchError
from use_cases.interfaces import IMarketDataSource


logger = logging.getLogger(__name__)


class YFinanceDataSource(IMarketDataSource):
    """Implementación de IMarketDataSource usando Yahoo Finance."""
    
    def __init__(self, interval: str = '1d'):
        self._interval = interval
    
    def fetch_historical_data(
        self, 
        ticker: Ticker, 
        date_range: DateRange
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de un ticker.
        
        MAPEO: Reemplaza tu yf.download() con manejo de errores.
        """
        logger.info(f"Fetching {ticker} from {date_range.start} to {date_range.end}")
        
        for attempt in range(3):
            try:
                data = yf.download(
                    str(ticker),
                    start=date_range.start,
                    end=date_range.end,
                    interval=self._interval,
                    progress=False,
                    auto_adjust=False
                )
                
                if data.empty:
                    raise DataFetchError(
                        message=f"No data for {ticker}",
                        details={"ticker": str(ticker)}
                    )
                
                cleaned = self._clean_data(data, ticker)
                logger.info(f"Fetched {len(cleaned)} rows for {ticker}")
                return cleaned
                
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Retry {attempt + 1} for {ticker}")
                    time.sleep(1)
                else:
                    raise DataFetchError(
                        message=f"Failed to fetch {ticker}",
                        details={"ticker": str(ticker), "error": str(e)}
                    )
    
    def fetch_multiple(
        self, 
        tickers: List[Ticker], 
        date_range: DateRange
    ) -> Dict[Ticker, pd.DataFrame]:
        """Descarga múltiples tickers en paralelo."""
        logger.info(f"Fetching {len(tickers)} tickers")
        
        ticker_strings = [str(t) for t in tickers]
        
        try:
            data = yf.download(
                ticker_strings,
                start=date_range.start,
                end=date_range.end,
                interval=self._interval,
                progress=False,
                group_by='ticker',
                auto_adjust=False
            )
            
            result = {}
            for ticker in tickers:
                try:
                    ticker_str = str(ticker)
                    ticker_data = data[ticker_str] if len(tickers) > 1 else data
                    cleaned = self._clean_data(ticker_data, ticker)
                    result[ticker] = cleaned
                    logger.info(f"Fetched {len(cleaned)} rows for {ticker}")
                except Exception as e:
                    logger.warning(f"Failed {ticker}: {e}")
                    continue
            
            if not result:
                raise DataFetchError(message="No tickers fetched")
            
            return result
            
        except Exception as e:
            raise DataFetchError(
                message="Failed to fetch multiple tickers",
                details={"error": str(e)}
            )
    
    def _clean_data(self, data: pd.DataFrame, ticker: Ticker) -> pd.DataFrame:
        """
        Limpia datos descargados.
        
        MAPEO: Tu código de limpieza de MultiIndex (líneas 60-78).
        """
        if data.empty:
            raise DataFetchError(f"Empty data for {ticker}")
        
        cleaned = data.copy()
        
        # Limpiar MultiIndex
        if isinstance(cleaned.columns, pd.MultiIndex):
            cleaned.columns = ['_'.join(col).strip() for col in cleaned.columns.values]
        
        # Normalizar nombres de columnas (remover espacios extra)
        cleaned.columns = cleaned.columns.str.strip()
        
        # Buscar columna Adj Close (case-insensitive, pero preferir Adj Close)
        adj_close_col = None
        
        # Primero intenta encontrar exactamente "Adj Close"
        if 'Adj Close' in cleaned.columns:
            adj_close_col = 'Adj Close'
        else:
            # Si no, busca variaciones
            for col in cleaned.columns:
                if 'Adj Close' in col or 'adj close' in col.lower():
                    adj_close_col = col
                    break
        
        # Si aún no encuentra, intenta con Close
        if adj_close_col is None:
            for col in cleaned.columns:
                if 'Close' in col:
                    adj_close_col = col
                    break
        
        if adj_close_col is None:
            raise DataFetchError(f"No Adj Close for {ticker}. Available columns: {list(cleaned.columns)}")
        
        # Renombrar a estándar si es necesario
        if adj_close_col != 'Adj Close':
            cleaned = cleaned.rename(columns={adj_close_col: 'Adj Close'})
        
        # Eliminar NaN
        cleaned = cleaned.dropna(subset=['Adj Close'])
        
        if cleaned.empty:
            raise DataFetchError(f"No valid data for {ticker}")
        
        # Asegurar DatetimeIndex
        if not isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = pd.to_datetime(cleaned.index)
        
        cleaned = cleaned.sort_index()
        
        return cleaned