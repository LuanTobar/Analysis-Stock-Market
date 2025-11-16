#!/usr/bin/env python3
"""Test which tickers are valid in yfinance"""

import yfinance as yf

tickers = ['IBB', 'MNR', 'MGPHF', 'SMH', 'LIT', 'EEM', 'AAPL', 'GOOGL', 'MSFT', 'TSLA']

print("Testing tickers:\n")
for t in tickers:
    try:
        data = yf.download(t, period="1d", progress=False)
        if len(data) > 0 and 'Adj Close' in data.columns:
            print(f"✅ {t:10} - Valid ({len(data)} rows)")
        else:
            print(f"❌ {t:10} - No Adj Close column")
    except Exception as e:
        print(f"❌ {t:10} - Error: {str(e)[:50]}")
