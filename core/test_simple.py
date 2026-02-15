import pandas as pd
from core.dataloader import StockDataLoader


def test_local_fetch():
    ticker_symbol = "AAPL"
    print(f"Starting Local CSV test for ticker: {ticker_symbol}")

    try:
        loader = StockDataLoader(ticker_symbol)
        df = loader.fetch(use_cache=False)

        if not df is None and not df.empty:
            print("✅ Success! Data fetched from CSV:")
            print(df[['Open', 'Close', 'Volume']].tail())
        else:
            print("⚠️ Result is empty.")
            
    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    test_local_fetch()
