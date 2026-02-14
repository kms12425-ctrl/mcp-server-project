import yfinance as yf
from curl_cffi import requests
import pandas as pd


def test_yfinance_fetch():
    ticker_symbol = "AAPL"
    print(f"Starting test for ticker: {ticker_symbol}")

    # # Method 1: Standard yfinance usage (Reference: https://ranaroussi.github.io/yfinance/)
    # print("\n[Test 1] Standard yfinance fetch...")
    # try:
    #     t = yf.Ticker(ticker_symbol)
    #     # Using a short period for quick testing
    #     hist = t.history(period="5d")

    #     if not hist.empty:
    #         print("✅ Success! specific data fetched:")
    #         print(hist[['Close', 'Volume']])
    #     else:
    #         print("⚠️ Result is empty (might be rate limited without custom session).")
    # except Exception as e:
    #     print(f"❌ Failed: {e}")

    # Method 2: With curl_cffi session (User's specific fix)
    print("\n[Test 2] Fetch with curl_cffi session (impersonate='chrome')...")
    try:
        session = requests.Session(impersonate="chrome")
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        })

        t = yf.Ticker(ticker_symbol, session=session)
        hist = t.history(period="5d")

        if not hist.empty:
            print("✅ Success! specific data fetched with session:")
            print(hist[['Close', 'Volume']])
        else:
            print("⚠️ Result is empty even with session.")
    except Exception as e:
        print(f"❌ Failed with session: {e}")


if __name__ == "__main__":
    test_yfinance_fetch()
