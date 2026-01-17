
import requests
import pandas as pd

BASE_URL = "https://www.oyakyatirim.com.tr/Equity/GetHistoricalEquityData?mode=2&code={}"

def get_oyak_data(ticker):
    url = BASE_URL.format(ticker)
    print(f"Fetching {url}...")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Data length: {len(data)}")
            if len(data) > 0:
                print("First 2 items:", data[:2])
                print("Last 2 items:", data[-2:])
        else:
            print("Response text:", response.text[:200])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_oyak_data("QNBFF")
    get_oyak_data("BRKT")
