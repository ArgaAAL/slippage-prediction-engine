import requests
import json

API_BASE = "http://localhost:5000"

def test_health():
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_prediction():
    print("Testing prediction endpoint...")
    data = {
        "symbol": "BTC/USDT",
        "amount": 0.5,
        "side": "buy"
    }
    
    response = requests.post(
        f"{API_BASE}/predict", 
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Best venue: {result['best_venue']}")
        print("Quotes:")
        for quote in result['quotes']:
            print(f"  {quote['exchange']}: ${quote['total_cost']:.2f} (slippage: {quote['predicted_slippage_pct']:.4f}%)")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_compare():
    print("Testing compare endpoint...")
    data = {
        "symbol": "ETH/USDT",
        "amount": 10,
        "side": "sell",
        "exchanges": ["binance", "kraken"]
    }
    
    response = requests.post(
        f"{API_BASE}/compare", 
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        summary = result['comparison_summary']
        print(f"Best exchange: {summary['best_exchange']}")
        print(f"Potential savings: ${summary['potential_savings']:.2f}")
        print("Details:")
        for quote in result['detailed_quotes']:
            print(f"  {quote['exchange']}: ${quote['total_cost']:.2f}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

if __name__ == "__main__":
    test_health()
    test_prediction()
    test_compare()