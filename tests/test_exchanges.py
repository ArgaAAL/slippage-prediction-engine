"""
Exchange Connection Test
========================

Simple test script to verify exchange connections and data fetching
before running the full dataset generation.
"""

import ccxt
import time
from typing import Dict, List

def test_exchange_connection(exchange_name: str) -> bool:
    """Test connection to a single exchange"""
    try:
        print(f"🔍 Testing {exchange_name}...")
        
        # Map exchange names
        exchange_mapping = {
            'coinbasepro': 'coinbase',
            'coinbase': 'coinbase',
        }
        
        ccxt_name = exchange_mapping.get(exchange_name, exchange_name)
        exchange_class = getattr(ccxt, ccxt_name)
        
        # Initialize exchange
        exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'spot'}
        })
        
        # Load markets
        exchange.load_markets()
        print(f"  ✅ Markets loaded: {len(exchange.markets)} pairs")
        
        # Test symbols
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/USD']
        available_symbols = []
        
        for symbol in test_symbols:
            if symbol in exchange.markets:
                available_symbols.append(symbol)
        
        print(f"  ✅ Available test symbols: {available_symbols}")
        
        if not available_symbols:
            print(f"  ⚠️ No test symbols available")
            return False
        
        # Test order book fetch
        test_symbol = available_symbols[0]
        order_book = exchange.fetch_order_book(test_symbol, limit=5)
        
        if order_book['bids'] and order_book['asks']:
            bid_price = order_book['bids'][0][0]
            ask_price = order_book['asks'][0][0]
            print(f"  ✅ Order book OK: {test_symbol} bid={bid_price}, ask={ask_price}")
        else:
            print(f"  ❌ Empty order book for {test_symbol}")
            return False
        
        # Test trades fetch
        try:
            trades = exchange.fetch_trades(test_symbol, limit=5)
            print(f"  ✅ Trades OK: {len(trades)} recent trades")
        except Exception as e:
            print(f"  ⚠️ Trades fetch failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def main():
    """Test all exchanges"""
    print("🧪 Testing Exchange Connections")
    print("=" * 40)
    
    # Test exchanges
    exchanges_to_test = ['binance', 'kraken', 'coinbase', 'okx']
    
    results = {}
    for exchange_name in exchanges_to_test:
        results[exchange_name] = test_exchange_connection(exchange_name)
        time.sleep(1)  # Rate limit protection
        print()
    
    # Summary
    print("📊 Summary:")
    working_exchanges = [name for name, status in results.items() if status]
    failed_exchanges = [name for name, status in results.items() if not status]
    
    print(f"✅ Working exchanges ({len(working_exchanges)}): {working_exchanges}")
    if failed_exchanges:
        print(f"❌ Failed exchanges ({len(failed_exchanges)}): {failed_exchanges}")
    
    if len(working_exchanges) >= 2:
        print("\n🎉 Sufficient exchanges available for dataset generation!")
    else:
        print("\n⚠️ Need at least 2 working exchanges for good dataset diversity")
    
    return working_exchanges

if __name__ == "__main__":
    working = main()
    print(f"\nNext step: Run dataset generation with {len(working)} working exchanges")