"""
AI-Powered Trade Execution Cost Predictor - Enhanced Dataset Generator
====================================================================

This script generates a comprehensive dataset for training an AI model to predict
cryptocurrency trade execution costs with realistic slippage simulation.

The dataset simulates trades against real market data with proper market impact modeling,
partial fill penalties, exchange-specific factors, and volatility adjustments.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class MarketDataCollector:
    """Collects and processes market data from multiple exchanges"""
    
    def __init__(self, exchanges: List[str] = None):
        if exchanges is None:
            exchanges = ['binance', 'kraken', 'coinbase', 'okx']
        
        self.exchanges = {}
        self.initialize_exchanges(exchanges)
        
    def initialize_exchanges(self, exchange_names: List[str]):
        """Initialize exchange connections"""
        for name in exchange_names:
            try:
                exchange_mapping = {
                    'coinbasepro': 'coinbase',
                    'coinbase': 'coinbase',
                    'binance': 'binance',
                    'kraken': 'kraken',
                    'okx': 'okx'
                }
                
                ccxt_name = exchange_mapping.get(name, name)
                exchange_class = getattr(ccxt, ccxt_name)
                
                exchange = exchange_class({
                    'apiKey': '',
                    'secret': '',
                    'timeout': 30000,
                    'enableRateLimit': True,
                    'sandbox': False,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
                
                exchange.load_markets()
                self.exchanges[name] = exchange
                print(f"✅ Initialized {name}")
                
            except Exception as e:
                print(f"❌ Failed to initialize {name}: {e}")
                continue
    
    def get_order_book(self, exchange_name: str, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Fetch order book data"""
        try:
            exchange = self.exchanges[exchange_name]
            
            if symbol not in exchange.markets:
                return None  # Silently return None for unavailable symbols
            
            order_book = exchange.fetch_order_book(symbol, limit)
            order_book['exchange'] = exchange_name
            order_book['symbol'] = symbol
            order_book['timestamp'] = exchange.milliseconds()
            return order_book
        except Exception as e:
            return None  # Silently handle errors
    
    def get_recent_trades(self, exchange_name: str, symbol: str, limit: int = 100) -> Optional[List]:
        """Fetch recent trades"""
        try:
            exchange = self.exchanges[exchange_name]
            
            if symbol not in exchange.markets:
                return None
                
            trades = exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            print(f"Error fetching trades from {exchange_name} for {symbol}: {e}")
            return None

class FeatureEngineer:
    """Generates features from raw market data"""
    
    @staticmethod
    def calculate_spread_metrics(order_book: Dict) -> Dict:
        """Calculate spread-related metrics"""
        if not order_book['bids'] or not order_book['asks']:
            return {}
        
        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        
        spread = best_ask - best_bid
        spread_percentage = (spread / mid_price) * 100 if mid_price > 0 else 0
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': spread,
            'spread_percentage': spread_percentage
        }
    
    @staticmethod
    def calculate_market_depth(order_book: Dict, levels: List[int] = [1, 5, 10]) -> Dict:
        """Calculate market depth at different levels"""
        metrics = {}
        
        for level in levels:
            bid_depth = sum([bid[1] * bid[0] for bid in order_book['bids'][:level]])
            ask_depth = sum([ask[1] * ask[0] for ask in order_book['asks'][:level]])
            
            metrics[f'bid_depth_level_{level}'] = bid_depth
            metrics[f'ask_depth_level_{level}'] = ask_depth
            metrics[f'total_depth_level_{level}'] = bid_depth + ask_depth
        
        return metrics
    
    @staticmethod
    def calculate_depth_by_percentage(order_book: Dict, percentages: List[float] = [0.5, 1.0, 2.0]) -> Dict:
        """Calculate market depth within percentage ranges of mid-price"""
        if not order_book['bids'] or not order_book['asks']:
            return {}
        
        mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
        metrics = {}
        
        for pct in percentages:
            upper_bound = mid_price * (1 + pct/100)
            lower_bound = mid_price * (1 - pct/100)
            
            bid_depth = sum([bid[1] * bid[0] for bid in order_book['bids'] if bid[0] >= lower_bound])
            ask_depth = sum([ask[1] * ask[0] for ask in order_book['asks'] if ask[0] <= upper_bound])
            
            metrics[f'bid_depth_pct_{pct}'] = bid_depth
            metrics[f'ask_depth_pct_{pct}'] = ask_depth
            metrics[f'total_depth_pct_{pct}'] = bid_depth + ask_depth
        
        return metrics
    
    @staticmethod
    def calculate_order_book_imbalance(order_book: Dict, levels: int = 10) -> float:
        """Calculate order book imbalance"""
        if not order_book['bids'] or not order_book['asks']:
            return 0.5
        
        total_bids = sum([bid[1] for bid in order_book['bids'][:levels]])
        total_asks = sum([ask[1] for ask in order_book['asks'][:levels]])
        total_volume = total_bids + total_asks
        
        if total_volume == 0:
            return 0.5
        
        return total_bids / total_volume
    
    @staticmethod
    def calculate_price_impact_slope(order_book: Dict, side: str = 'both') -> Dict:
        """Calculate price impact slope (liquidity slope)"""
        metrics = {}
        
        if side in ['buy', 'both'] and len(order_book['asks']) >= 5:
            ask_prices = [ask[0] for ask in order_book['asks'][:5]]
            ask_slope = (ask_prices[-1] - ask_prices[0]) / ask_prices[0] * 100
            metrics['ask_price_slope'] = ask_slope
        
        if side in ['sell', 'both'] and len(order_book['bids']) >= 5:
            bid_prices = [bid[0] for bid in order_book['bids'][:5]]
            bid_slope = (bid_prices[0] - bid_prices[-1]) / bid_prices[0] * 100
            metrics['bid_price_slope'] = bid_slope
        
        return metrics
    
    @staticmethod
    def calculate_trade_volatility(trades: List[Dict], window_minutes: int = 1) -> Dict:
        """Calculate recent trade volatility and volume"""
        if not trades:
            return {'trade_volatility_1m': 0, 'trade_volume_1m': 0, 'trade_count_1m': 0}
        
        current_time = time.time() * 1000
        cutoff_time = current_time - (window_minutes * 60 * 1000)
        
        recent_trades = [t for t in trades if t['timestamp'] >= cutoff_time]
        
        if len(recent_trades) < 2:
            return {'trade_volatility_1m': 0, 'trade_volume_1m': 0, 'trade_count_1m': 0}
        
        prices = [t['price'] for t in recent_trades]
        volumes = [t['amount'] for t in recent_trades]
        
        volatility = np.std(prices) if len(prices) > 1 else 0
        total_volume = sum(volumes)
        
        return {
            f'trade_volatility_{window_minutes}m': volatility,
            f'trade_volume_{window_minutes}m': total_volume,
            f'trade_count_{window_minutes}m': len(recent_trades)
        }

class RealisticSlippageSimulator:
    """Enhanced slippage simulator with realistic market impact modeling"""
    
    def __init__(self):
        # Exchange-specific factors
        self.exchange_factors = {
            'binance': 0.9,     # Was 0.85, slightly higher slippage
            'coinbase': 1.1,    # Was 0.95, higher slippage
            'kraken': 1.3,      # Was 1.05, much higher slippage  
            'okx': 1.0,         # Was 0.90, neutral
            'default': 1.2      # Higher default
        }
        
        # Minimum slippage thresholds based on order size (USD)
        self.minimum_slippage = {
            (0, 1000): 0.002,         # 0.2% minimum for small orders
            (1000, 5000): 0.003,      # 0.3% minimum  
            (5000, 25000): 0.005,     # 0.5% minimum
            (25000, 100000): 0.008,   # 0.8% minimum
            (100000, float('inf')): 0.015  # 1.5% minimum for large orders
        }
    
    def get_minimum_slippage(self, order_size_usd: float) -> float:
        """Get minimum slippage based on order size"""
        for (min_size, max_size), min_slip in self.minimum_slippage.items():
            if min_size <= order_size_usd < max_size:
                return min_slip
        return 0.005  # Default high minimum
    
    def calculate_market_impact(self, cumulative_volume: float, total_depth: float, 
                           level_index: int, order_size_usd: float) -> float:
        # Base impact increases exponentially with depth - INCREASED VALUES
        depth_impact = 1 + (level_index ** 1.5) * 0.002  # Was 0.0002, now 0.002
        
        # Volume impact - INCREASED VALUES
        volume_ratio = cumulative_volume / max(total_depth, 1)
        volume_impact = 1 + (volume_ratio ** 2) * 0.05  # Was 0.01, now 0.05
        
        # Size-based impact - INCREASED VALUES  
        size_impact = 1 + (order_size_usd / 100000) * 0.01  # Was 0.001, now 0.01
        
        # Combined multiplicative impact
        total_impact = depth_impact * volume_impact * size_impact
        
        return min(total_impact, 1.5)
    
    def apply_within_level_impact(self, level_price: float, level_amount: float, 
                                 fill_amount: float, is_buy: bool) -> float:
        """
        Apply micro-impact within a single order book level for large fills
        """
        if fill_amount <= 0 or level_amount <= 0:
            return level_price
        
        # Calculate what percentage of this level we're consuming
        consumption_ratio = min(fill_amount / level_amount, 1.0)
        
        if consumption_ratio < 0.1:  # Less than 10% of level
            return level_price
        
        # Apply progressive price impact within the level
        # The more we consume, the worse the effective price becomes
        impact_factor = 1 + (consumption_ratio ** 2) * 0.001
        
        if is_buy:
            return level_price * impact_factor  # Price increases for buys
        else:
            return level_price / impact_factor  # Price decreases for sells
    
    def calculate_volatility_multiplier(self, volatility_metrics: Dict, 
                                  spread_percentage: float) -> float:
        base_multiplier = 1.2  # Start higher, was 1.0
        
        # Recent volatility impact - INCREASED
        recent_volatility = volatility_metrics.get('trade_volatility_1m', 0)
        if recent_volatility > 0:
            vol_multiplier = 1 + min(recent_volatility * 0.05, 1.0)  # Was 0.01, now 0.05
            base_multiplier *= vol_multiplier
        
        # Spread-based volatility - INCREASED
        if spread_percentage > 0.05:  # Lower threshold, was 0.1
            spread_multiplier = 1 + (spread_percentage - 0.05) * 0.5  # Was 0.1, now 0.5
            base_multiplier *= min(spread_multiplier, 3.0)  # Higher cap
        
        # Trade activity impact - INCREASED
        trade_count = volatility_metrics.get('trade_count_1m', 0)
        if trade_count < 10:  # Higher threshold, was 5
            activity_multiplier = 1 + (10 - trade_count) * 0.05  # Was 0.02, now 0.05
            base_multiplier *= min(activity_multiplier, 2.0)  # Higher cap
        
        return min(base_multiplier, 5.0)
    
    def simulate_market_order(self, order_book: Dict, order_size_crypto: float, 
                            trade_side: int, quote_price: float, exchange_name: str,
                            volatility_metrics: Dict, spread_percentage: float) -> Dict:
        """
        Enhanced market order simulation with realistic impact modeling
        
        Args:
            order_book: Order book data
            order_size_crypto: Amount of crypto to trade
            trade_side: 0 for sell, 1 for buy
            quote_price: Expected price (best bid/ask)
            exchange_name: Exchange name for exchange-specific adjustments
            volatility_metrics: Recent volatility data
            spread_percentage: Current spread percentage
        """
        remaining_amount = order_size_crypto
        total_cost = 0
        total_amount_filled = 0
        cumulative_volume_usd = 0
        
        # Choose appropriate side of order book
        book_side = order_book['asks'] if trade_side == 1 else order_book['bids']
        is_buy = trade_side == 1
        
        if not book_side:
            return {
                'average_price': quote_price,
                'total_cost': order_size_crypto * quote_price,
                'amount_filled': 0,
                'slippage_percentage': 1.0
            }
        
        # Calculate order size in USD
        order_size_usd = order_size_crypto * quote_price
        
        # Get total depth for impact calculations
        total_depth = sum([level[1] * level[0] for level in book_side[:20]])
        
        # Exchange-specific adjustment
        exchange_factor = self.exchange_factors.get(exchange_name, 1.0)
        
        # Walk through order book levels with realistic impact
        for level_index, level in enumerate(book_side):
            if remaining_amount <= 1e-8:
                break
            
            level_price = level[0]
            level_amount = level[1]
            
            # Amount we can fill at this level
            fill_amount = min(remaining_amount, level_amount)
            
            # Apply within-level micro-impact for large fills
            effective_price = self.apply_within_level_impact(
                level_price, level_amount, fill_amount, is_buy
            )
            
            # Apply progressive market impact
            market_impact = self.calculate_market_impact(
                cumulative_volume_usd, total_depth, level_index, order_size_usd
            )
            
            if is_buy:
                effective_price *= market_impact
            else:
                effective_price /= market_impact
            
            # Calculate cost for this fill
            level_cost = fill_amount * effective_price
            
            total_cost += level_cost
            total_amount_filled += fill_amount
            remaining_amount -= fill_amount
            cumulative_volume_usd += level_cost
            
            # Add escalating impact for very deep orders
            if level_index >= 10:
                deep_impact = 1 + (level_index - 9) * 0.002
                if is_buy:
                    total_cost *= deep_impact
                else:
                    total_cost /= deep_impact
        
        # Handle partial fills with penalties
        fill_percentage = total_amount_filled / order_size_crypto if order_size_crypto > 0 else 0
        
        if remaining_amount > 1e-8:  # Partial fill
            if fill_percentage < 0.5:  # Less than 50% filled - severe penalty
                return {
                    'average_price': quote_price * (2.0 if is_buy else 0.5),
                    'total_cost': total_cost,
                    'amount_filled': total_amount_filled,
                    'slippage_percentage': 0.3  # 30% slippage penalty
                }
            else:
                # Moderate partial fill penalty
                partial_fill_penalty = (1 - fill_percentage) * 0.05
        else:
            partial_fill_penalty = 0
        
        # Calculate base slippage
        if total_amount_filled > 1e-8:
            average_price = total_cost / total_amount_filled
            base_slippage = abs(quote_price - average_price) / quote_price
        else:
            average_price = quote_price
            base_slippage = 1.0
        
        # Apply volatility multiplier
        volatility_multiplier = self.calculate_volatility_multiplier(
            volatility_metrics, spread_percentage
        )
        
        # Apply all adjustments
        final_slippage = base_slippage * volatility_multiplier * exchange_factor
        final_slippage += partial_fill_penalty
        
        # Apply minimum slippage threshold
        minimum_slippage = self.get_minimum_slippage(order_size_usd)
        final_slippage = max(final_slippage, minimum_slippage)
        
        # Add randomness to simulate real-world variability (±20%)
        randomness_factor = np.random.uniform(0.8, 1.5)
        final_slippage *= randomness_factor
        
        # Cap maximum slippage at reasonable levels
        max_slippage = min(0.05, order_size_usd / 10000 * 0.01)
        final_slippage = min(final_slippage, max_slippage)
        
        return {
            'average_price': average_price,
            'total_cost': total_cost,
            'amount_filled': total_amount_filled,
            'slippage_percentage': final_slippage,
            'fill_percentage': fill_percentage,
            'volatility_multiplier': volatility_multiplier,
            'exchange_factor': exchange_factor
        }

class EnhancedDatasetGenerator:
    """Enhanced dataset generator with realistic slippage simulation"""
    
    def __init__(self):
        self.collector = MarketDataCollector()
        self.engineer = FeatureEngineer()
        self.simulator = RealisticSlippageSimulator()
        
        # Common trading pairs
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT'
        ]
        
        # Symbol alternatives for specific exchanges
        self.symbol_alternatives = {
            'kraken': {
                'BTC/USDT': 'BTC/USD',
                'ETH/USDT': 'ETH/USD',
                'ADA/USDT': 'ADA/USD',
                'SOL/USDT': 'SOL/USD',
                'XRP/USDT': 'XRP/USD',
                'DOT/USDT': 'DOT/USD',
                'AVAX/USDT': 'AVAX/USD',
                'MATIC/USDT': 'MATIC/USD',
                'LINK/USDT': 'LINK/USD'
            }
        }
        
        # More realistic order size distribution
        self.order_sizes_usd = [
            100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000,
            25000, 50000, 75000, 100000, 250000, 500000, 1000000
        ]
        
        # Weights favoring smaller, more common order sizes (must sum to 1.0)
        self.size_weights = [
            0.25, 0.15, 0.15, 0.10, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02,
            0.015, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005
        ]
        
        # Ensure weights sum to exactly 1.0
        self.size_weights = np.array(self.size_weights)
        self.size_weights = self.size_weights / self.size_weights.sum()
        
        # Enhanced symbol alternatives for more exchanges
        self.symbol_alternatives = {
            'kraken': {
                'BTC/USDT': 'BTC/USD',
                'ETH/USDT': 'ETH/USD',
                'BNB/USDT': None,  # Not available on Kraken
                'ADA/USDT': 'ADA/USD',
                'SOL/USDT': 'SOL/USD',
                'XRP/USDT': 'XRP/USD',
                'DOT/USDT': 'DOT/USD',
                'AVAX/USDT': 'AVAX/USD',
                'MATIC/USDT': 'MATIC/USD',
                'LINK/USDT': 'LINK/USD'
            },
            'okx': {
                'BTC/USDT': 'BTC/USDT',
                'ETH/USDT': 'ETH/USDT',
                'BNB/USDT': None,  # Not available on OKX
                'ADA/USDT': 'ADA/USDT',
                'SOL/USDT': 'SOL/USDT',
                'XRP/USDT': 'XRP/USDT',
                'DOT/USDT': 'DOT/USDT',
                'AVAX/USDT': 'AVAX/USDT',
                'MATIC/USDT': None,  # Not available on OKX
                'LINK/USDT': 'LINK/USDT'
            },
            'coinbase': {
                'BTC/USDT': 'BTC/USD',
                'ETH/USDT': 'ETH/USD',
                'BNB/USDT': None,  # Not available
                'ADA/USDT': 'ADA/USD',
                'SOL/USDT': 'SOL/USD',
                'XRP/USDT': 'XRP/USD',
                'DOT/USDT': 'DOT/USD',
                'AVAX/USDT': 'AVAX/USD',
                'MATIC/USDT': 'MATIC/USD',
                'LINK/USDT': 'LINK/USD'
            }
        }
    
    def generate_sample(self, exchange_name: str, symbol: str) -> Optional[Dict]:
        """Generate a single training sample with enhanced realism"""
        try:
            # Handle symbol alternatives
            actual_symbol = symbol
            if exchange_name in self.symbol_alternatives:
                alternative = self.symbol_alternatives[exchange_name].get(symbol, symbol)
                if alternative is None:
                    # Symbol not available on this exchange, skip
                    return None
                actual_symbol = alternative
            
            # Fetch market data
            order_book = self.collector.get_order_book(exchange_name, actual_symbol)
            trades = self.collector.get_recent_trades(exchange_name, actual_symbol)
            
            if not order_book or not order_book['bids'] or not order_book['asks']:
                return None
            
            # Calculate features
            spread_metrics = self.engineer.calculate_spread_metrics(order_book)
            depth_metrics = self.engineer.calculate_market_depth(order_book)
            depth_pct_metrics = self.engineer.calculate_depth_by_percentage(order_book)
            order_book_imbalance = self.engineer.calculate_order_book_imbalance(order_book)
            price_slopes = self.engineer.calculate_price_impact_slope(order_book)
            
            volatility_metrics = {'trade_volatility_1m': 0, 'trade_volume_1m': 0, 'trade_count_1m': 0}
            if trades:
                volatility_metrics = self.engineer.calculate_trade_volatility(trades)
            
            mid_price = spread_metrics.get('mid_price', 0)
            if mid_price <= 0:
                return None
            
            # Generate realistic trade parameters
            trade_side = np.random.randint(0, 2)
            
            # Select order size with fixed weights
            try:
                order_size_usd = np.random.choice(self.order_sizes_usd, p=self.size_weights)
            except Exception as e:
                # Fallback to simple random selection if weights still have issues
                order_size_usd = np.random.choice([100, 500, 1000, 5000, 10000, 25000])
                
            order_size_crypto = order_size_usd / mid_price
            
            # Validate against market depth
            total_depth = depth_metrics.get('total_depth_level_10', 0)
            if total_depth <= 0 or order_size_usd > total_depth * 0.3:  # Order too large, skip
                return None
            
            # Get quote price
            quote_price = spread_metrics['best_ask'] if trade_side == 1 else spread_metrics['best_bid']
            
            # Enhanced simulation
            execution_result = self.simulator.simulate_market_order(
                order_book, order_size_crypto, trade_side, quote_price, 
                exchange_name, volatility_metrics, spread_metrics.get('spread_percentage', 0)
            )

            # 🔧 DEBUG PRINT - MOVE TO HERE (after execution_result is available)
            print(f"Order: ${order_size_usd:.0f}, Exchange: {exchange_name}, Slippage: {execution_result['slippage_percentage']:.4f} ({execution_result['slippage_percentage']*100:.2f}%)")
            
            # Quality check - ensure realistic slippage range
            slippage = execution_result['slippage_percentage']
            if slippage > 0.2 or slippage < 0:  # Skip unrealistic values
                return None
            
            # Additional derived features
            depth_utilization = order_size_usd / max(total_depth, 1)
            size_vs_spread = order_size_usd / max(spread_metrics.get('spread', 1), 0.01)
            
            # Compile comprehensive feature set
            sample = {
                # Identifiers
                'exchange': exchange_name,
                'symbol': actual_symbol,
                'original_symbol': symbol,
                'timestamp': int(time.time() * 1000),
                
                # Order features
                'order_size_fiat': order_size_usd,
                'order_size_crypto': order_size_crypto,
                'trade_side': trade_side,
                
                # Market microstructure
                'spread_percentage': spread_metrics.get('spread_percentage', 0),
                'market_depth_level_1': depth_metrics.get('total_depth_level_1', 0),
                'market_depth_level_5': depth_metrics.get('total_depth_level_5', 0),
                'market_depth_level_10': depth_metrics.get('total_depth_level_10', 0),
                'market_depth_percent_0.5': depth_pct_metrics.get('total_depth_pct_0.5', 0),
                'market_depth_percent_1.0': depth_pct_metrics.get('total_depth_pct_1.0', 0),
                'market_depth_percent_2.0': depth_pct_metrics.get('total_depth_pct_2.0', 0),
                'order_book_imbalance': order_book_imbalance,
                
                # Price impact features
                'ask_price_slope': price_slopes.get('ask_price_slope', 0),
                'bid_price_slope': price_slopes.get('bid_price_slope', 0),
                
                # Volatility features
                'trade_volatility_1m': volatility_metrics.get('trade_volatility_1m', 0),
                'trade_volume_1m': volatility_metrics.get('trade_volume_1m', 0),
                'trade_count_1m': volatility_metrics.get('trade_count_1m', 0),
                
                # Enhanced features
                'mid_price': mid_price,
                'depth_utilization': depth_utilization,
                'size_vs_spread_ratio': size_vs_spread,
                'bid_ask_level_ratio': len(order_book['bids']) / max(len(order_book['asks']), 1),
                'order_book_depth_ratio': depth_metrics.get('bid_depth_level_5', 0) / max(depth_metrics.get('ask_depth_level_5', 1), 1),
                
                # Execution metrics
                'fill_percentage': execution_result.get('fill_percentage', 1.0),
                'volatility_multiplier': execution_result.get('volatility_multiplier', 1.0),
                'exchange_factor': execution_result.get('exchange_factor', 1.0),
                
                # Target variable
                'slippage_percentage': execution_result['slippage_percentage']
            }
            
            return sample
            
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None
    
    def generate_dataset(self, num_samples: int = 10000, output_file: str = 'enhanced_trade_cost_dataset.csv') -> pd.DataFrame:
        """Generate complete dataset with enhanced realism"""
        print(f"🚀 Starting enhanced dataset generation - Target: {num_samples} samples")
        print(f"📊 Exchanges: {list(self.collector.exchanges.keys())}")
        print(f"💱 Symbols: {self.symbols}")
        print(f"🎯 Enhanced features: Market impact, volatility multipliers, exchange factors")
        
        samples = []
        successful_samples = 0
        failed_attempts = 0
        
        while successful_samples < num_samples:
            exchange_name = np.random.choice(list(self.collector.exchanges.keys()))
            symbol = np.random.choice(self.symbols)
            
            try:
                sample = self.generate_sample(exchange_name, symbol)
                
                if sample is not None:
                    samples.append(sample)
                    successful_samples += 1
                    
                    if successful_samples % 100 == 0:
                        print(f"✅ Generated {successful_samples}/{num_samples} samples")
                        
                        # Show current statistics
                        temp_df = pd.DataFrame(samples[-100:])
                        avg_slippage = temp_df['slippage_percentage'].mean()
                        print(f"   📈 Recent batch avg slippage: {avg_slippage:.4f}")
                        
                else:
                    failed_attempts += 1
                    
            except Exception as e:
                failed_attempts += 1
                if failed_attempts % 100 == 0:
                    print(f"❌ Failed attempts: {failed_attempts}")
            
            time.sleep(0.15)  # Rate limiting
            
            # Safety break with more lenient threshold
            if failed_attempts > num_samples * 1.5:
                print(f"⚠️ Too many failed attempts ({failed_attempts}), stopping early")
                break
        
        # Create and analyze dataset
        df = pd.DataFrame(samples)
        
        if not df.empty:
            # Save dataset
            df.to_csv(output_file, index=False)
            print(f"💾 Dataset saved to {output_file}")
            print(f"📈 Dataset shape: {df.shape}")
            
            # Comprehensive analysis
            print(f"\n📊 Dataset Analysis:")
            print(f"- Total samples: {len(df)}")
            print(f"- Exchanges: {df['exchange'].value_counts().to_dict()}")
            print(f"- Order sizes (USD): ${df['order_size_fiat'].min():.0f} - ${df['order_size_fiat'].max():.0f}")
            
            print(f"\n🎯 Slippage Statistics:")
            print(f"- Mean: {df['slippage_percentage'].mean():.4f}")
            print(f"- Median: {df['slippage_percentage'].median():.4f}")
            print(f"- Std: {df['slippage_percentage'].std():.4f}")
            print(f"- Min: {df['slippage_percentage'].min():.4f}")
            print(f"- Max: {df['slippage_percentage'].max():.4f}")
            
            # Slippage distribution by order size
            print(f"\n💰 Slippage by Order Size:")
            size_bins = [0, 1000, 5000, 25000, 100000, float('inf')]
            size_labels = ['<$1K', '$1K-5K', '$5K-25K', '$25K-100K', '>$100K']
            df['size_category'] = pd.cut(df['order_size_fiat'], bins=size_bins, labels=size_labels)
            size_analysis = df.groupby('size_category')['slippage_percentage'].agg(['mean', 'count'])
            print(size_analysis)
            
            # Exchange comparison
            print(f"\n🏢 Slippage by Exchange:")
            exchange_analysis = df.groupby('exchange')['slippage_percentage'].agg(['mean', 'count'])
            print(exchange_analysis)
            
            # Quality validation
            print(f"\n🔍 Quality Checks:")
            print(f"- Samples with slippage = 0: {(df['slippage_percentage'] == 0).sum()}")
            print(f"- Samples with slippage > 1%: {(df['slippage_percentage'] > 0.01).sum()}")
            print(f"- Samples with slippage > 5%: {(df['slippage_percentage'] > 0.05).sum()}")
            print(f"- Average fill percentage: {df['fill_percentage'].mean():.3f}")
            print(f"- Average volatility multiplier: {df['volatility_multiplier'].mean():.3f}")
            
            # Feature correlation with slippage
            print(f"\n🔗 Top Features Correlated with Slippage:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlations = df[numeric_cols].corr()['slippage_percentage'].abs().sort_values(ascending=False)
            print(correlations.head(10))
            
        return df


# Usage example and validation
if __name__ == "__main__":
    print("🔧 Initializing Enhanced Dataset Generator...")
    print("=" * 60)
    
    # Initialize generator
    generator = EnhancedDatasetGenerator()
    
    if not generator.collector.exchanges:
        print("❌ No exchanges initialized. Please check your connection.")
        exit(1)
    
    print(f"✅ Initialized {len(generator.collector.exchanges)} exchanges")
    print("🚀 Starting dataset generation with realistic slippage modeling...")
    print("\n🎯 Key Improvements:")
    print("- Progressive market impact modeling")
    print("- Exchange-specific adjustment factors") 
    print("- Volatility-based slippage multipliers")
    print("- Minimum slippage thresholds by order size")
    print("- Partial fill penalties")
    print("- Within-level micro-impact for large orders")
    print("- Realistic order size distributions")
    print("\n" + "=" * 60)
    
    # Generate enhanced dataset
    dataset = generator.generate_dataset(
        num_samples=5000,  # Start with 5K for testing
        output_file='enhanced_trade_cost_dataset.csv'
    )
    
    if not dataset.empty:
        print("\n" + "=" * 60)
        print("🎉 Enhanced Dataset Generation Complete!")
        print("=" * 60)
        
        # Additional validation and insights
        print(f"\n📋 Dataset Summary:")
        print(f"- File: enhanced_trade_cost_dataset.csv")
        print(f"- Samples: {len(dataset)}")
        print(f"- Features: {len(dataset.columns)}")
        print(f"- Size: {dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Realistic slippage validation
        realistic_samples = len(dataset[
            (dataset['slippage_percentage'] >= 0.0001) & 
            (dataset['slippage_percentage'] <= 0.1)
        ])
        print(f"- Realistic slippage range (0.01%-10%): {realistic_samples}/{len(dataset)} ({realistic_samples/len(dataset)*100:.1f}%)")
        
        print(f"\n📈 Next Steps:")
        print("1. ✅ Dataset generated with realistic slippage simulation")
        print("2. 🔍 Review data quality and distributions")
        print("3. 🧠 Train ML models (Random Forest, XGBoost, Neural Networks)")
        print("4. 📊 Validate model predictions against real trading data")
        print("5. 🚀 Scale up to 50K-100K samples for production model")
        print("6. 🔧 Fine-tune parameters based on model performance")
        
        # Save feature importance guide
        feature_guide = """
# Enhanced Trade Cost Dataset - Feature Guide

## Target Variable
- `slippage_percentage`: Predicted slippage as percentage of quote price

## Order Features
- `order_size_fiat`: Order size in USD
- `order_size_crypto`: Order size in cryptocurrency units  
- `trade_side`: 0=sell, 1=buy

## Market Microstructure
- `spread_percentage`: Bid-ask spread as % of mid-price
- `market_depth_level_*`: Total liquidity at different order book levels
- `market_depth_percent_*`: Liquidity within % range of mid-price
- `order_book_imbalance`: Ratio of bid vs ask volume

## Market Impact Indicators  
- `depth_utilization`: Order size relative to available depth
- `size_vs_spread_ratio`: Order size relative to current spread
- `ask_price_slope`: Price progression in ask levels
- `bid_price_slope`: Price progression in bid levels

## Volatility & Activity
- `trade_volatility_1m`: Recent price volatility
- `trade_volume_1m`: Recent trading volume
- `trade_count_1m`: Recent trade frequency

## Execution Factors
- `fill_percentage`: Expected order fill rate
- `volatility_multiplier`: Slippage adjustment for market conditions
- `exchange_factor`: Exchange-specific slippage adjustment

## Key Model Features (High Correlation with Slippage)
1. Order size relative to market depth
2. Current spread percentage
3. Recent market volatility
4. Exchange liquidity characteristics
5. Order book imbalance
"""
        
        with open('feature_guide.md', 'w') as f:
            f.write(feature_guide)
        
        print("📚 Feature guide saved to: feature_guide.md")
        
    else:
        print("❌ Dataset generation failed - no samples created")
        print("🔧 Troubleshooting:")
        print("- Check internet connection")
        print("- Verify exchange API access") 
        print("- Try with fewer symbols or exchanges")
        print("- Increase failure tolerance threshold")