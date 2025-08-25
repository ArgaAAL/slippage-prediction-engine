"""
Real-time Trade Cost Predictor - Deployment System (Fixed)
==================================================

This system provides real-time trade cost predictions using the trained AI model.
It fetches live market data, calculates features, and recommends the best execution venue.

Features:
- Real-time market data fetching
- Live slippage prediction
- Best execution venue recommendation
- REST API for integration
- Performance monitoring
"""

import numpy as np
import pandas as pd
import ccxt
import time
import json
import joblib
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from flask import Flask, request, jsonify
import logging

# For ONNX inference (faster deployment)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available - using sklearn models")

@dataclass
class TradeRequest:
    """Structure for trade cost prediction requests"""
    symbol: str
    amount: float  # Amount in crypto
    side: str  # 'buy' or 'sell'
    amount_usd: Optional[float] = None

@dataclass
class ExchangeQuote:
    """Structure for exchange quotes with predicted costs"""
    exchange: str
    symbol: str
    quote_price: float
    predicted_slippage: float
    total_cost: float  # Including all fees and slippage
    recommendation_score: float
    fees: Dict[str, float]

class RealTimePredictor:
    """Real-time trade cost prediction system"""
    
    def __init__(self, model_path: str = None, use_onnx: bool = False):
        self.exchanges = {}
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        
        # Initialize exchanges
        self.initialize_exchanges()
        
        # Load model
        self.load_model(model_path)
        
        # Exchange fees (you should update these with current rates)
        self.exchange_fees = {
            'binance': {'maker': 0.001, 'taker': 0.001},
            'kraken': {'maker': 0.0016, 'taker': 0.0026},
            'coinbase': {'maker': 0.005, 'taker': 0.005},
            'okx': {'maker': 0.0008, 'taker': 0.001}
        }
    
    def initialize_exchanges(self):
        """Initialize exchange connections"""
        exchange_configs = {
            'binance': ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}}),
            'kraken': ccxt.kraken({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        }
        
        for name, exchange in exchange_configs.items():
            try:
                exchange.load_markets()
                self.exchanges[name] = exchange
                print(f"✅ Connected to {name}")
            except Exception as e:
                print(f"❌ Failed to connect to {name}: {e}")
    
    def load_model(self, model_path: str = None):
        """Load the trained model and supporting files"""
        try:
            # Default to models directory if no path specified
            if model_path is None:
                model_dir = 'models'
            else:
                model_dir = model_path
            
            # Check if ONNX model exists and use_onnx is True
            onnx_path = os.path.join(model_dir, 'trade_cost_predictor.onnx') if os.path.isdir(model_dir) else f"{model_dir}.onnx"
            
            if self.use_onnx and os.path.exists(onnx_path):
                # Load ONNX model
                self.onnx_session = ort.InferenceSession(onnx_path)
                print("✅ Loaded ONNX model")
                
                # Load feature columns
                features_path = onnx_path.replace('.onnx', '_features.json')
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.feature_columns = json.load(f)
                else:
                    # Fallback to pickle file
                    features_pkl = os.path.join(model_dir, 'feature_columns.pkl')
                    self.feature_columns = joblib.load(features_pkl)
                    
            else:
                # Load sklearn models (default to lightgbm as it performed best)
                self.use_onnx = False
                
                # Try to load lightgbm model first (as it's typically the best performer)
                try:
                    model_file = os.path.join(model_dir, 'lightgbm_model.pkl')
                    scaler_file = os.path.join(model_dir, 'lightgbm_scaler.pkl')
                    features_file = os.path.join(model_dir, 'feature_columns.pkl')
                    
                    self.model = joblib.load(model_file)
                    self.scaler = joblib.load(scaler_file)
                    self.feature_columns = joblib.load(features_file)
                    print("✅ Loaded LightGBM sklearn model")
                    
                except Exception as lgb_error:
                    print(f"Failed to load LightGBM model: {lgb_error}")
                    
                    # Fallback to other models
                    for model_type in ['xgboost', 'random_forest']:
                        try:
                            model_file = os.path.join(model_dir, f'{model_type}_model.pkl')
                            scaler_file = os.path.join(model_dir, f'{model_type}_scaler.pkl')
                            
                            self.model = joblib.load(model_file)
                            self.scaler = joblib.load(scaler_file)
                            self.feature_columns = joblib.load(features_file)
                            print(f"✅ Loaded {model_type} sklearn model")
                            break
                            
                        except Exception as fallback_error:
                            print(f"Failed to load {model_type} model: {fallback_error}")
                            continue
                    
                    if self.model is None:
                        raise Exception("No valid model could be loaded")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def get_market_data(self, exchange_name: str, symbol: str) -> Optional[Dict]:
        """Fetch real-time market data"""
        try:
            exchange = self.exchanges[exchange_name]
            
            # Get order book
            order_book = exchange.fetch_order_book(symbol, limit=20)
            
            # Get recent trades
            trades = exchange.fetch_trades(symbol, limit=50)
            
            # Get ticker for additional info
            ticker = exchange.fetch_ticker(symbol)
            
            return {
                'order_book': order_book,
                'trades': trades,
                'ticker': ticker,
                'timestamp': exchange.milliseconds()
            }
            
        except Exception as e:
            print(f"Error fetching data from {exchange_name}: {e}")
            return None
    
    def calculate_features(self, market_data: Dict, trade_request: TradeRequest) -> Optional[Dict]:
        """Calculate features for the ML model"""
        try:
            order_book = market_data['order_book']
            trades = market_data['trades']
            
            if not order_book['bids'] or not order_book['asks']:
                return None
            
            # Basic price information
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
            
            # Convert trade side to numeric
            trade_side = 1 if trade_request.side.lower() == 'buy' else 0
            
            # Calculate order size in USD if not provided
            if trade_request.amount_usd is None:
                order_size_usd = trade_request.amount * mid_price
            else:
                order_size_usd = trade_request.amount_usd
            
            # Spread metrics
            spread = best_ask - best_bid
            spread_percentage = (spread / mid_price) * 100 if mid_price > 0 else 0
            
            # Market depth calculations
            market_depth_level_1 = (order_book['bids'][0][1] * best_bid + 
                                   order_book['asks'][0][1] * best_ask)
            
            # Level 5 depth
            bid_depth_5 = sum([bid[1] * bid[0] for bid in order_book['bids'][:5]])
            ask_depth_5 = sum([ask[1] * ask[0] for ask in order_book['asks'][:5]])
            market_depth_level_5 = bid_depth_5 + ask_depth_5
            
            # Level 10 depth
            bid_depth_10 = sum([bid[1] * bid[0] for bid in order_book['bids'][:10]])
            ask_depth_10 = sum([ask[1] * ask[0] for ask in order_book['asks'][:10]])
            market_depth_level_10 = bid_depth_10 + ask_depth_10
            
            # Percentage-based depth (0.5% from mid-price)
            upper_bound = mid_price * 1.005
            lower_bound = mid_price * 0.995
            
            bid_depth_pct = sum([bid[1] * bid[0] for bid in order_book['bids'] if bid[0] >= lower_bound])
            ask_depth_pct = sum([ask[1] * ask[0] for ask in order_book['asks'] if ask[0] <= upper_bound])
            market_depth_percent_05 = bid_depth_pct + ask_depth_pct
            
            # Order book imbalance
            total_bids = sum([bid[1] for bid in order_book['bids'][:10]])
            total_asks = sum([ask[1] for ask in order_book['asks'][:10]])
            total_volume = total_bids + total_asks
            order_book_imbalance = total_bids / total_volume if total_volume > 0 else 0.5
            
            # Price slopes
            ask_prices = [ask[0] for ask in order_book['asks'][:5]]
            bid_prices = [bid[0] for bid in order_book['bids'][:5]]
            
            ask_price_slope = (ask_prices[-1] - ask_prices[0]) / ask_prices[0] * 100 if len(ask_prices) >= 5 else 0
            bid_price_slope = (bid_prices[0] - bid_prices[-1]) / bid_prices[0] * 100 if len(bid_prices) >= 5 else 0
            
            # Trade volatility and volume (last 1 minute)
            current_time = time.time() * 1000
            recent_trades = [t for t in trades if current_time - t['timestamp'] <= 60000]
            
            if recent_trades:
                prices = [t['price'] for t in recent_trades]
                volumes = [t['amount'] for t in recent_trades]
                trade_volatility_1m = np.std(prices) if len(prices) > 1 else 0
                trade_volume_1m = sum(volumes)
            else:
                trade_volatility_1m = 0
                trade_volume_1m = 0
            
            # Calculate depth utilization (important feature from training)
            if market_depth_level_10 > 0:
                depth_utilization = order_size_usd / market_depth_level_10
            else:
                depth_utilization = 0
            
            # Additional features
            relative_order_size = order_size_usd / market_depth_level_10 if market_depth_level_10 > 0 else 0
            bid_ask_ratio = len(order_book['bids']) / max(len(order_book['asks']), 1)
            order_book_depth_ratio = bid_depth_5 / max(ask_depth_5, 1) if ask_depth_5 > 0 else 1
            
            # Base features dictionary
            features = {
                'order_size_fiat': order_size_usd,
                'order_size_crypto': trade_request.amount,
                'trade_side': trade_side,
                'spread_percentage': spread_percentage,
                'market_depth_level_1': market_depth_level_1,
                'market_depth_level_5': market_depth_level_5,
                'market_depth_level_10': market_depth_level_10,
                'market_depth_percent_0.5': market_depth_percent_05,
                'market_depth_percent_1.0': market_depth_percent_05 * 2,  # Approximation for 1%
                'order_book_imbalance': order_book_imbalance,
                'ask_price_slope': ask_price_slope,
                'bid_price_slope': bid_price_slope,
                'trade_volatility_1m': trade_volatility_1m,
                'trade_volume_1m': trade_volume_1m,
                'mid_price': mid_price,
                'relative_order_size': relative_order_size,
                'bid_ask_ratio': bid_ask_ratio,
                'order_book_depth_ratio': order_book_depth_ratio,
                'depth_utilization': depth_utilization  # Important feature from training
            }
            
            # Add engineered features (matching training pipeline)
            self._add_engineered_features(features)
            
            return features
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
    
    def _add_engineered_features(self, features: Dict):
        """Add engineered features that were created during training"""
        # Interaction features
        features['size_spread_interaction'] = features['order_size_fiat'] * features['spread_percentage']
        features['size_volatility_interaction'] = features['order_size_fiat'] * features['trade_volatility_1m']
        features['depth_utilization_squared'] = features['depth_utilization'] ** 2
        
        features['imbalance_spread'] = features['order_book_imbalance'] * features['spread_percentage']
        features['depth_spread_ratio'] = features['market_depth_level_5'] / (features['spread_percentage'] + 1e-8)
        features['volatility_spread'] = features['trade_volatility_1m'] * features['spread_percentage']
        
        features['avg_price_slope'] = (features['ask_price_slope'] + features['bid_price_slope']) / 2
        features['price_slope_asymmetry'] = features['ask_price_slope'] - features['bid_price_slope']
        
        features['depth_ratio_1_5'] = features['market_depth_level_1'] / (features['market_depth_level_5'] + 1e-8)
        features['depth_ratio_5_10'] = features['market_depth_level_5'] / (features['market_depth_level_10'] + 1e-8)
        
        # Logarithmic features
        log_features = ['order_size_fiat', 'market_depth_level_1', 'market_depth_level_5', 
                       'market_depth_level_10', 'trade_volume_1m']
        
        for feature in log_features:
            if feature in features:
                features[f'{feature}_log'] = np.log1p(features[feature])
    
    def predict_slippage(self, features: Dict) -> float:
        """Predict slippage using the trained model"""
        try:
            # Convert features to the correct format with proper DataFrame
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            # Create DataFrame with proper column names to avoid sklearn warnings
            feature_df = pd.DataFrame([feature_vector], columns=self.feature_columns)
            
            if self.use_onnx:
                # ONNX prediction
                input_name = self.onnx_session.get_inputs()[0].name
                prediction = self.onnx_session.run(None, {input_name: feature_df.values.astype(np.float32)})[0]
                return float(prediction[0])
            else:
                # sklearn prediction with proper DataFrame
                feature_df_scaled = pd.DataFrame(
                    self.scaler.transform(feature_df), 
                    columns=self.feature_columns
                )
                prediction = self.model.predict(feature_df_scaled)
                return float(prediction[0])
                
        except Exception as e:
            print(f"Error predicting slippage: {e}")
            return 0.01  # Return a conservative default
    
    def calculate_total_cost(self, quote_price: float, predicted_slippage: float, 
                           exchange_name: str, trade_side: str, amount: float) -> Tuple[float, Dict]:
        """Calculate total execution cost including all fees"""
        
        # Get exchange fees
        fees = self.exchange_fees.get(exchange_name, {'maker': 0.001, 'taker': 0.001})
        trading_fee = fees['taker']  # Assume market orders (taker fees)
        
        # Calculate slipped price
        if trade_side.lower() == 'buy':
            actual_price = quote_price * (1 + predicted_slippage)
        else:
            actual_price = quote_price * (1 - predicted_slippage)
        
        # Calculate costs
        gross_cost = amount * actual_price
        fee_cost = gross_cost * trading_fee
        total_cost = gross_cost + fee_cost
        
        fee_breakdown = {
            'trading_fee': fee_cost,
            'slippage_cost': amount * abs(actual_price - quote_price),
            'total_fees': fee_cost
        }
        
        return total_cost, fee_breakdown
    
    async def get_best_execution_venue(self, trade_request: TradeRequest) -> List[ExchangeQuote]:
        """Find the best execution venue for a trade"""
        quotes = []
        
        for exchange_name in self.exchanges.keys():
            try:
                # Get market data
                market_data = self.get_market_data(exchange_name, trade_request.symbol)
                if not market_data:
                    continue
                
                # Calculate features
                features = self.calculate_features(market_data, trade_request)
                if not features:
                    continue
                
                # Predict slippage
                predicted_slippage = self.predict_slippage(features)
                
                # Get quote price
                order_book = market_data['order_book']
                if trade_request.side.lower() == 'buy':
                    quote_price = order_book['asks'][0][0]
                else:
                    quote_price = order_book['bids'][0][0]
                
                # Calculate total cost
                total_cost, fees = self.calculate_total_cost(
                    quote_price, predicted_slippage, exchange_name, 
                    trade_request.side, trade_request.amount
                )
                
                # Calculate recommendation score (lower cost = higher score)
                recommendation_score = 1.0 / (total_cost + 1e-8)
                
                quote = ExchangeQuote(
                    exchange=exchange_name,
                    symbol=trade_request.symbol,
                    quote_price=quote_price,
                    predicted_slippage=predicted_slippage,
                    total_cost=total_cost,
                    recommendation_score=recommendation_score,
                    fees=fees
                )
                
                quotes.append(quote)
                
            except Exception as e:
                print(f"Error processing {exchange_name}: {e}")
                continue
        
        # Sort by total cost (ascending)
        quotes.sort(key=lambda x: x.total_cost)
        
        return quotes

class TradeCostAPI:
    """REST API for trade cost predictions"""
    
    def __init__(self, predictor: RealTimePredictor):
        self.predictor = predictor
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_routes(self):
        """Setup API routes"""

        @self.app.route('/', methods=['GET'])
        def home():
            """Home page with API documentation"""
            return jsonify({
                'service': 'AI Trade Cost Predictor API',
                'version': '1.0.0',
                'status': 'active',
                'endpoints': {
                    'GET /health': 'Health check',
                    'POST /predict': 'Get trade cost predictions for all exchanges',
                    'POST /compare': 'Compare specific exchanges'
                },
                'example_request': {
                    'symbol': 'BTC/USDT',
                    'amount': 1.0,
                    'side': 'sell'
                },
                'documentation': 'Send POST requests to /predict with JSON body containing symbol, amount, and side'
            })
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'exchanges': list(self.predictor.exchanges.keys()),
                'model_type': 'ONNX' if self.predictor.use_onnx else 'sklearn',
                'feature_count': len(self.predictor.feature_columns)
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint"""
            try:
                data = request.json
                
                # Validate request
                if not all(key in data for key in ['symbol', 'amount', 'side']):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Create trade request
                trade_request = TradeRequest(
                    symbol=data['symbol'],
                    amount=float(data['amount']),
                    side=data['side'],
                    amount_usd=data.get('amount_usd')
                )
                
                # Get predictions
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                quotes = loop.run_until_complete(
                    self.predictor.get_best_execution_venue(trade_request)
                )
                
                if not quotes:
                    return jsonify({'error': 'No quotes available'}), 500
                
                # Format response
                response = {
                    'request': {
                        'symbol': trade_request.symbol,
                        'amount': trade_request.amount,
                        'side': trade_request.side
                    },
                    'timestamp': datetime.now().isoformat(),
                    'best_venue': quotes[0].exchange,
                    'quotes': []
                }
                
                for quote in quotes:
                    quote_data = {
                        'exchange': quote.exchange,
                        'quote_price': quote.quote_price,
                        'predicted_slippage': quote.predicted_slippage,
                        'predicted_slippage_pct': quote.predicted_slippage * 100,
                        'total_cost': quote.total_cost,
                        'fees': quote.fees,
                        'savings_vs_worst': quotes[-1].total_cost - quote.total_cost if len(quotes) > 1 else 0
                    }
                    response['quotes'].append(quote_data)
                
                self.logger.info(f"Prediction request: {trade_request.symbol} {trade_request.amount} {trade_request.side}")
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/compare', methods=['POST'])
        def compare_exchanges():
            """Compare specific exchanges"""
            try:
                data = request.json
                trade_request = TradeRequest(
                    symbol=data['symbol'],
                    amount=float(data['amount']),
                    side=data['side']
                )
                
                exchanges = data.get('exchanges', list(self.predictor.exchanges.keys()))
                
                # Filter predictor exchanges
                original_exchanges = self.predictor.exchanges.copy()
                self.predictor.exchanges = {k: v for k, v in original_exchanges.items() if k in exchanges}
                
                # Get quotes
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                quotes = loop.run_until_complete(
                    self.predictor.get_best_execution_venue(trade_request)
                )
                
                # Restore original exchanges
                self.predictor.exchanges = original_exchanges
                
                if not quotes:
                    return jsonify({'error': 'No quotes available'}), 500
                
                # Calculate comparison metrics
                best_cost = min(quote.total_cost for quote in quotes)
                worst_cost = max(quote.total_cost for quote in quotes)
                potential_savings = worst_cost - best_cost
                
                response = {
                    'comparison_summary': {
                        'best_exchange': quotes[0].exchange,
                        'best_cost': best_cost,
                        'worst_cost': worst_cost,
                        'potential_savings': potential_savings,
                        'savings_percentage': (potential_savings / worst_cost) * 100 if worst_cost > 0 else 0
                    },
                    'detailed_quotes': [
                        {
                            'exchange': q.exchange,
                            'total_cost': q.total_cost,
                            'predicted_slippage_pct': q.predicted_slippage * 100,
                            'quote_price': q.quote_price,
                            'fees': q.fees
                        } for q in quotes
                    ]
                }
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Comparison error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        print(f"🚀 Starting Trade Cost Prediction API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Example usage and testing
def test_predictor():
    """Test the prediction system"""
    print("🧪 Testing Real-time Trade Cost Predictor")
    print("="*50)
    
    # Initialize predictor with sklearn models (not ONNX)
    predictor = RealTimePredictor(model_path='models', use_onnx=False)
    
    # Test trade request
    trade_request = TradeRequest(
        symbol='BTC/USDT',
        amount=1.0,  # 1 BTC
        side='sell'
    )
    
    print(f"📊 Testing trade: {trade_request.side} {trade_request.amount} {trade_request.symbol}")
    
    # Get quotes
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    quotes = loop.run_until_complete(predictor.get_best_execution_venue(trade_request))
    
    if quotes:
        print(f"\n🏆 Best execution venue: {quotes[0].exchange}")
        print(f"💰 Predicted total cost: ${quotes[0].total_cost:,.2f}")
        print(f"📉 Predicted slippage: {quotes[0].predicted_slippage*100:.4f}%")
        
        print(f"\n📋 All quotes:")
        for i, quote in enumerate(quotes):
            savings = quotes[-1].total_cost - quote.total_cost if len(quotes) > 1 else 0
            print(f"{i+1}. {quote.exchange}: ${quote.total_cost:,.2f} "
                  f"(slippage: {quote.predicted_slippage*100:.4f}%, "
                  f"savings: ${savings:,.2f})")
    else:
        print("❌ No quotes received")

def main():
    """Main function to run the system"""
    print("🚀 AI-Powered Trade Execution Cost Predictor - Deployment")
    print("="*60)
    
    try:
        # Initialize predictor with sklearn models (disable ONNX by default)
        predictor = RealTimePredictor(model_path='models', use_onnx=False)
        
        # Test the system
        test_predictor()
        
        # Start API server
        api = TradeCostAPI(predictor)
        
        print("\n🌐 Starting REST API...")
        print("Endpoints:")
        print("  GET  /health - Health check")
        print("  POST /predict - Get trade cost predictions")
        print("  POST /compare - Compare specific exchanges")
        print("\nExample request:")
        print("""{
  "symbol": "BTC/USDT",
  "amount": 1.0,
  "side": "sell"
}""")
        
        # Run API server
        api.run(port=5000, debug=False)
        
    except Exception as e:
        print(f"❌ Startup error: {e}")

if __name__ == "__main__":
    main()