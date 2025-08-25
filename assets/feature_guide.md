
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
