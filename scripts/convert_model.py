import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np

# This script only needs to be run ONCE to get the scaler parameters.

print("--- Extracting RobustScaler Parameters for Rust ---")

# --- Load data and feature names (same as before) ---
data_path = r"D:\AAL\Coding\piton\ChainPrice\enhanced_trade_cost_dataset.csv"
feature_names_path = r"D:\AAL\Coding\piton\ChainPrice\models\feature_columns.pkl"
df = pd.read_csv(data_path)
feature_names = joblib.load(feature_names_path)

# --- Replicate data preparation ---
class FeatureEngineer: # We need this class again
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy(); df_new['size_spread_interaction'] = df_new['order_size_fiat'] * df_new['spread_percentage']; df_new['size_volatility_interaction'] = df_new['order_size_fiat'] * df_new['trade_volatility_1m']; df_new['depth_utilization_squared'] = df_new['depth_utilization'] ** 2; df_new['imbalance_spread'] = df_new['order_book_imbalance'] * df_new['spread_percentage']; df_new['depth_spread_ratio'] = df_new['market_depth_level_5'] / (df_new['spread_percentage'] + 1e-8); df_new['volatility_spread'] = df_new['trade_volatility_1m'] * df_new['spread_percentage']; df_new['avg_price_slope'] = (df_new['ask_price_slope'] + df_new['bid_price_slope']) / 2; df_new['price_slope_asymmetry'] = df_new['ask_price_slope'] - df_new['bid_price_slope']; df_new['depth_ratio_1_5'] = df_new['market_depth_level_1'] / (df_new['market_depth_level_5'] + 1e-8); df_new['depth_ratio_5_10'] = df_new['market_depth_level_5'] / (df_new['market_depth_level_10'] + 1e-8); return df_new
    @staticmethod
    def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy(); log_features = ['order_size_fiat', 'market_depth_level_1', 'market_depth_level_5', 'market_depth_level_10', 'trade_volume_1m'];
        for feature in log_features:
            if feature in df_new.columns: df_new[f'{feature}_log'] = np.log1p(df_new[feature])
        return df_new

fe = FeatureEngineer()
df_engineered = fe.create_interaction_features(df)
df_engineered = fe.create_log_features(df_engineered)
X_train = df_engineered[feature_names].copy()
X_train.replace([np.inf, -np.inf], np.nan, inplace=True); X_train.fillna(0, inplace=True)

# --- Fit the scaler ---
scaler = RobustScaler()
scaler.fit(X_train)
print("Scaler fitted successfully.")

# --- Print the parameters for Rust ---
print("\nCOPY THE FOLLOWING ARRAYS INTO YOUR Rust lib.rs FILE:\n")

# Print the centers (medians)
print("const ROBUST_SCALER_CENTER: [f32; 39] = [")
center_str = ", ".join([f"{val:.8f}" for val in scaler.center_])
print(f"    {center_str}")
print("];")

# Print the scales (interquartile ranges)
print("\nconst ROBUST_SCALER_SCALE: [f32; 39] = [")
scale_str = ", ".join([f"{val:.8f}" for val in scaler.scale_])
print(f"    {scale_str}")
print("];")