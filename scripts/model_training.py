"""
Streamlined AI Trade Execution Cost Predictor - Model Training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        
        # Size-related interactions (using existing depth_utilization)
        df_new['size_spread_interaction'] = df_new['order_size_fiat'] * df_new['spread_percentage']
        df_new['size_volatility_interaction'] = df_new['order_size_fiat'] * df_new['trade_volatility_1m']
        df_new['depth_utilization_squared'] = df_new['depth_utilization'] ** 2
        
        # Market microstructure interactions
        df_new['imbalance_spread'] = df_new['order_book_imbalance'] * df_new['spread_percentage']
        df_new['depth_spread_ratio'] = df_new['market_depth_level_5'] / (df_new['spread_percentage'] + 1e-8)
        df_new['volatility_spread'] = df_new['trade_volatility_1m'] * df_new['spread_percentage']
        
        # Price impact features
        df_new['avg_price_slope'] = (df_new['ask_price_slope'] + df_new['bid_price_slope']) / 2
        df_new['price_slope_asymmetry'] = df_new['ask_price_slope'] - df_new['bid_price_slope']
        
        # Depth ratios at different levels
        df_new['depth_ratio_1_5'] = df_new['market_depth_level_1'] / (df_new['market_depth_level_5'] + 1e-8)
        df_new['depth_ratio_5_10'] = df_new['market_depth_level_5'] / (df_new['market_depth_level_10'] + 1e-8)
        
        return df_new
    
    @staticmethod
    def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        
        log_features = ['order_size_fiat', 'market_depth_level_1', 'market_depth_level_5', 
                       'market_depth_level_10', 'trade_volume_1m']
        
        for feature in log_features:
            if feature in df_new.columns:
                df_new[f'{feature}_log'] = np.log1p(df_new[feature])
        
        return df_new

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.results = {}
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'slippage_percentage'):
        df_clean = df.dropna(subset=[target_col]).copy()
        
        # Feature engineering
        fe = FeatureEngineer()
        df_clean = fe.create_interaction_features(df_clean)
        df_clean = fe.create_log_features(df_clean)
        
        # Select features
        exclude_cols = ['exchange', 'symbol', 'original_symbol', 'timestamp', target_col]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Clean features
        for col in feature_cols:
            if df_clean[col].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            elif df_clean[col].dtype == 'bool':
                df_clean[col] = df_clean[col].astype(int)
        
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
        df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], 0)
        
        for col in feature_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
        
        self.feature_columns = feature_cols
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        return X, y
    
    def train_lightgbm(self, X_train, X_test, y_train, y_test):
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                     columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                    columns=X_test.columns, index=X_test.index)
        
        self.scalers['lightgbm'] = scaler
        
        param_grid = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.05, 0.1, 0.2],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'min_data_in_leaf': [10, 20, 50]
        }
        
        lgb_model = lgb.LGBMRegressor(objective='regression', metric='mae', 
                                     boosting_type='gbdt', verbose=-1, random_state=42)
        
        grid_search = GridSearchCV(lgb_model, param_grid, cv=3, 
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_lgb = grid_search.best_estimator_
        
        y_pred_train = best_lgb.predict(X_train_scaled)
        y_pred_test = best_lgb.predict(X_test_scaled)
        
        results = self.evaluate_model(y_train, y_pred_train, y_test, y_pred_test, "LightGBM")
        self.models['lightgbm'] = best_lgb
        self.results['lightgbm'] = results
        
        return best_lgb
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                     columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                    columns=X_test.columns, index=X_test.index)
        
        self.scalers['xgboost'] = scaler
        
        param_grid = {
            'max_depth': [3, 6, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_xgb = grid_search.best_estimator_
        
        y_pred_train = best_xgb.predict(X_train_scaled)
        y_pred_test = best_xgb.predict(X_test_scaled)
        
        results = self.evaluate_model(y_train, y_pred_train, y_test, y_pred_test, "XGBoost")
        self.models['xgboost'] = best_xgb
        self.results['xgboost'] = results
        
        return best_xgb
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                     columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                    columns=X_test.columns, index=X_test.index)
        
        self.scalers['random_forest'] = scaler
        
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(rf_model, param_grid, cv=3,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_rf = grid_search.best_estimator_
        
        y_pred_train = best_rf.predict(X_train_scaled)
        y_pred_test = best_rf.predict(X_test_scaled)
        
        results = self.evaluate_model(y_train, y_pred_train, y_test, y_pred_test, "Random Forest")
        self.models['random_forest'] = best_rf
        self.results['random_forest'] = results
        
        return best_rf
    
    def evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test, model_name):
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        def safe_mape(y_true, y_pred):
            mask = np.abs(y_true) > 1e-6
            if mask.sum() == 0:
                return 0.0
            y_true_nonzero = y_true[mask]
            y_pred_nonzero = y_pred[mask]
            return np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
        
        test_mape = safe_mape(y_test, y_pred_test)
        
        results = {
            'model': model_name,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'overfitting_score': train_mae / test_mae if test_mae > 0 else 1.0
        }
        
        print(f"{model_name} - Test MAE: {test_mae:.6f}, Test R²: {test_r2:.4f}, MAPE: {test_mape:.2f}%")
        
        return results
    
    def plot_comparison(self):
        if not self.results:
            return
        
        comparison_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=14)
        
        sns.barplot(x=comparison_df.index, y=comparison_df['test_mae'].astype(float), 
                   ax=axes[0, 0], palette='viridis')
        axes[0,0].set_title('Test MAE')
        axes[0,0].tick_params(axis='x', rotation=15)
        
        sns.barplot(x=comparison_df.index, y=comparison_df['test_r2'].astype(float), 
                   ax=axes[0, 1], palette='plasma')
        axes[0,1].set_title('Test R²')
        axes[0,1].tick_params(axis='x', rotation=15)
        
        sns.barplot(x=comparison_df.index, y=comparison_df['overfitting_score'].astype(float), 
                   ax=axes[1, 0], palette='magma')
        axes[1,0].set_title('Overfitting Score')
        axes[1,0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        axes[1,0].tick_params(axis='x', rotation=15)
        
        sns.barplot(x=comparison_df.index, y=comparison_df['test_mape'].astype(float), 
                   ax=axes[1, 1], palette='cividis')
        axes[1,1].set_title('Test MAPE (%)')
        axes[1,1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance(self, model_name='lightgbm', top_n=15):
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features, palette='rocket')
        plt.title(f'Top {top_n} Features - {model_name.title()}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def save_models(self, base_path='models'):
        import os
        os.makedirs(base_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(base_path, f"{model_name}_model.pkl")
            scaler_path = os.path.join(base_path, f"{model_name}_scaler.pkl")
            
            joblib.dump(model, model_path)
            joblib.dump(self.scalers[model_name], scaler_path)
        
        features_path = os.path.join(base_path, "feature_columns.pkl")
        joblib.dump(self.feature_columns, features_path)
        print(f"Models saved to {base_path}/")

def main():
    try:
        df = pd.read_csv('enhanced_trade_cost_dataset.csv')
        print(f"Loaded dataset: {df.shape}")
    except FileNotFoundError:
        print("Dataset not found. Run the data generation script first.")
        return
    
    trainer = ModelTrainer()
    X, y = trainer.prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {len(trainer.feature_columns)}")
    
    print("\nTraining models...")
    trainer.train_lightgbm(X_train, X_test, y_train, y_test)
    trainer.train_xgboost(X_train, X_test, y_train, y_test)
    trainer.train_random_forest(X_train, X_test, y_train, y_test)
    
    trainer.plot_comparison()
    trainer.feature_importance()
    trainer.save_models()
    
    best_result = trainer.results['lightgbm']
    print(f"\nBest model (LightGBM): Test MAE {best_result['test_mae']:.6f}")
    print(f"For $10k trade, avg error: ${best_result['test_mae'] * 100:.2f}")

if __name__ == "__main__":
    main()