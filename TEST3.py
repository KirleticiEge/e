import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm
import datetime as dt

warnings.filterwarnings("ignore")

class PairsTradingML:
    def __init__(self, data_folder="S&P500 Data", output_folder="ML_Models"):
        """
        Initialize the pairs trading model with folder paths
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # For storing pair data
        self.pairs_data = {}
        self.ml_features = {}
        self.ml_models = {}
        
        # Load stationary pairs
        self.load_stationary_pairs()

    def run_enhanced_workflow(self, ticker1="ISCTR", ticker2="YKBNK"):
        """
        ISCTR-YKBNK gibi ADF p-değeri çok düşük olan (en mean-revertive) çifte
        gelişmiş özelliklerin yaratılması, model eğitimi ve backtest sürecinin çalıştırılması.
        """
        pair_key = f"{ticker1}_{ticker2}"
        
        # 1) Veri hazırlığı (zaten prepare_pair_data() içinde çağrılmalı)
        if pair_key not in self.pairs_data:
            print(f"Veri bulunamadı: Lütfen önce prepare_pair_data() çalıştırın.")
            return
        
        # 2) Gelişmiş feature engineering
        self.enhanced_feature_engineering(ticker1, ticker2)
        
        # 3) Model eğitimine geçmeden önce yeni feature'ları tekrar gözden geçirmek isterseniz
        # dilediğiniz gibi ek düzenlemeler yapabilirsiniz. Ardından model eğitimi:
        self.train_entry_model(pair_key)
        self.train_exit_model(pair_key)
        self.train_stop_loss_model(pair_key)
        
        # 4) Backtest
        if pair_key in self.ml_models and self.ml_models[pair_key]['entry'] is not None:
            self.backtest_strategy(ticker1, ticker2)
        else:
            print("Model bulunamadı veya eğitim başarısız oldu, backtest yapılamıyor.")
        
        print(f"Gelişmiş iş akışı (enhanced workflow) tamamlandı: {pair_key}")
        
    def load_stationary_pairs(self, filepath="Data2/stationary_spread_pairs.csv"):
        """
        Load the pre-identified stationary pairs
        """
        self.stationary_pairs = pd.read_csv(filepath)
        print(f"Loaded {len(self.stationary_pairs)} stationary pairs")
        
    def load_price_data(self):
        """
        Load all price data from CSV files
        """
        files = [f for f in os.listdir(self.data_folder) if f.endswith(".csv")]
        dfs = {}
        
        print("Loading stock price data...")
        for file in tqdm(files):
            ticker = file.split(".")[0]
            file_path = os.path.join(self.data_folder, file)
            
            try:
                # Skip the first two rows and read the data
                df = pd.read_csv(file_path, skiprows=2)
                
                # Select only the first 6 columns if they exist
                if df.shape[1] >= 6:
                    df = df.iloc[:, :6]
                else:
                    print(f"{ticker} file doesn't have expected columns.")
                    continue
                
                # Rename columns
                df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
                
                # Convert date to datetime and set as index
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Rename Close column with ticker name
                df.rename(columns={"Close": ticker}, inplace=True)
                
                # Store in dictionary
                dfs[ticker] = df[[ticker]]
                
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
        
        self.price_data = dfs
        print(f"Successfully loaded {len(dfs)} stock price datasets")
        
    def prepare_pair_data(self):
        """
        Prepare data for each pair identified in stationary_pairs.csv
        """
        print("Preparing pair data with advanced features...")
        for idx, row in tqdm(self.stationary_pairs.iterrows()):
            ticker1 = row['Ticker1']
            ticker2 = row['Ticker2']
            hedge_ratio = row['Hedge_Ratio']
            
            # Skip if either ticker is missing from our data
            if ticker1 not in self.price_data or ticker2 not in self.price_data:
                print(f"Skipping pair {ticker1}-{ticker2}: Data missing")
                continue
            
            # Get price data for both tickers
            price1 = self.price_data[ticker1]
            price2 = self.price_data[ticker2]
            
            # Join the data
            pair_data = pd.concat([price1, price2], axis=1).dropna()
            
            if len(pair_data) < 100:  # Skip pairs with insufficient data
                print(f"Skipping pair {ticker1}-{ticker2}: Insufficient data points")
                continue
                
            # Calculate spread
            pair_data['spread'] = pair_data[ticker1] - hedge_ratio * pair_data[ticker2]
            
            # Get additional price and volume data
            for ticker in [ticker1, ticker2]:
                try:
                    full_data = pd.read_csv(("Data/BIST100 Data/"+ticker+".csv"), skiprows=2)
                    full_data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
                    full_data['Date'] = pd.to_datetime(full_data['Date'])
                    full_data.set_index('Date', inplace=True)
                    
                    # Add high, low, open, and volume data to pair_data
                    pair_data[f'{ticker}_high'] = full_data['High']
                    pair_data[f'{ticker}_low'] = full_data['Low']  
                    pair_data[f'{ticker}_open'] = full_data['Open']
                    pair_data[f'{ticker}_volume'] = full_data['Volume']
                except Exception as e:
                    print(f"Couldn't load additional data for {ticker}: {e}")
            
            # Store the prepared pair data
            self.pairs_data[f"{ticker1}_{ticker2}"] = pair_data
            
            # Create features for this pair
            self.create_features(ticker1, ticker2, hedge_ratio)
    
    def create_features(self, ticker1, ticker2, hedge_ratio):
        """
        Create advanced features for the pair
        """
        pair_key = f"{ticker1}_{ticker2}"
        
        if pair_key not in self.pairs_data:
            print(f"No data available for {pair_key}")
            return
            
        df = self.pairs_data[pair_key].copy()
        
        # ------ Basic spread features ------
        # Calculate Z-score with different windows
        for window in [10, 20, 50]:
            mean = df['spread'].rolling(window=window).mean()
            std = df['spread'].rolling(window=window).std()
            df[f'zscore_{window}d'] = (df['spread'] - mean) / std
        
        # ------ Volatility features ------
        # Calculate historical volatility for each ticker
        for ticker in [ticker1, ticker2]:
            if f'{ticker}_high' in df.columns and f'{ticker}_low' in df.columns:
                # Calculate True Range
                df[f'{ticker}_tr'] = np.maximum(
                    df[f'{ticker}_high'] - df[f'{ticker}_low'],
                    np.maximum(
                        abs(df[f'{ticker}_high'] - df[ticker].shift(1)),
                        abs(df[f'{ticker}_low'] - df[ticker].shift(1))
                    )
                )
                # Average True Range
                df[f'{ticker}_atr_14'] = df[f'{ticker}_tr'].rolling(14).mean()
        
        # Volatility ratio between the pair
        if f'{ticker1}_atr_14' in df.columns and f'{ticker2}_atr_14' in df.columns:
            df['vol_ratio'] = df[f'{ticker1}_atr_14'] / df[f'{ticker2}_atr_14']
        
        # ------ Volume features ------
        # Volume ratio and imbalance
        if f'{ticker1}_volume' in df.columns and f'{ticker2}_volume' in df.columns:
            # Volume ratio
            df['volume_ratio'] = df[f'{ticker1}_volume'] / df[f'{ticker2}_volume']
            
            # Volume moving averages
            for window in [5, 10, 20]:
                df[f'{ticker1}_vol_ma_{window}'] = df[f'{ticker1}_volume'].rolling(window).mean()
                df[f'{ticker2}_vol_ma_{window}'] = df[f'{ticker2}_volume'].rolling(window).mean()
                
            # Volume imbalance (unusual volume in one stock vs the other)
            df['volume_imbalance'] = (
                df[f'{ticker1}_volume'] / df[f'{ticker1}_vol_ma_20'] - 
                df[f'{ticker2}_volume'] / df[f'{ticker2}_vol_ma_20']
            )
        
        # ------ Technical indicators ------
        # RSI for both stocks
        for ticker in [ticker1, ticker2]:
            if ticker in df.columns:

                    # If talib fails, implement a simple RSI
                    delta = df[ticker].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    df[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        # RSI divergence between stocks
        if f'{ticker1}_rsi' in df.columns and f'{ticker2}_rsi' in df.columns:
            df['rsi_divergence'] = df[f'{ticker1}_rsi'] - df[f'{ticker2}_rsi']
        
        # ------ Market regime indicators ------
        # Simple market regime based on spread momentum
        df['spread_return'] = df['spread'].pct_change()
        df['spread_return_ma'] = df['spread_return'].rolling(window=20).mean()
        
        # Define market regime (3 = high volatility, 2 = trending, 1 = mean reverting, 0 = undefined)
        df['spread_vol'] = df['spread_return'].rolling(window=20).std()
        df['market_regime'] = 0
        
        # High volatility regime
        high_vol_mask = df['spread_vol'] > df['spread_vol'].rolling(window=60).mean() * 1.5
        df.loc[high_vol_mask, 'market_regime'] = 3
        
        # Trending regime
        trending_mask = (df['spread_return_ma'].abs() > 0.0015) & (~high_vol_mask)
        df.loc[trending_mask, 'market_regime'] = 2
        
        # Mean reverting regime
        mean_rev_mask = (~high_vol_mask) & (~trending_mask) & (df['spread_vol'] > 0)
        df.loc[mean_rev_mask, 'market_regime'] = 1
        
        # ------ Signal features ------
        # Generate signals based on z-score thresholds
        df['entry_long'] = (df['zscore_20d'] < -2) & (df['zscore_20d'].shift(1) >= -2)
        df['entry_short'] = (df['zscore_20d'] > 2) & (df['zscore_20d'].shift(1) <= 2)
        
        # Exit signals (z-score crosses back over 0)
        df['exit_long'] = (df['zscore_20d'] > 0) & (df['zscore_20d'].shift(1) <= 0)
        df['exit_short'] = (df['zscore_20d'] < 0) & (df['zscore_20d'].shift(1) >= 0)
        
        # Calculate forward returns for ML label creation
        for days in [1, 3, 5, 10]:
            df[f'spread_fwd_ret_{days}d'] = df['spread'].pct_change(periods=days).shift(-days)
        
        # Drop NaN values after creating all features
        df.dropna(inplace=True)
        
        # Store the feature-engineered data
        self.ml_features[pair_key] = df
    
    def create_ml_labels(self, pair_key, threshold=0.01):
        """
        Create ML labels for entry, exit and stop-loss decisions
        """
        if pair_key not in self.ml_features:
            print(f"No feature data for {pair_key}")
            return None
            
        df = self.ml_features[pair_key].copy()
        
        # Entry signal label (1 = go long, -1 = go short, 0 = do nothing)
        df['entry_signal'] = 0
        df.loc[(df['zscore_20d'] < -2) & (df['spread_fwd_ret_5d'] > threshold), 'entry_signal'] = 1  # Long entry
        df.loc[(df['zscore_20d'] > 2) & (df['spread_fwd_ret_5d'] < -threshold), 'entry_signal'] = -1  # Short entry
        
        # Exit signal label (1 = exit long, -1 = exit short, 0 = hold)
        df['exit_signal'] = 0
        df.loc[(df['zscore_20d'] > -0.5) & (df['zscore_20d'].shift(1) <= -0.5), 'exit_signal'] = 1  # Exit long
        df.loc[(df['zscore_20d'] < 0.5) & (df['zscore_20d'].shift(1) >= 0.5), 'exit_signal'] = -1  # Exit short
        
        # Stop-loss trigger (1 = trigger stop loss)
        df['stop_loss'] = 0
        # Stop loss when z-score moves further against position with high volatility
        df.loc[
            ((df['zscore_20d'] < -3) & (df['zscore_20d'].shift(1) > df['zscore_20d']) & (df['spread_vol'] > df['spread_vol'].rolling(30).mean())) | 
            ((df['zscore_20d'] > 3) & (df['zscore_20d'].shift(1) < df['zscore_20d']) & (df['spread_vol'] > df['spread_vol'].rolling(30).mean())),
            'stop_loss'
        ] = 1
        
        return df
    
    def select_features(self, df):
        """
        Select relevant features for the ML model
        """
        # List of features to include - adjust as needed
        feature_cols = [col for col in df.columns if any(x in col for x in [
            'zscore_', 'vol_ratio', 'volume_ratio', 'volume_imbalance', 
            '_rsi', 'rsi_divergence', 'market_regime', 'spread_vol'
        ])]
        
        return feature_cols
    
    def train_entry_model(self, pair_key):
        df = self.create_ml_labels(pair_key)
        if df is None:
            return None

        feature_cols = self.select_features(df)

        # We need 'entry_signal' for entry model
        df = df.dropna(subset=feature_cols + ['entry_signal'])
        X = df[feature_cols]
        y = df['entry_signal']

        # Replace inf or -inf with NaN, then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]

        if len(X) < 100 or sum(y != 0) < 10:
            print(f"Not enough data for {pair_key} entry model")
            return None

        # Time-based split
        train_size = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nEntry Model for {pair_key} - Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        joblib.dump(pipeline, os.path.join(self.output_folder, f"{pair_key}_entry_model.pkl"))
        return pipeline
    
    def train_exit_model(self, pair_key):
        df = self.create_ml_labels(pair_key)
        if df is None:
            return None

        feature_cols = self.select_features(df)

        # We need 'exit_signal' for exit model
        df = df.dropna(subset=feature_cols + ['exit_signal'])
        X = df[feature_cols]
        y = df['exit_signal']

        # Replace inf or -inf with NaN, then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]

        if len(X) < 100 or sum(y != 0) < 10:
            print(f"Not enough data for {pair_key} exit model")
            return None

        train_size = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nExit Model for {pair_key} - Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        joblib.dump(pipeline, os.path.join(self.output_folder, f"{pair_key}_exit_model.pkl"))
        return pipeline

    
    def train_stop_loss_model(self, pair_key):
        """
        Train ML model to predict stop-loss points
        """
        # Get data with labels
        df = self.create_ml_labels(pair_key)
        if df is None:
            return None
            
        # Select features
        feature_cols = self.select_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna(subset=feature_cols + ['stop_loss'])
        
        # Prepare feature matrix and target vector
        X = df[feature_cols]
        y = df['stop_loss']

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]
        
        # Skip if not enough samples or no stop-loss signals
        if len(df) < 100 or sum(y != 0) < 10:
            print(f"Not enough data for {pair_key} stop-loss model")
            return None
        
        # Split data - using time-based split
        train_size = int(len(df) * 0.7)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Define model pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nStop-Loss Model for {pair_key} - Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = os.path.join(self.output_folder, f"{pair_key}_stop_loss_model.pkl")
        joblib.dump(pipeline, model_path)
        
        return pipeline
    
    def train_all_models(self):
        """
        Train all models for each pair
        """
        for pair_idx, row in self.stationary_pairs.iterrows():
            ticker1 = row['Ticker1']
            ticker2 = row['Ticker2']
            pair_key = f"{ticker1}_{ticker2}"

            if pair_key in self.ml_features:
                print(f"\n======= Training models for {pair_key} =======")

                # Train entry model
                entry_model = self.train_entry_model(pair_key)

                # Train exit model
                exit_model = self.train_exit_model(pair_key)

                # Train stop-loss model
                stop_loss_model = self.train_stop_loss_model(pair_key)

                # MODELLERİ KAYDET!
                self.ml_models[pair_key] = {
                    'entry': entry_model,
                    'exit': exit_model,
                    'stop_loss': stop_loss_model
                }

                print(f"✅ {pair_key} modeli başarıyla kaydedildi.")

    
    def visualize_pair(self, ticker1, ticker2):
        """
        Visualize pair relationship and signals
        """
        pair_key = f"{ticker1}_{ticker2}"
        if pair_key not in self.ml_features:
            print(f"No data available for {pair_key}")
            return
            
        df = self.ml_features[pair_key].copy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        
        # Plot 1: Individual stock prices (normalized)
        ax1 = axes[0]
        norm_price1 = df[ticker1] / df[ticker1].iloc[0]
        norm_price2 = df[ticker2] / df[ticker2].iloc[0]
        ax1.plot(df.index, norm_price1, label=f"{ticker1} (normalized)")
        ax1.plot(df.index, norm_price2, label=f"{ticker2} (normalized)")
        ax1.set_title(f"Normalized Prices: {ticker1} vs {ticker2}")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Spread
        ax2 = axes[1]
        ax2.plot(df.index, df['spread'])
        ax2.set_title(f"Price Spread: {ticker1} - {row['Hedge_Ratio']} * {ticker2}")
        ax2.grid(True)
        
        # Plot 3: Z-score with entry/exit thresholds
        ax3 = axes[2]
        ax3.plot(df.index, df['zscore_20d'])
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(2, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(-2, color='green', linestyle='--', alpha=0.7)
        
        # Add entry and exit signals
        long_entries = df[df['entry_signal'] == 1].index
        short_entries = df[df['entry_signal'] == -1].index
        long_exits = df[df['exit_signal'] == 1].index
        short_exits = df[df['exit_signal'] == -1].index
        
        ax3.scatter(long_entries, df.loc[long_entries, 'zscore_20d'], 
                    marker='^', color='green', s=100, label='Long Entry')
        ax3.scatter(short_entries, df.loc[short_entries, 'zscore_20d'], 
                    marker='v', color='red', s=100, label='Short Entry')
        ax3.scatter(long_exits, df.loc[long_exits, 'zscore_20d'], 
                    marker='o', color='blue', s=80, label='Long Exit')
        ax3.scatter(short_exits, df.loc[short_exits, 'zscore_20d'], 
                    marker='o', color='purple', s=80, label='Short Exit')
                    
        ax3.set_title(f"Z-Score (20-day) with Trading Signals")
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Market regime and volume imbalance
        ax4 = axes[3]
        if 'volume_imbalance' in df.columns:
            ax4.plot(df.index, df['volume_imbalance'], color='blue', label='Volume Imbalance')
            ax4.set_ylabel('Volume Imbalance', color='blue')
        
        # Add market regime as background color
        for regime in range(1, 4):
            regime_periods = df[df['market_regime'] == regime]
            if not regime_periods.empty:
                colors = {1: 'green', 2: 'orange', 3: 'red'}
                labels = {1: 'Mean-reverting', 2: 'Trending', 3: 'High Volatility'}
                for i in range(len(regime_periods) - 1):
                    if i == 0 or regime_periods.index[i] != regime_periods.index[i-1] + pd.Timedelta(days=1):
                        start_idx = regime_periods.index[i]
                        j = i
                        while j < len(regime_periods) - 1 and regime_periods.index[j+1] == regime_periods.index[j] + pd.Timedelta(days=1):
                            j += 1
                        end_idx = regime_periods.index[j]
                        ax4.axvspan(start_idx, end_idx, alpha=0.2, color=colors[regime], label=labels[regime] if i == 0 else "")
        
        ax4.set_title("Volume Imbalance and Market Regime")
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f"{pair_key}_analysis.png"), dpi=300)
        plt.close()

    def backtest_strategy(self, ticker1, ticker2, initial_capital=10000, trade_fraction=0.5):
        pair_key = f"{ticker1}_{ticker2}"
        if pair_key not in self.ml_features or pair_key not in self.ml_models:
            print(f"No data or models available for {pair_key}")
            return

        df = self.ml_features[pair_key].copy()

        # Load models
        entry_model = self.ml_models[pair_key]['entry']
        exit_model = self.ml_models[pair_key]['exit']
        stop_loss_model = self.ml_models[pair_key]['stop_loss']

        if entry_model is None or exit_model is None:
            print(f"Missing models for {pair_key}")
            return

        # Select features and generate predictions
        feature_cols = self.select_features(df)
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        df = df.loc[X.index]  # Align df with X

        df['entry_pred'] = entry_model.predict(X)
        df['exit_pred'] = exit_model.predict(X)
        df['stop_loss_pred'] = stop_loss_model.predict(X) if stop_loss_model else 0

        # Backtest simulation with dynamic position sizing
        position = 0
        equity = initial_capital  # Başlangıç sermayesi: 10.000₺
        entry_price = None
        allocated_capital = 0
        df['equity'] = equity
        df['position'] = position

        for i in range(1, len(df)):
            current_spread = df.iloc[i]['spread']

            if position == 0:
                # Pozisyon alımı: giriş sinyali verildiğinde, mevcut sermayenin %50’siyle pozisyona gir
                if df.iloc[i]['entry_pred'] == 1:  # Long pozisyon
                    position = 1
                    entry_price = current_spread
                    allocated_capital = trade_fraction * equity
                elif df.iloc[i]['entry_pred'] == -1:  # Short pozisyon
                    position = -1
                    entry_price = current_spread
                    allocated_capital = trade_fraction * equity

            else:
                # Pozisyonda olduğunuz sürece çıkış ya da stop-loss sinyali geldiğinde pozisyonu kapat
                if (position == 1 and df.iloc[i]['exit_pred'] == 1) or \
                (position == -1 and df.iloc[i]['exit_pred'] == -1) or \
                (df.iloc[i]['stop_loss_pred'] == 1):
                    # Getiriyi, pozisyona girişte ayrılan sermayeye göre hesaplayın
                    if position == 1:
                        pnl = (current_spread - entry_price) / entry_price * allocated_capital
                    else:  # position == -1
                        pnl = (entry_price - current_spread) / entry_price * allocated_capital
                    equity += pnl
                    position = 0
                    entry_price = None
                    allocated_capital = 0

            df.at[df.index[i], 'equity'] = equity
            df.at[df.index[i], 'position'] = position

        # Hesaplama sonuçları
        total_return = (equity / initial_capital) - 1
        daily_returns = df['equity'].pct_change().dropna()
        annualized_return = (1 + daily_returns.mean())**252 - 1
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

        df['cummax_equity'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax_equity']) / df['cummax_equity']
        max_drawdown = df['drawdown'].min()

        print(f"\nBacktest results for {pair_key}:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")

        self.plot_backtest_results(df, ticker1, ticker2)
        return df

    def plot_backtest_results(self, df, ticker1, ticker2):
        """
        Plot backtest results including the equity curve, drawdown, and Z-score with positions.
        """
        pair_key = f"{ticker1}_{ticker2}"
        
        fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Equity Curve
        axs[0].plot(df.index, df['equity'])
        axs[0].set_title(f"Equity Curve - {pair_key}")
        axs[0].set_ylabel("Equity")
        axs[0].grid(True)
        
        # Drawdown
        axs[1].fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        axs[1].set_title("Drawdown")
        axs[1].set_ylabel("Drawdown %")
        axs[1].grid(True)
        
        # Z-Score with positions highlighted
        axs[2].plot(df.index, df['zscore_20d'])
        axs[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axs[2].axhline(y=2, color='r', linestyle='--', alpha=0.3)
        axs[2].axhline(y=-2, color='g', linestyle='--', alpha=0.3)
        
        # Highlight long positions
        long_positions = df[df['position'] == 1]
        if not long_positions.empty:
            axs[2].fill_between(long_positions.index, -2, 2, color='green', alpha=0.2)
        
        # Highlight short positions
        short_positions = df[df['position'] == -1]
        if not short_positions.empty:
            axs[2].fill_between(short_positions.index, -2, 2, color='red', alpha=0.2)
        
        axs[2].set_title("Z-Score with Positions")
        axs[2].set_ylabel("Z-Score")
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f"{pair_key}_backtest_results.png"), dpi=300)
        plt.close()

    def optimize_model_parameters(self, ticker1, ticker2):
        """
        Optimize model hyperparameters using grid search
        """
        from sklearn.model_selection import GridSearchCV
        
        pair_key = f"{ticker1}_{ticker2}"
        if pair_key not in self.ml_features:
            print(f"No feature data for {pair_key}")
            return None
        
        print(f"Optimizing model parameters for {pair_key}...")
        
        # Get data with labels
        df = self.create_ml_labels(pair_key)
        if df is None:
            return None
        
        # Select features
        feature_cols = self.select_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna(subset=feature_cols + ['entry_signal'])
        
        # Prepare feature matrix and target vector
        X = df[feature_cols]
        y = df['entry_signal']

        X = df[feature_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()

        y = y.loc[X.index]
        
        # Skip if not enough samples
        if len(df) < 100 or sum(y != 0) < 10:
            print(f"Not enough data for {pair_key} optimization")
            return None
        
        # Split data
        train_size = int(len(df) * 0.7)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Define parameter grid for GradientBoostingClassifier
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10],
            'model__subsample': [0.8, 0.9, 1.0]
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid=param_grid, 
            cv=TimeSeriesSplit(n_splits=5), 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = grid_search.predict(X_test)
        print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save best model
        best_model = grid_search.best_estimator_
        model_path = os.path.join(self.output_folder, f"{pair_key}_optimized_entry_model.pkl")
        joblib.dump(best_model, model_path)
        
        return best_model

    def feature_importance_analysis(self, ticker1, ticker2):
        """
        Analyze feature importance
        """
        pair_key = f"{ticker1}_{ticker2}"
        if pair_key not in self.ml_models:
            print(f"No models found for {pair_key}")
            return
        
        # Get models
        entry_model = self.ml_models[pair_key]['entry']
        exit_model = self.ml_models[pair_key]['exit']
        
        if entry_model is None or exit_model is None:
            print(f"Missing models for {pair_key}")
            return
        
        # Get feature names
        feature_cols = self.select_features(self.ml_features[pair_key])
        
        # Feature importance for entry model
        try:
            # Get feature importance from the model (assuming it's a tree-based model)
            if 'model' in entry_model.named_steps:
                model = entry_model.named_steps['model']
                importances = model.feature_importances_
                
                # Create a pandas Series for easy visualization
                importance_df = pd.Series(importances, index=feature_cols)
                importance_df = importance_df.sort_values(ascending=False)
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                importance_df.plot(kind='bar')
                plt.title(f"Feature Importance - {pair_key} Entry Model")
                plt.ylabel('Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_folder, f"{pair_key}_feature_importance.png"), dpi=300)
                plt.close()
                
                # Print top features
                print(f"\nTop features for {pair_key} entry model:")
                print(importance_df.head(10))
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")

    def analyze_specific_pair(self, ticker1="ISCTR", ticker2="YKBNK"):
        """
        Analyze a specific pair with additional advanced features
        """
        pair_key = f"{ticker1}_{ticker2}"
        
        print(f"\n========== Detailed Analysis for {pair_key} ==========")
        
        # Check if data is available
        if pair_key not in self.ml_features:
            print(f"No data for {pair_key}. Please run prepare_pair_data() first.")
            return
        
        # Get data
        df = self.ml_features[pair_key].copy()
        
        # Additional advanced features
        
        # 1. Add Hurst exponent calculation for mean reversion strength
        def calculate_hurst_exponent(time_series, max_lag=20):
            """Calculate Hurst exponent to measure mean reversion strength"""
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]  # Hurst exponent
        
        # Calculate Hurst exponent for rolling windows
        window_size = 60  # 60 days
        if len(df) > window_size * 2:
            df['hurst_exponent'] = df['spread'].rolling(window=window_size).apply(
                lambda x: calculate_hurst_exponent(x) if len(x.dropna()) > window_size/2 else np.nan
            )
        
        # 2. Add half-life of mean reversion calculation
        def calculate_half_life(spread):
            """Calculate half-life of mean reversion"""
            spread = np.array(spread)
            lag_spread = np.roll(spread, 1)
            lag_spread[0] = lag_spread[1]
            ret = spread - lag_spread
            lag_ret = np.roll(ret, 1)
            lag_ret[0] = lag_ret[1]
            
            # Run OLS regression
            X = lag_spread[1:]
            y = ret[1:]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            # Calculate half-life
            half_life = -np.log(2) / model.params[1] if model.params[1] < 0 else np.nan
            return half_life
        
        # Calculate half-life for rolling windows
        try:
            import statsmodels.api as sm
            df['half_life'] = df['spread'].rolling(window=window_size).apply(
                lambda x: calculate_half_life(x) if len(x.dropna()) > window_size/2 else np.nan
            )
        except ImportError:
            print("statsmodels not installed. Skipping half-life calculation.")
        
        # 3. Add correlation analysis between the two stocks
        # Calculate rolling correlation
        if ticker1 in df.columns and ticker2 in df.columns:
            df['rolling_correlation'] = df[ticker1].rolling(window=30).corr(df[ticker2])
        
        # 4. Add cross-sectional volatility ratio
        if f'{ticker1}_atr_14' in df.columns and f'{ticker2}_atr_14' in df.columns:
            df['cross_sectional_vol_ratio'] = df[f'{ticker1}_atr_14'] / df[f'{ticker2}_atr_14'].rolling(30).mean()
        
        # 5. Add mean reversion intensity score
        # Combine z-score and market regime into a single score
        df['mean_reversion_score'] = np.where(
            df['market_regime'] == 1,  # Mean-reverting regime
            df['zscore_20d'].abs() * 2,  # Double the effect of z-score
            df['zscore_20d'].abs()
        )
        
        # 6. Add ML probability scores
        # We'll need to create and train models first
        # This will be populated after model training
        df['entry_long_prob'] = np.nan
        df['entry_short_prob'] = np.nan
        
        # Save the enhanced features
        self.ml_features[pair_key] = df
        
        # Analyze the pair statistically
        self._analyze_pair_statistics(df, ticker1, ticker2)
        
        # Plot the enhanced features
        self._plot_enhanced_features(df, ticker1, ticker2)
        
        return df

    def _analyze_pair_statistics(self, df, ticker1, ticker2):
        """
        Perform statistical analysis on the pair
        """
        # Calculate key statistics
        stats = {}
        
        # Mean reversion statistics
        if 'hurst_exponent' in df.columns:
            stats['avg_hurst'] = df['hurst_exponent'].mean()
            stats['recent_hurst'] = df['hurst_exponent'].iloc[-20:].mean() if len(df) > 20 else np.nan
        
        if 'half_life' in df.columns:
            stats['avg_half_life'] = df['half_life'].mean()
            stats['recent_half_life'] = df['half_life'].iloc[-20:].mean() if len(df) > 20 else np.nan
        
        # Correlation statistics
        if 'rolling_correlation' in df.columns:
            stats['avg_correlation'] = df['rolling_correlation'].mean()
            stats['recent_correlation'] = df['rolling_correlation'].iloc[-20:].mean() if len(df) > 20 else np.nan
        
        # Z-score statistics
        if 'zscore_20d' in df.columns:
            stats['avg_zscore_abs'] = df['zscore_20d'].abs().mean()
            stats['zscore_std'] = df['zscore_20d'].std()
            stats['recent_zscore'] = df['zscore_20d'].iloc[-1] if len(df) > 0 else np.nan
        
        # Volatility statistics
        if 'vol_ratio' in df.columns:
            stats['avg_vol_ratio'] = df['vol_ratio'].mean()
            stats['recent_vol_ratio'] = df['vol_ratio'].iloc[-20:].mean() if len(df) > 20 else np.nan
        
        # Market regime statistics
        if 'market_regime' in df.columns:
            stats['percent_mean_reverting'] = (df['market_regime'] == 1).mean() * 100
            stats['percent_trending'] = (df['market_regime'] == 2).mean() * 100
            stats['percent_high_vol'] = (df['market_regime'] == 3).mean() * 100
            stats['recent_regime'] = df['market_regime'].iloc[-1] if len(df) > 0 else np.nan
        
        # Print statistics
        print("\nPair Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Store statistics for future reference
        self.pair_statistics = stats

    def _plot_enhanced_features(self, df, ticker1, ticker2):
        """
        Plot enhanced features for visualization
        """
        pair_key = f"{ticker1}_{ticker2}"
        
        # Create figure with subplots
        fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
        
        # Plot 1: Spread with volatility bands
        ax1 = axes[0]
        ax1.plot(df.index, df['spread'], label='Spread')
        
        # Add volatility bands if available
        if 'spread_vol' in df.columns:
            spread_ma = df['spread'].rolling(window=20).mean()
            spread_std = df['spread'].rolling(window=20).std()
            ax1.fill_between(df.index, 
                            spread_ma - 2*spread_std, 
                            spread_ma + 2*spread_std, 
                            color='gray', alpha=0.2, label='2-sigma band')
        
        ax1.set_title(f"Spread: {ticker1} - Hedge Ratio * {ticker2}")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Z-score with mean reversion indicators
        ax2 = axes[1]
        ax2.plot(df.index, df['zscore_20d'], label='Z-score (20d)')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(2, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(-2, color='green', linestyle='--', alpha=0.7)
        
        # Add mean reversion indicators if available
        if 'hurst_exponent' in df.columns:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df.index, df['hurst_exponent'], color='purple', alpha=0.5, label='Hurst Exponent')
            ax2_twin.axhline(0.5, color='purple', linestyle='--', alpha=0.5)
            ax2_twin.set_ylabel('Hurst Exponent', color='purple')
            ax2_twin.tick_params(axis='y', labelcolor='purple')
            ax2_twin.set_ylim(0, 1)
        
        ax2.set_title("Z-Score and Mean Reversion Strength")
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Plot 3: Market regime and volume imbalance
        ax3 = axes[2]
        ax3.plot(df.index, df['market_regime'], color='blue', label='Market Regime')
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['Undefined', 'Mean-reverting', 'Trending', 'High Vol'])
        
        # Add volume imbalance if available
        if 'volume_imbalance' in df.columns:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(df.index, df['volume_imbalance'], color='orange', alpha=0.7, label='Volume Imbalance')
            ax3_twin.set_ylabel('Volume Imbalance', color='orange')
            ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        ax3.set_title("Market Regime and Volume Analysis")
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        # Plot 4: Correlation and volatility ratio
        ax4 = axes[3]
        
        # Add correlation if available
        if 'rolling_correlation' in df.columns:
            ax4.plot(df.index, df['rolling_correlation'], color='green', label='30d Correlation')
            ax4.set_ylabel('Correlation', color='green')
            ax4.tick_params(axis='y', labelcolor='green')
            ax4.set_ylim(-1, 1)
        
        # Add volatility ratio if available
        if 'vol_ratio' in df.columns:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(df.index, df['vol_ratio'], color='red', alpha=0.7, label='Volatility Ratio')
            ax4_twin.set_ylabel('Volatility Ratio', color='red')
            ax4_twin.tick_params(axis='y', labelcolor='red')
        
        ax4.set_title("Correlation and Volatility Analysis")
        ax4.legend(loc='upper left')
        ax4.grid(True)
        
        # Plot 5: Mean reversion score and half-life
        ax5 = axes[4]
        
        # Add mean reversion score if available
        if 'mean_reversion_score' in df.columns:
            ax5.plot(df.index, df['mean_reversion_score'], color='blue', label='Mean Reversion Score')
            ax5.set_ylabel('Score', color='blue')
            ax5.tick_params(axis='y', labelcolor='blue')
        
        # Add half-life if available
        if 'half_life' in df.columns:
            ax5_twin = ax5.twinx()
            # Clip half-life to a reasonable range for visualization
            half_life_clipped = np.clip(df['half_life'], 0, 50)
            ax5_twin.plot(df.index, half_life_clipped, color='brown', alpha=0.7, label='Half-life (days)')
            ax5_twin.set_ylabel('Half-life (days)', color='brown')
            ax5_twin.tick_params(axis='y', labelcolor='brown')
        
        ax5.set_title("Mean Reversion Metrics")
        ax5.legend(loc='upper left')
        ax5.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f"{pair_key}_enhanced_analysis.png"), dpi=300)
        plt.close()

    def advanced_model_training(self, ticker1="ISCTR", ticker2="YKBNK"):
        """
        Advanced model training for a specific pair with hyperparameter tuning and cross-validation
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        
        pair_key = f"{ticker1}_{ticker2}"
        
        print(f"\n======= Advanced Model Training for {pair_key} =======")
        
        # Check if data is available
        if pair_key not in self.ml_features:
            print(f"No data for {pair_key}. Please run prepare_pair_data() first.")
            return
        
        # Get data with labels
        df = self.create_ml_labels(pair_key)
        if df is None:
            return None
        
        # Select features
        feature_cols = self.select_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna(subset=feature_cols + ['entry_signal', 'exit_signal', 'stop_loss'])
        
        # Prepare feature matrix and target vectors
        X = df[feature_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()

        y_entry = df['entry_signal']
        y_exit = df['exit_signal']
        y_stop = df['stop_loss']
        y_entry = y_entry.loc[X.index]
        y_exit = y_exit.loc[X.index]
        y_stop = y_stop.loc[X.index]
        # Skip if not enough samples
        if len(df) < 100:
            print(f"Not enough data for {pair_key} advanced training")
            return None
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Store results
        results = {}
        
        # Train and evaluate each model for each target
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # For entry signal
            print("Entry Signal Model:")
            entry_scores = cross_val_score(model, X, y_entry, cv=tscv, scoring='f1_weighted')
            print(f"  Cross-validation F1 score: {entry_scores.mean():.4f} (+/- {entry_scores.std()*2:.4f})")
            
            # For exit signal
            print("Exit Signal Model:")
            exit_scores = cross_val_score(model, X, y_exit, cv=tscv, scoring='f1_weighted')
            print(f"  Cross-validation F1 score: {exit_scores.mean():.4f} (+/- {exit_scores.std()*2:.4f})")
            
            # For stop-loss signal
            print("Stop-Loss Signal Model:")
            stop_scores = cross_val_score(model, X, y_stop, cv=tscv, scoring='f1_weighted')
            print(f"  Cross-validation F1 score: {stop_scores.mean():.4f} (+/- {stop_scores.std()*2:.4f})")
            
            # Store results
            results[model_name] = {
                'entry': entry_scores.mean(),
                'exit': exit_scores.mean(),
                'stop': stop_scores.mean(),
                'overall': (entry_scores.mean() + exit_scores.mean() + stop_scores.mean()) / 3
            }
        
        # Print overall results
        print("\nOverall Model Performance:")
        for model_name, scores in results.items():
            print(f"{model_name}: {scores['overall']:.4f} (Entry: {scores['entry']:.4f}, Exit: {scores['exit']:.4f}, Stop: {scores['stop']:.4f})")
        
        # Select the best model for each task
        best_entry_model = max(results.items(), key=lambda x: x[1]['entry'])[0]
        best_exit_model = max(results.items(), key=lambda x: x[1]['exit'])[0]
        best_stop_model = max(results.items(), key=lambda x: x[1]['stop'])[0]
        
        print(f"\nBest models selected:")
        print(f"  Entry: {best_entry_model}")
        print(f"  Exit: {best_exit_model}")
        print(f"  Stop-Loss: {best_stop_model}")
        
        # Train final models on full dataset
        final_entry_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', models[best_entry_model])
        ])
        final_entry_model.fit(X, y_entry)
        
        final_exit_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', models[best_exit_model])
        ])
        final_exit_model.fit(X, y_exit)
        
        final_stop_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', models[best_stop_model])
        ])
        final_stop_model.fit(X, y_stop)
        
        # Save final models
        joblib.dump(final_entry_model, os.path.join(self.output_folder, f"{pair_key}_advanced_entry_model.pkl"))
        joblib.dump(final_exit_model, os.path.join(self.output_folder, f"{pair_key}_advanced_exit_model.pkl"))
        joblib.dump(final_stop_model, os.path.join(self.output_folder, f"{pair_key}_advanced_stop_model.pkl"))
        
        print("Model Kaydediliyor...")
        self.ml_models[pair_key] = {
            'entry': entry_model,
            'exit': exit_model,
            'stop_loss': stop_loss_model
        }
        print(f"✅ {pair_key} modeli başarıyla kaydedildi.")
        
        # Generate prediction probabilities for the dataset
        df['entry_long_prob'] = final_entry_model.predict_proba(X)[:, 1] if 1 in final_entry_model.classes_ else 0
        df['entry_short_prob'] = final_entry_model.predict_proba(X)[:, 2] if 2 in final_entry_model.classes_ else 0
        
        # Update the dataset with probabilities
        self.ml_features[pair_key] = df
        
        return self.ml_models[pair_key]

    def enhanced_feature_engineering(self, ticker1="ISCTR", ticker2="YKBNK"):
        """
        Perform enhanced feature engineering for a specific pair
        """
        pair_key = f"{ticker1}_{ticker2}"
        
        print(f"\n======= Enhanced Feature Engineering for {pair_key} =======")
        
        # Check if data is available
        if pair_key not in self.pairs_data:
            print(f"No raw data for {pair_key}. Please run prepare_pair_data() first.")
            return
        
        # Get the raw data
        df = self.pairs_data[pair_key].copy()
        hedge_ratio = self.stationary_pairs[
            (self.stationary_pairs['Ticker1'] == ticker1) & 
            (self.stationary_pairs['Ticker2'] == ticker2)
        ]['Hedge_Ratio'].values[0]
        
        # Calculate the spread
        df['spread'] = df[ticker1] - hedge_ratio * df[ticker2]
        
        # ----------------------
        # 1. Basic spread features
        # ----------------------
        
        # Calculate Z-score with different windows
        for window in [5, 10, 20, 50]:
            mean = df['spread'].rolling(window=window).mean()
            std = df['spread'].rolling(window=window).std()
            df[f'zscore_{window}d'] = (df['spread'] - mean) / std
        
        # Z-score moving averages
        for window in [5, 10, 20]:
            df[f'zscore_20d_ma_{window}'] = df['zscore_20d'].rolling(window=window).mean()
        
        # Z-score momentum (rate of change)
        df['zscore_momentum'] = df['zscore_20d'] - df['zscore_20d'].shift(5)
        
        # Z-score acceleration
        df['zscore_acceleration'] = df['zscore_momentum'] - df['zscore_momentum'].shift(5)
        
        # ----------------------
        # 2. Volatility features
        # ----------------------
        
        # Calculate price volatility
        for ticker in [ticker1, ticker2]:
            # Price volatility
            df[f'{ticker}_returns'] = df[ticker].pct_change()
            df[f'{ticker}_vol_20d'] = df[f'{ticker}_returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Parkinson volatility (uses high-low range)
            if f'{ticker}_high' in df.columns and f'{ticker}_low' in df.columns:
                df[f'{ticker}_parkinson_vol'] = np.sqrt(
                    1 / (4 * np.log(2)) * 
                    (np.log(df[f'{ticker}_high'] / df[f'{ticker}_low']) ** 2)
                ).rolling(window=20).mean() * np.sqrt(252)
        
        # Volatility ratio between pairs
        df['vol_ratio'] = df[f'{ticker1}_vol_20d'] / df[f'{ticker2}_vol_20d']
        
        # Volatility ratio percentile
        df['vol_ratio_percentile'] = df['vol_ratio'].rolling(window=252).rank(pct=True)
        
        # Spread volatility
        df['spread_returns'] = df['spread'].pct_change()
        df['spread_vol_20d'] = df['spread_returns'].rolling(window=20).std() * np.sqrt(252)
        df['spread_vol'] = df['spread_returns'].rolling(20).std()
        # Spread volatility Z-score
        mean_spread_vol = df['spread_vol_20d'].rolling(window=60).mean()
        std_spread_vol = df['spread_vol_20d'].rolling(window=60).std()
        df['spread_vol_zscore'] = (df['spread_vol_20d'] - mean_spread_vol) / std_spread_vol
        
        # Relative volatility
        df['rel_vol_ratio'] = df['spread_vol_20d'] / (df[f'{ticker1}_vol_20d'] + df[f'{ticker2}_vol_20d'])
        
        # ----------------------
        # 3. Volume features
        # ----------------------
        
        # Volume features
        for ticker in [ticker1, ticker2]:
            if f'{ticker}_volume' in df.columns:
                # Volume rolling averages
                for window in [5, 10, 20, 50]:
                    df[f'{ticker}_vol_ma_{window}'] = df[f'{ticker}_volume'].rolling(window=window).mean()
                
                # Volume ratio to moving average
                df[f'{ticker}_vol_ratio'] = df[f'{ticker}_volume'] / df[f'{ticker}_vol_ma_20']
                
                # Volume Z-score
                vol_mean = df[f'{ticker}_volume'].rolling(window=20).mean()
                vol_std = df[f'{ticker}_volume'].rolling(window=20).std()
                df[f'{ticker}_vol_zscore'] = (df[f'{ticker}_volume'] - vol_mean) / vol_std
        
        # Volume imbalance indicators
        if f'{ticker1}_volume' in df.columns and f'{ticker2}_volume' in df.columns:
            # Simple volume imbalance
            df['volume_imbalance'] = df[f'{ticker1}_volume'] / df[f'{ticker2}_volume']
            
            # Normalized volume imbalance
            df['norm_volume_imbalance'] = (
                df[f'{ticker1}_volume'] / df[f'{ticker1}_vol_ma_20'] - 
                df[f'{ticker2}_volume'] / df[f'{ticker2}_vol_ma_20']
            )
            
            # Volume imbalance Z-score
            vol_imb_mean = df['norm_volume_imbalance'].rolling(window=20).mean()
            vol_imb_std = df['norm_volume_imbalance'].rolling(window=20).std()
            df['volume_imbalance_zscore'] = (df['norm_volume_imbalance'] - vol_imb_mean) / vol_imb_std
            
            # Volume trend
            df['volume_trend'] = df[f'{ticker1}_vol_ma_5'] / df[f'{ticker1}_vol_ma_20'] + \
                                df[f'{ticker2}_vol_ma_5'] / df[f'{ticker2}_vol_ma_20']
        
        # ----------------------
        # 4. Market regime indicators
        # ----------------------
        
        # Calculate market regime indicators
        
        # Spread momentum
        df['spread_momentum'] = df['spread'].pct_change(periods=5)
        df['spread_momentum_ma'] = df['spread_momentum'].rolling(window=20).mean()
        
        # Volatility regime
        df['volatility_regime'] = 0  # Default: normal volatility
        
        # High volatility regime (volatility significantly above its moving average)
        high_vol_mask = df['spread_vol_20d'] > df['spread_vol_20d'].rolling(window=60).mean() * 1.5
        df.loc[high_vol_mask, 'volatility_regime'] = 1  # High volatility
        
        # Low volatility regime (volatility significantly below its moving average)
        low_vol_mask = df['spread_vol_20d'] < df['spread_vol_20d'].rolling(window=60).mean() * 0.5
        df.loc[low_vol_mask, 'volatility_regime'] = -1  # Low volatility
        
        # Trend regime (based on the spread trend)
        df['trend_regime'] = 0  # Default: no trend
        
        # Uptrend regime (moving average of spread consistently increasing)
        uptrend_mask = df['spread'].rolling(window=20).mean() > df['spread'].rolling(window=50).mean()
        df.loc[uptrend_mask, 'trend_regime'] = 1  # Uptrend
        
        # Downtrend regime (moving average of spread consistently decreasing)
        downtrend_mask = df['spread'].rolling(window=20).mean() < df['spread'].rolling(window=50).mean()
        df.loc[downtrend_mask, 'trend_regime'] = -1  # Downtrend
        
        # Combined market regime
        # 1: Strong mean reversion (low volatility, no trend)
        # 2: Weak mean reversion (normal volatility, no trend)
        # 3: Trending (either uptrend or downtrend)
        # 4: High volatility (trading might be risky)
        df['market_regime'] = 2  # Default: weak mean reversion
        
        # Strong mean reversion
        strong_mr_mask = (df['volatility_regime'] == -1) & (df['trend_regime'] == 0)
        df.loc[strong_mr_mask, 'market_regime'] = 1
        
        # Trending
        trending_mask = (df['trend_regime'] != 0)
        df.loc[trending_mask, 'market_regime'] = 3
        
        # High volatility
        high_vol_mask = (df['volatility_regime'] == 1)
        df.loc[high_vol_mask, 'market_regime'] = 4
        
        # Mean reversion strength indicator
        # Hurst exponent calculation (if possible)
        try:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools.tools import add_constant
            
            def hurst_exponent(time_series, max_lag=20):
                lags = range(2, max_lag)
                tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
                reg = OLS(np.log(tau), add_constant(np.log(lags))).fit()
                return reg.params[1]
            
            # Calculate Hurst exponent for rolling windows
            window_size = 60
            df['hurst_exponent'] = df['spread'].rolling(window=window_size).apply(
                lambda x: hurst_exponent(x.dropna().values) if len(x.dropna()) > 30 else np.nan,
                #raw=True
            )
        except ImportError:
            print("statsmodels not installed. Skipping Hurst exponent calculation.")
        
        # ----------------------
        # 5. Technical indicators
        # ----------------------
        
        # Technical indicators for both stocks
        for ticker in [ticker1, ticker2]:
            if ticker in df.columns:
                # Moving Averages
                df[f'{ticker}_ma_10'] = df[ticker].rolling(window=10).mean()
                df[f'{ticker}_ma_20'] = df[ticker].rolling(window=20).mean()
                df[f'{ticker}_ma_50'] = df[ticker].rolling(window=50).mean()
                
                            # MACD
                df[f'{ticker}_ema_12'] = df[ticker].ewm(span=12, adjust=False).mean()
                df[f'{ticker}_ema_26'] = df[ticker].ewm(span=26, adjust=False).mean()
                df[f'{ticker}_macd'] = df[f'{ticker}_ema_12'] - df[f'{ticker}_ema_26']
                df[f'{ticker}_macd_signal'] = df[f'{ticker}_macd'].ewm(span=9, adjust=False).mean()
                df[f'{ticker}_macd_hist'] = df[f'{ticker}_macd'] - df[f'{ticker}_macd_signal']
        
        # ----------------------
        # 6. Gelişmiş Sinyal / Label Oluşturma
        # ----------------------
        
        # Örnek: Spread’in son 20 günlük Z-score’una bakarak basit bir giriş/çıkış sinyali
        # Sinyalleri kendinizin ihtiyacına göre özelleştirebilirsiniz.
        
        # Z-score tabanlı basit sinyaller:
        df['signal_long'] = 0
        df['signal_short'] = 0
        
        # Z-score < -2 olan yerlerde long sinyali:
        df.loc[df['zscore_20d'] < -2, 'signal_long'] = 1
        
        # Z-score > +2 olan yerlerde short sinyali:
        df.loc[df['zscore_20d'] > 2, 'signal_short'] = 1
        
        # Örnek exit sinyali: Z-score sıfıra yaklaşınca (abs(zscore) < 0.5)
        df['signal_exit'] = 0
        df.loc[df['zscore_20d'].abs() < 0.5, 'signal_exit'] = 1
        
        # ----------------------
        # 7. Veri Temizleme ve Depolama
        # ----------------------
        
        # Tüm hesaplamalardan sonra NaN değerleri temizleyelim
        for days in [1, 3, 5, 10]:
            df[f'spread_fwd_ret_{days}d'] = df['spread'].pct_change(periods=days).shift(-days)

        df.dropna(inplace=True)
        
        # Hazırlanan gelişmiş özellikli veriyi self.ml_features sözlüğünde saklıyoruz
        self.ml_features[pair_key] = df
        
        print(f"Enhanced feature engineering completed for {pair_key}. Data points: {len(df)}")

    
    def plot_strategy_performance(self, df, ticker1, ticker2):
        """
        Plot strategy performance with Equity Curve, Drawdown, and Spread/Z-Score with signals.
        """
        pair_key = f"{ticker1}_{ticker2}"
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

        # Equity Curve
        axes[0].plot(df.index, df['equity'], label="Sermaye Eğrisi", color='blue')
        axes[0].set_title(f"{pair_key} Strateji Performansı")
        axes[0].set_ylabel("Bakiye ($)")
        axes[0].grid(True)
        axes[0].legend()

        # Drawdown
        axes[1].fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        axes[1].set_title("Maksimum Kayıp (Drawdown)")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True)

        # Spread and Z-Score with Trading Signals
        axes[2].plot(df.index, df['zscore_20d'], label="Z-Skor (20 Gün)", color='purple')
        axes[2].axhline(2, color='r', linestyle='--', alpha=0.7, label="Satış Sinyali")
        axes[2].axhline(-2, color='g', linestyle='--', alpha=0.7, label="Alış Sinyali")
        axes[2].scatter(df[df['entry_pred'] == 1].index, df[df['entry_pred'] == 1]['zscore_20d'],
                        marker='^', color='green', s=80, label='Alış (Long)')
        axes[2].scatter(df[df['entry_pred'] == -1].index, df[df['entry_pred'] == -1]['zscore_20d'],
                        marker='v', color='red', s=80, label='Satış (Short)')
        axes[2].set_title("Z-Skor ve Alım-Satım Sinyalleri")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()
        
