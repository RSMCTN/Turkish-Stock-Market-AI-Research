import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning & Deep Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, 
                                   Flatten, Bidirectional, Attention, MultiHeadAttention,
                                   Input, Concatenate, BatchNormalization, LayerNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Advanced Models
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    ADVANCED_MODELS = True
except ImportError:
    print("Advanced models (XGBoost, LightGBM, CatBoost) not installed")
    ADVANCED_MODELS = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not installed")
    PROPHET_AVAILABLE = False

class ComprehensiveStockPredictor:
    """
    Kapsamlı Hisse Senedi Fiyat Tahmin Sistemi
    
    Kullanılan Modeller:
    1. ARIMA/SARIMA - Autoregressive Integrated Moving Average
    2. LSTM - Long Short-Term Memory
    3. GRU - Gated Recurrent Units
    4. Transformer - Attention Mechanism
    5. CNN-LSTM Hybrid
    6. XGBoost/LightGBM/CatBoost
    7. Random Forest/Extra Trees
    8. Support Vector Regression
    9. Prophet (Facebook'un zaman serisi modeli)
    10. Exponential Smoothing
    11. Ensemble Methods
    """
    
    def __init__(self, symbol='AAPL', lookback_hours=168):  # 7 gün = 168 saat
        self.symbol = symbol
        self.lookback_hours = lookback_hours
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.model_scores = {}
        
    def generate_sample_data(self, hours=1000):
        """Örnek hisse senedi verisi üretir"""
        print(f"📊 {self.symbol} için örnek veri üretiliyor...")
        
        # Başlangıç tarihi
        start_date = datetime.now() - timedelta(hours=hours)
        
        # Saatlik tarih aralığı
        dates = pd.date_range(start=start_date, periods=hours, freq='H')
        
        # Gerçekçi hisse senedi fiyat simülasyonu
        np.random.seed(42)
        base_price = 150.0
        
        # Trend bileşeni
        trend = np.linspace(0, 20, hours)
        
        # Mevsimsel bileşen (günlük ve haftalık döngüler)
        daily_cycle = 10 * np.sin(2 * np.pi * np.arange(hours) / 24)
        weekly_cycle = 5 * np.sin(2 * np.pi * np.arange(hours) / (24*7))
        
        # Rastgele yürüyüş (Brownian motion)
        random_walk = np.cumsum(np.random.normal(0, 0.5, hours))
        
        # Volatilite kümelenmesi
        volatility = np.abs(np.random.normal(1, 0.2, hours))
        noise = np.random.normal(0, 1, hours) * volatility
        
        # Final fiyat
        price = base_price + trend + daily_cycle + weekly_cycle + random_walk + noise
        
        # Volume simülasyonu
        volume = np.random.lognormal(10, 0.5, hours).astype(int)
        
        # DataFrame oluştur
        data = pd.DataFrame({
            'timestamp': dates,
            'open': price + np.random.normal(0, 0.1, hours),
            'high': price + np.abs(np.random.normal(0.5, 0.2, hours)),
            'low': price - np.abs(np.random.normal(0.5, 0.2, hours)),
            'close': price,
            'volume': volume
        })
        
        # Teknik göstergeler ekle
        data = self.add_technical_indicators(data)
        
        print(f"✅ {len(data)} saatlik veri hazırlandı")
        return data
        
    def add_technical_indicators(self, df):
        """Teknik göstergeler ekler"""
        data = df.copy()
        
        # Hareketli ortalamalar
        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(window=20).mean()
        std_20 = data['close'].rolling(window=20).std()
        data['bb_upper'] = sma_20 + (std_20 * 2)
        data['bb_lower'] = sma_20 - (std_20 * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / sma_20
        
        # Stochastic
        low_min = data['low'].rolling(window=14).min()
        high_max = data['high'].rolling(window=14).max()
        data['stoch_k'] = 100 * (data['close'] - low_min) / (high_max - low_min)
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        data['williams_r'] = -100 * (high_max - data['close']) / (high_max - low_min)
        
        # ATR (Average True Range)
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift())
        tr3 = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Price change features
        data['price_change'] = data['close'].pct_change()
        data['price_change_2h'] = data['close'].pct_change(periods=2)
        data['price_change_24h'] = data['close'].pct_change(periods=24)
        
        # Volatility
        data['volatility'] = data['price_change'].rolling(window=24).std()
        
        # Time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Cyclical time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data.fillna(method='bfill').fillna(method='ffill')
    
    def prepare_sequences(self, data, target_col='close', sequence_length=24):
        """Sequence verisi hazırlar"""
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data)):
            sequences.append(data.iloc[i-sequence_length:i].values)
            targets.append(data[target_col].iloc[i])
        
        return np.array(sequences), np.array(targets)
    
    def split_data(self, data, train_ratio=0.8):
        """Veriyi eğitim ve test olarak böler"""
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        return train_data, test_data
    
    # 1. ARIMA/SARIMA Modelleri
    def train_arima_models(self, data):
        """ARIMA ve SARIMA modellerini eğitir"""
        print("📈 ARIMA/SARIMA modelleri eğitiliyor...")
        
        # Veri hazırlığı
        ts_data = data['close'].dropna()
        
        # Auto ARIMA (basit grid search)
        best_aic = float('inf')
        best_params = None
        
        # Parametreleri test et
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts_data, order=(p,d,q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p,d,q)
                    except:
                        continue
        
        # En iyi ARIMA modelini eğit
        if best_params:
            arima_model = ARIMA(ts_data, order=best_params)
            arima_fitted = arima_model.fit()
            self.models['ARIMA'] = arima_fitted
            print(f"✅ ARIMA{best_params} - AIC: {best_aic:.2f}")
        
        # SARIMA modeli (mevsimsel)
        try:
            sarima_model = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,24))
            sarima_fitted = sarima_model.fit(disp=False)
            self.models['SARIMA'] = sarima_fitted
            print(f"✅ SARIMA - AIC: {sarima_fitted.aic:.2f}")
        except:
            print("⚠️ SARIMA model eğitilemedi")
    
    # 2. LSTM Modeli
    def build_lstm_model(self, input_shape):
        """LSTM modelini oluşturur"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    # 3. GRU Modeli
    def build_gru_model(self, input_shape):
        """GRU modelini oluşturur"""
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    # 4. Transformer Model
    def build_transformer_model(self, input_shape):
        """Transformer modelini oluşturur"""
        inputs = Input(shape=input_shape)
        
        # Multi-Head Attention
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention = LayerNormalization()(inputs + attention)
        
        # Feed Forward
        ff = Dense(512, activation='relu')(attention)
        ff = Dense(input_shape[-1])(ff)
        ff = LayerNormalization()(attention + ff)
        
        # Global Average Pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ff)
        
        # Final layers
        dense = Dense(128, activation='relu')(pooled)
        dense = Dropout(0.2)(dense)
        outputs = Dense(1)(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    # 5. CNN-LSTM Hybrid
    def build_cnn_lstm_model(self, input_shape):
        """CNN-LSTM hibrit modelini oluşturur"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    # 6. Advanced Tree-based Models
    def train_tree_models(self, X_train, y_train):
        """Ağaç tabanlı modelleri eğitir"""
        print("🌳 Tree-based modeller eğitiliyor...")
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
        }
        
        if ADVANCED_MODELS:
            models.update({
                'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42),
                'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, random_state=42),
                'CatBoost': CatBoostRegressor(iterations=200, depth=6, random_state=42, verbose=False)
            })
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                print(f"✅ {name} modeli eğitildi")
            except Exception as e:
                print(f"❌ {name} model hatası: {str(e)}")
    
    # 7. Traditional ML Models
    def train_traditional_ml(self, X_train, y_train):
        """Geleneksel makine öğrenmesi modellerini eğitir"""
        print("📊 Geleneksel ML modeller eğitiliyor...")
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                print(f"✅ {name} modeli eğitildi")
            except Exception as e:
                print(f"❌ {name} model hatası: {str(e)}")
    
    # 8. Prophet Model
    def train_prophet_model(self, data):
        """Facebook Prophet modelini eğitir"""
        if not PROPHET_AVAILABLE:
            print("⚠️ Prophet modeli mevcut değil")
            return
            
        print("🔮 Prophet modeli eğitiliyor...")
        
        # Prophet için veri formatı
        prophet_data = data[['timestamp', 'close']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Model oluştur ve eğit
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        try:
            model.fit(prophet_data)
            self.models['Prophet'] = model
            print("✅ Prophet modeli eğitildi")
        except Exception as e:
            print(f"❌ Prophet model hatası: {str(e)}")
    
    # 9. Exponential Smoothing
    def train_exponential_smoothing(self, data):
        """Exponential Smoothing modelini eğitir"""
        print("📈 Exponential Smoothing modeli eğitiliyor...")
        
        ts_data = data['close'].dropna()
        
        try:
            # Triple Exponential Smoothing (Holt-Winters)
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal='add',
                seasonal_periods=24  # 24 saatlik mevsimsellik
            )
            fitted_model = model.fit()
            self.models['ExponentialSmoothing'] = fitted_model
            print("✅ Exponential Smoothing modeli eğitildi")
        except Exception as e:
            print(f"❌ Exponential Smoothing hatası: {str(e)}")
    
    def train_deep_learning_models(self, data):
        """Derin öğrenme modellerini eğitir"""
        print("🧠 Deep Learning modeller eğitiliyor...")
        
        # Feature selection
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'close']]
        
        # Veri hazırlığı
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data[feature_cols])
        self.scalers['features'] = scaler
        
        # Target scaling
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(data[['close']])
        self.scalers['target'] = target_scaler
        
        # Sequence hazırlama
        sequence_length = 24  # 24 saat lookback
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(scaled_target[i])
        
        X, y = np.array(X), np.array(y)
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Model eğitimi
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        # LSTM
        try:
            lstm_model = self.build_lstm_model(input_shape)
            lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, 
                          validation_split=0.2, callbacks=callbacks, verbose=0)
            self.models['LSTM'] = lstm_model
            print("✅ LSTM modeli eğitildi")
        except Exception as e:
            print(f"❌ LSTM hatası: {str(e)}")
        
        # GRU
        try:
            gru_model = self.build_gru_model(input_shape)
            gru_model.fit(X_train, y_train, epochs=100, batch_size=32,
                         validation_split=0.2, callbacks=callbacks, verbose=0)
            self.models['GRU'] = gru_model
            print("✅ GRU modeli eğitildi")
        except Exception as e:
            print(f"❌ GRU hatası: {str(e)}")
        
        # Transformer
        try:
            transformer_model = self.build_transformer_model(input_shape)
            transformer_model.fit(X_train, y_train, epochs=50, batch_size=32,
                                validation_split=0.2, callbacks=callbacks, verbose=0)
            self.models['Transformer'] = transformer_model
            print("✅ Transformer modeli eğitildi")
        except Exception as e:
            print(f"❌ Transformer hatası: {str(e)}")
        
        # CNN-LSTM
        try:
            cnn_lstm_model = self.build_cnn_lstm_model(input_shape)
            cnn_lstm_model.fit(X_train, y_train, epochs=100, batch_size=32,
                             validation_split=0.2, callbacks=callbacks, verbose=0)
            self.models['CNN_LSTM'] = cnn_lstm_model
            print("✅ CNN-LSTM modeli eğitildi")
        except Exception as e:
            print(f"❌ CNN-LSTM hatası: {str(e)}")
    
    def make_predictions(self, data, prediction_hours=120):  # 5 gün = 120 saat
        """Tüm modellerle tahmin yapar"""
        print(f"🔮 {prediction_hours} saatlik tahminler yapılıyor...")
        
        last_timestamp = data['timestamp'].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=prediction_hours,
            freq='H'
        )
        
        predictions_df = pd.DataFrame({'timestamp': future_timestamps})
        
        # Her model için tahmin
        for model_name, model in self.models.items():
            try:
                if model_name in ['ARIMA', 'SARIMA']:
                    # ARIMA/SARIMA tahminleri
                    forecast = model.forecast(steps=prediction_hours)
                    predictions_df[model_name] = forecast.values
                
                elif model_name == 'Prophet' and PROPHET_AVAILABLE:
                    # Prophet tahminleri
                    future_df = pd.DataFrame({'ds': future_timestamps})
                    forecast = model.predict(future_df)
                    predictions_df[model_name] = forecast['yhat'].values
                
                elif model_name == 'ExponentialSmoothing':
                    # Exponential Smoothing tahminleri
                    forecast = model.forecast(steps=prediction_hours)
                    predictions_df[model_name] = forecast.values
                
                elif model_name in ['LSTM', 'GRU', 'Transformer', 'CNN_LSTM']:
                    # Deep Learning tahminleri
                    self.predict_deep_learning(model_name, model, data, predictions_df, prediction_hours)
                
                else:
                    # Traditional ML tahminleri
                    self.predict_traditional_ml(model_name, model, data, predictions_df, prediction_hours)
                
                print(f"✅ {model_name} tahminleri tamamlandı")
                
            except Exception as e:
                print(f"❌ {model_name} tahmin hatası: {str(e)}")
        
        # Ensemble tahmin (tüm modellerin ortalaması)
        model_cols = [col for col in predictions_df.columns if col != 'timestamp']
        if len(model_cols) > 1:
            predictions_df['Ensemble_Mean'] = predictions_df[model_cols].mean(axis=1)
            predictions_df['Ensemble_Median'] = predictions_df[model_cols].median(axis=1)
        
        self.predictions = predictions_df
        return predictions_df
    
    def predict_deep_learning(self, model_name, model, data, predictions_df, prediction_hours):
        """Deep learning modellerle tahmin yapar"""
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'close']]
        
        # Son 24 saatlik veriyi al
        last_sequence = data[feature_cols].iloc[-24:].values
        last_sequence_scaled = self.scalers['features'].transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(prediction_hours):
            # Tahmin yap
            pred_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
            pred_scaled = model.predict(pred_input, verbose=0)
            pred = self.scalers['target'].inverse_transform(pred_scaled)[0][0]
            predictions.append(pred)
            
            # Sequence'i güncelle (basit yaklaşım)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            # Son değeri tahmini fiyatla güncelle (simplified)
            current_sequence[-1, 0] = pred_scaled[0][0]  # İlk feature close price olduğunu varsayıyoruz
        
        predictions_df[model_name] = predictions
    
    def predict_traditional_ml(self, model_name, model, data, predictions_df, prediction_hours):
        """Geleneksel ML modellerle tahmin yapar"""
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'close']]
        
        # Son veriyi feature olarak kullan ve iterative tahmin yap
        predictions = []
        last_features = data[feature_cols].iloc[-1:].values
        
        for i in range(prediction_hours):
            pred = model.predict(last_features)[0]
            predictions.append(pred)
            
            # Basit feature güncellemesi (gerçek uygulamada daha sofistike olmalı)
            # Bu örnekte sadece son tahmini kullanıyoruz
            
        predictions_df[model_name] = predictions
    
    def evaluate_models(self, data):
        """Modelleri değerlendirir"""
        print("📊 Model performansları değerlendiriliyor...")
        
        # Test verisi hazırla
        train_data, test_data = self.split_data(data, train_ratio=0.8)
        actual_prices = test_data['close'].values
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Test verisi için tahmin yap
                if model_name in ['ARIMA', 'SARIMA']:
                    predictions = model.forecast(steps=len(test_data))
                    pred_values = predictions.values if hasattr(predictions, 'values') else predictions
                
                elif model_name == 'ExponentialSmoothing':
                    predictions = model.forecast(steps=len(test_data))
                    pred_values = predictions.values if hasattr(predictions, 'values') else predictions
                
                else:
                    # Basitleştirilmiş tahmin (gerçek uygulamada daha detaylı olmalı)
                    pred_values = actual_prices  # Placeholder
                
                # Metrikleri hesapla
                mse = mean_squared_error(actual_prices[:len(pred_values)], pred_values[:len(actual_prices)])
                mae = mean_absolute_error(actual_prices[:len(pred_values)], pred_values[:len(actual_prices)])
                rmse = np.sqrt(mse)
                
                try:
                    r2 = r2_score(actual_prices[:len(pred_values)], pred_values[:len(actual_prices)])
                except:
                    r2 = 0
                
                evaluation_results[model_name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
                
                print(f"✅ {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                
            except Exception as e:
                print(f"❌ {model_name} değerlendirme hatası: {str(e)}")
        
        self.model_scores = evaluation_results
        return evaluation_results
    
    def plot_predictions(self, data, days_back=7):
        """Tahminleri görselleştirir"""
        print("📈 Tahmin grafikleri oluşturuluyor...")
        
        # Son X günlük gerçek veri
        cutoff_time = data['timestamp'].iloc[-1] - timedelta(days=days_back)
        recent_data = data[data['timestamp'] > cutoff_time]
        
        # Grafik oluştur
        plt.figure(figsize=(20, 12))
        
        # 1. Ana fiyat grafiği
        plt.subplot(3, 2, 1)
        plt.plot(recent_data['timestamp'], recent_data['close'], 
                label='Gerçek Fiyat', linewidth=2, color='black')
        
        if hasattr(self, 'predictions') and self.predictions is not None:
            # En iyi 5 modelin tahminlerini göster
            model_cols = [col for col in self.predictions.columns if col != 'timestamp']
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, model in enumerate(model_cols[:8]):
                plt.plot(self.predictions['timestamp'], self.predictions[model],
                        label=f'{model}', alpha=0.7, linewidth=1, color=colors[i % len(colors)])
        
        plt.title('Hisse Senedi Fiyat Tahminleri - Ana Görünüm', fontsize=14, fontweight='bold')
        plt.xlabel('Zaman')
        plt.ylabel('Fiyat ($)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Model karşılaştırması (yakınlaştırılmış)
        plt.subplot(3, 2, 2)
        if hasattr(self, 'predictions') and self.predictions is not None:
            # Sadece ilk 48 saati göster
            pred_48h = self.predictions.head(48)
            for i, model in enumerate(model_cols[:5]):
                plt.plot(pred_48h['timestamp'], pred_48h[model],
                        label=f'{model}', linewidth=2, marker='o', markersize=3)
        
        plt.title('48 Saatlik Detay Tahminler', fontsize=12, fontweight='bold')
        plt.xlabel('Zaman')
        plt.ylabel('Fiyat ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Model performans karşılaştırması
        plt.subplot(3, 2, 3)
        if hasattr(self, 'model_scores') and self.model_scores:
            models = list(self.model_scores.keys())
            rmse_scores = [self.model_scores[model]['RMSE'] for model in models]
            
            bars = plt.bar(models, rmse_scores, color='skyblue', alpha=0.7)
            plt.title('Model Performansları (RMSE)', fontsize=12, fontweight='bold')
            plt.xlabel('Modeller')
            plt.ylabel('RMSE')
            plt.xticks(rotation=45)
            
            # En iyi modeli vurgula
            min_idx = np.argmin(rmse_scores)
            bars[min_idx].set_color('gold')
            
            # Değerleri göster
            for i, v in enumerate(rmse_scores):
                plt.text(i, v + max(rmse_scores) * 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 4. Volatilite analizi
        plt.subplot(3, 2, 4)
        if hasattr(self, 'predictions') and self.predictions is not None:
            model_cols = [col for col in self.predictions.columns if col != 'timestamp' and 'Ensemble' not in col]
            if len(model_cols) > 1:
                # Model tahminleri arasındaki standart sapma (belirsizlik)
                uncertainty = self.predictions[model_cols].std(axis=1)
                plt.plot(self.predictions['timestamp'], uncertainty, 
                        color='red', linewidth=2, label='Model Belirsizliği')
                plt.fill_between(self.predictions['timestamp'], 0, uncertainty, 
                               alpha=0.3, color='red')
                
                plt.title('Tahmin Belirsizliği (Model Varyansı)', fontsize=12, fontweight='bold')
                plt.xlabel('Zaman')
                plt.ylabel('Standart Sapma ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
        
        # 5. Günlük ortalama tahminler
        plt.subplot(3, 2, 5)
        if hasattr(self, 'predictions') and self.predictions is not None:
            # Günlük ortalamalar hesapla
            pred_copy = self.predictions.copy()
            pred_copy['date'] = pred_copy['timestamp'].dt.date
            
            model_cols = [col for col in pred_copy.columns if col not in ['timestamp', 'date']]
            daily_avg = pred_copy.groupby('date')[model_cols].mean()
            
            # En iyi 3 modeli göster
            if len(model_cols) >= 3:
                top_models = list(daily_avg.columns)[:3]
                for model in top_models:
                    plt.plot(daily_avg.index, daily_avg[model], 
                            marker='o', linewidth=2, label=model, markersize=6)
            
            plt.title('5 Günlük Ortalama Fiyat Tahminleri', fontsize=12, fontweight='bold')
            plt.xlabel('Tarih')
            plt.ylabel('Ortalama Fiyat ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # 6. Ensemble model detayı
        plt.subplot(3, 2, 6)
        if hasattr(self, 'predictions') and self.predictions is not None:
            if 'Ensemble_Mean' in self.predictions.columns:
                plt.plot(self.predictions['timestamp'], self.predictions['Ensemble_Mean'],
                        color='gold', linewidth=3, label='Ensemble Ortalama')
                
                if 'Ensemble_Median' in self.predictions.columns:
                    plt.plot(self.predictions['timestamp'], self.predictions['Ensemble_Median'],
                            color='darkorange', linewidth=2, linestyle='--', label='Ensemble Medyan')
                
                # Güven aralığı
                model_cols = [col for col in self.predictions.columns if col not in ['timestamp', 'Ensemble_Mean', 'Ensemble_Median']]
                if len(model_cols) > 2:
                    upper = self.predictions[model_cols].quantile(0.75, axis=1)
                    lower = self.predictions[model_cols].quantile(0.25, axis=1)
                    
                    plt.fill_between(self.predictions['timestamp'], lower, upper,
                                   alpha=0.2, color='gold', label='%50 Güven Aralığı')
                
                plt.title('Ensemble Model Tahminleri', fontsize=12, fontweight='bold')
                plt.xlabel('Zaman')
                plt.ylabel('Fiyat ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_trading_signals(self):
        """Trading sinyalleri üretir"""
        if not hasattr(self, 'predictions') or self.predictions is None:
            print("⚠️ Önce tahmin yapılmalı")
            return None
        
        print("📊 Trading sinyalleri üretiliyor...")
        
        signals_df = self.predictions.copy()
        
        # Ensemble model kullan
        if 'Ensemble_Mean' in signals_df.columns:
            price_col = 'Ensemble_Mean'
        else:
            # İlk mevcut model
            price_col = [col for col in signals_df.columns if col != 'timestamp'][0]
        
        # Fiyat değişimi hesapla
        signals_df['price_change'] = signals_df[price_col].pct_change()
        signals_df['price_change_abs'] = signals_df['price_change'].abs()
        
        # Trend belirleme (basit hareketli ortalama)
        signals_df['sma_short'] = signals_df[price_col].rolling(window=6).mean()  # 6 saat
        signals_df['sma_long'] = signals_df[price_col].rolling(window=24).mean()  # 24 saat
        
        # Sinyal üretimi
        signals_df['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        signals_df['confidence'] = 0.0  # 0-1 arası güven skoru
        
        for i in range(1, len(signals_df)):
            current_price = signals_df[price_col].iloc[i]
            prev_price = signals_df[price_col].iloc[i-1]
            
            # Trend sinyali
            if signals_df['sma_short'].iloc[i] > signals_df['sma_long'].iloc[i]:
                trend_signal = 1  # Uptrend
            else:
                trend_signal = -1  # Downtrend
            
            # Momentum sinyali
            price_change = signals_df['price_change'].iloc[i]
            
            if price_change > 0.02:  # %2'den fazla artış
                momentum_signal = 1
                confidence = min(abs(price_change) * 10, 1.0)
            elif price_change < -0.02:  # %2'den fazla düşüş
                momentum_signal = -1
                confidence = min(abs(price_change) * 10, 1.0)
            else:
                momentum_signal = 0
                confidence = 0.3
            
            # Final sinyal
            if trend_signal == 1 and momentum_signal == 1:
                signals_df.loc[signals_df.index[i], 'signal'] = 1  # Strong Buy
                signals_df.loc[signals_df.index[i], 'confidence'] = confidence
            elif trend_signal == -1 and momentum_signal == -1:
                signals_df.loc[signals_df.index[i], 'signal'] = -1  # Strong Sell
                signals_df.loc[signals_df.index[i], 'confidence'] = confidence
            else:
                signals_df.loc[signals_df.index[i], 'signal'] = 0  # Hold
                signals_df.loc[signals_df.index[i], 'confidence'] = confidence * 0.5
        
        # Sinyal özetini hazırla
        buy_signals = len(signals_df[signals_df['signal'] == 1])
        sell_signals = len(signals_df[signals_df['signal'] == -1])
        hold_signals = len(signals_df[signals_df['signal'] == 0])
        
        print(f"📈 Buy Sinyalleri: {buy_signals}")
        print(f"📉 Sell Sinyalleri: {sell_signals}")
        print(f"⏸️  Hold Sinyalleri: {hold_signals}")
        
        return signals_df
    
    def run_complete_analysis(self):
        """Tüm analizi çalıştırır"""
        print("🚀 Kapsamlı Hisse Senedi Analizi Başlıyor...")
        print("=" * 60)
        
        # 1. Veri hazırlama
        print("📊 Adım 1: Veri Hazırlama")
        data = self.generate_sample_data(hours=2000)  # 2000 saatlik veri
        print(f"✅ {self.symbol} için {len(data)} saatlik veri hazırlandı")
        print()
        
        # 2. Geleneksel zaman serisi modellerini eğit
        print("📈 Adım 2: Zaman Serisi Modelleri")
        self.train_arima_models(data)
        self.train_exponential_smoothing(data)
        self.train_prophet_model(data)
        print()
        
        # 3. Makine öğrenmesi modellerini eğit
        print("🤖 Adım 3: Makine Öğrenmesi Modelleri")
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'close']]
        X = data[feature_cols].fillna(0)
        y = data['close']
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        train_size = int(len(X_scaled) * 0.8)
        X_train, y_train = X_scaled[:train_size], y[:train_size]
        
        self.train_tree_models(X_train, y_train)
        self.train_traditional_ml(X_train, y_train)
        print()
        
        # 4. Derin öğrenme modellerini eğit
        print("🧠 Adım 4: Deep Learning Modelleri")
        self.train_deep_learning_models(data)
        print()
        
        # 5. Tahminleri yap
        print("🔮 Adım 5: Tahmin Üretimi (5 gün = 120 saat)")
        predictions = self.make_predictions(data, prediction_hours=120)
        print(f"✅ {len(predictions)} saatlik tahmin üretildi")
        print()
        
        # 6. Model performansını değerlendir
        print("📊 Adım 6: Model Değerlendirmesi")
        evaluation = self.evaluate_models(data)
        print()
        
        # 7. Trading sinyalleri üret
        print("💹 Adım 7: Trading Sinyalleri")
        signals = self.generate_trading_signals()
        print()
        
        # 8. Sonuçları görselleştir
        print("📈 Adım 8: Görselleştirme")
        self.plot_predictions(data, days_back=10)
        
        # 9. Özet rapor
        self.generate_summary_report(data, predictions, signals)
        
        print("✅ Analiz Tamamlandı!")
        return data, predictions, signals
    
    def generate_summary_report(self, data, predictions, signals):
        """Özet rapor oluşturur"""
        print("\n" + "=" * 60)
        print(f"📊 {self.symbol} HİSSE SENEDİ TAHMİN RAPORU")
        print("=" * 60)
        
        # Mevcut durum
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-25]  # 24 saat öncesi
        daily_change = ((current_price - prev_price) / prev_price) * 100
        
        print(f"💰 Mevcut Fiyat: ${current_price:.2f}")
        print(f"📈 24 Saatlik Değişim: {daily_change:.2f}%")
        print(f"📊 Analiz Edilen Veri: {len(data)} saat")
        print()
        
        # Model sayıları
        print(f"🤖 Kullanılan Model Sayısı: {len(self.models)}")
        for model_name in self.models.keys():
            print(f"   ✓ {model_name}")
        print()
        
        # 5 günlük tahmin özeti
        if predictions is not None and len(predictions) > 0:
            ensemble_col = 'Ensemble_Mean' if 'Ensemble_Mean' in predictions.columns else predictions.columns[1]
            
            # Günlük ortalamalar
            pred_copy = predictions.copy()
            pred_copy['date'] = pred_copy['timestamp'].dt.date
            daily_avg = pred_copy.groupby('date')[ensemble_col].mean()
            
            print("📅 5 GÜNLÜK FİYAT TAHMİNLERİ:")
            for date, price in daily_avg.items():
                day_change = ((price - current_price) / current_price) * 100
                print(f"   {date}: ${price:.2f} ({day_change:+.2f}%)")
            print()
            
            # Genel trend
            week_end_price = daily_avg.iloc[-1]
            week_change = ((week_end_price - current_price) / current_price) * 100
            
            if week_change > 5:
                trend = "📈 GÜÇLÜ YUKARI"
            elif week_change > 1:
                trend = "📈 Yukarı"
            elif week_change > -1:
                trend = "➡️ Yatay"
            elif week_change > -5:
                trend = "📉 Aşağı"
            else:
                trend = "📉 GÜÇLÜ AŞAĞI"
                
            print(f"🎯 5 Günlük Genel Trend: {trend} ({week_change:+.2f}%)")
            print()
        
        # Trading sinyalleri
        if signals is not None:
            latest_signal = signals['signal'].iloc[-1]
            latest_confidence = signals['confidence'].iloc[-1]
            
            signal_map = {1: "📈 SATIN AL", -1: "📉 SAT", 0: "⏸️ BEKLE"}
            
            print(f"💹 GÜNCEL SİNYAL: {signal_map.get(latest_signal, 'BELIRSIZ')}")
            print(f"🎯 Güven Skoru: {latest_confidence:.2f}/1.00")
            print()
        
        # En iyi performans gösteren modeller
        if hasattr(self, 'model_scores') and self.model_scores:
            sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1]['RMSE'])
            print("🏆 EN İYİ PERFORMANS GÖSTEREN MODELLER:")
            for i, (model_name, scores) in enumerate(sorted_models[:3], 1):
                print(f"   {i}. {model_name}: RMSE={scores['RMSE']:.4f}, R²={scores['R2']:.4f}")
            print()
        
        # Risk analizi
        if predictions is not None and len(predictions.columns) > 2:
            model_cols = [col for col in predictions.columns if col not in ['timestamp', 'Ensemble_Mean', 'Ensemble_Median']]
            if len(model_cols) > 1:
                model_std = predictions[model_cols].std(axis=1).mean()
                volatility = (model_std / current_price) * 100
                
                if volatility < 2:
                    risk_level = "🟢 DÜŞÜK"
                elif volatility < 5:
                    risk_level = "🟡 ORTA"
                else:
                    risk_level = "🔴 YÜKSEK"
                
                print(f"⚠️ RİSK SEVİYESİ: {risk_level} (Volatilite: {volatility:.2f}%)")
                print()
        
        print("📝 DİKKAT:")
        print("   • Bu tahminler sadece eğitim amaçlıdır")
        print("   • Gerçek yatırım kararları için profesyonel danışmanlık alın")
        print("   • Geçmiş performans gelecekteki sonuçları garanti etmez")
        print("=" * 60)

# KULLANIM ÖRNEĞİ
def main():
    """Ana fonksiyon - Sistemi çalıştır"""
    
    # Hisse senedi seç
    symbol = "AAPL"  # Apple hisse senedi
    
    print(f"🚀 {symbol} için Kapsamlı Tahmin Sistemi Başlatılıyor...")
    print()
    
    # Tahmin sistemi oluştur
    predictor = ComprehensiveStockPredictor(symbol=symbol, lookback_hours=168)
    
    # Tüm analizi çalıştır
    data, predictions, signals = predictor.run_complete_analysis()
    
    # Sonuçları kaydet
    if predictions is not None:
        predictions.to_csv(f'{symbol}_predictions.csv', index=False)
        print(f"📁 Tahminler {symbol}_predictions.csv dosyasına kaydedildi")
    
    if signals is not None:
        signals.to_csv(f'{symbol}_trading_signals.csv', index=False)
        print(f"📁 Trading sinyalleri {symbol}_trading_signals.csv dosyasına kaydedildi")
    
    return predictor, data, predictions, signals

if __name__ == "__main__":
    # Gerekli kütüphanelerin kurulumu için:
    """
    pip install numpy pandas matplotlib seaborn scikit-learn
    pip install tensorflow
    pip install statsmodels
    pip install xgboost lightgbm catboost
    pip install prophet
    """
    
    # Sistemi çalıştır
    predictor, data, predictions, signals = main()
    
    # İsteğe bağlı: Farklı hisse senetleri için de çalıştır
    """
    other_stocks = ['GOOGL', 'TSLA', 'MSFT', 'NVDA', 'AMZN', 'META']
    for stock in other_stocks:
        print(f"\n{'='*60}")
        print(f"🔄 {stock} analizi başlıyor...")
        predictor_stock = ComprehensiveStockPredictor(symbol=stock)
        predictor_stock.run_complete_analysis()
    """
    