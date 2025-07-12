import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockPredictor:
    def __init__(self, symbol='AAPL', period='2y'):
        """
        Initialize the Stock Predictor
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
            period (str): Time period for data ('1y', '2y', '5y', '10y', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            print(f"✓ Successfully fetched {len(self.data)} days of data for {self.symbol}")
            return True
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return False
    
    def clean_data(self):
        """Comprehensive data cleaning and preprocessing"""
        print("🧹 Starting data cleaning process...")
        df = self.data.copy()
        
        #Displaying initial data 
        print(f"Initial data shape: {df.shape}")
        print(f"Initial missing values:\n{df.isnull().sum()}")
        
        #Handling missing values
        print("\n1. Handling missing values...")
        
        #Check for any missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"Found missing values: {missing_values[missing_values > 0]}")
            
            # Forward fill for price data (carry forward last known price)
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')
            
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(df['Volume'].median())
        
        #Removing outliers
        print("\n2. Detecting and handling outliers...")
        
        #Detecting outliers using IQR method for price columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    print(f"Found {len(outliers)} outliers in {col}")
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        #Validation of Data
        print("\n3. Validating data consistency...")
        
        #Checking for logical inconsistencies
        invalid_rows = df[df['High'] < df['Low']]
        if len(invalid_rows) > 0:
            print(f"Found {len(invalid_rows)} rows where High < Low - fixing...")
            # Swap High and Low values
            df.loc[invalid_rows.index, ['High', 'Low']] = df.loc[invalid_rows.index, ['Low', 'High']].values
        
        #Checking for zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                zero_prices = df[df[col] <= 0]
                if len(zero_prices) > 0:
                    print(f"Found {len(zero_prices)} zero/negative prices in {col} - removing...")
                    df = df[df[col] > 0]
        
        #Checking for zero volume (might be valid for some stocks)
        if 'Volume' in df.columns:
            zero_volume = df[df['Volume'] == 0]
            if len(zero_volume) > 0:
                print(f"Found {len(zero_volume)} zero volume days - keeping as valid")
        
        #Removing duplicates
        print("\n4. Checking for duplicate dates...")
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} duplicate entries")
        
        #Sorting by date
        df = df.sort_index()
        
        #Data quality summary
        print(f"\n✅ Data cleaning completed!")
        print(f"Final data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Final missing values: {df.isnull().sum().sum()}")
        
        self.data = df
        return True

    def create_features(self):
        """Create technical indicators and features"""
        df = self.data.copy()
        
        #Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Price_Range'] = df['High'] - df['Low']
        
        #Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        #Technical indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['Bollinger_Upper'], df['Bollinger_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        #Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        #Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        #Handle NaN values created by feature engineering
        print(f"NaN values after feature creation: {df.isnull().sum().sum()}")
        
        #Filling NaN values for moving averages with method='bfill' then 'ffill'
        ma_columns = ['MA_5', 'MA_10', 'MA_20', 'MA_50']
        for col in ma_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        #Filling NaN values for technical indicators
        tech_columns = ['RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
        for col in tech_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        # For lag features, we need to drop the first few rows
        # as they will have NaN values by design
        print(f"Rows before dropping NaN: {len(df)}")
        df = df.dropna()
        print(f"Rows after dropping NaN: {len(df)}")
        
        #Final validation - ensure no infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        #Selectinf features for modeling
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'Price_Change', 'High_Low_Ratio', 'Price_Range',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower',
            'Volume_MA', 'Volume_Ratio',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_5'
        ]
        
        self.features = df[feature_columns]
        self.target = df['Close']
        
        print(f"✓ Created {len(feature_columns)} features")
        print(f"✓ Dataset shape: {self.features.shape}")
        
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def prepare_data(self, test_size=0.2):
        """Split and scale the data"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store data
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.train_dates = y_train.index
        self.test_dates = y_test.index
        self.scalers['features'] = scaler
        
        print(f"✓ Data split - Training: {len(X_train)}, Testing: {len(X_test)}")
        
    def train_linear_regression(self):
        """Train Linear Regression model"""
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        
        # Predictions
        train_pred = lr.predict(self.X_train)
        test_pred = lr.predict(self.X_test)
        
        self.models['Linear Regression'] = lr
        self.predictions['Linear Regression'] = {
            'train': train_pred,
            'test': test_pred,
            'train_actual': self.y_train,
            'test_actual': self.y_test
        }
        
        print("✓ Linear Regression trained")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        
        # Predictions
        train_pred = rf.predict(self.X_train)
        test_pred = rf.predict(self.X_test)
        
        self.models['Random Forest'] = rf
        self.predictions['Random Forest'] = {
            'train': train_pred,
            'test': test_pred,
            'train_actual': self.y_train,
            'test_actual': self.y_test
        }
        
        print("✓ Random Forest trained")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        results = {}
        
        for model_name, preds in self.predictions.items():
            # Calculate metrics
            train_mse = mean_squared_error(preds['train_actual'], preds['train'])
            test_mse = mean_squared_error(preds['test_actual'], preds['test'])
            train_mae = mean_absolute_error(preds['train_actual'], preds['train'])
            test_mae = mean_absolute_error(preds['test_actual'], preds['test'])
            train_r2 = r2_score(preds['train_actual'], preds['train'])
            test_r2 = r2_score(preds['test_actual'], preds['test'])
            
            results[model_name] = {
                'Train MSE': train_mse,
                'Test MSE': test_mse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R²': train_r2,
                'Test R²': test_r2
            }
        
        # Create results dataframe
        results_df = pd.DataFrame(results).T
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(results_df.round(4))
        
        return results_df
    
    def plot_predictions(self):
        """Plot actual vs predicted prices with enhanced visualization"""
        print("\n📊 Generating Actual vs Predicted Price Comparison Graphs...")
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(n_models, 1, figsize=(16, 8*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, preds) in enumerate(self.predictions.items()):
            ax = axes[i]
            
            # Plot training data (lighter colors)
            ax.plot(self.train_dates, preds['train_actual'], 
                   label='Actual (Training)', color='lightblue', alpha=0.6, linewidth=1)
            ax.plot(self.train_dates, preds['train'], 
                   label='Predicted (Training)', color='lightcoral', alpha=0.6, linewidth=1)
            
            # Plot test data (bold colors) - THIS IS THE MAIN COMPARISON
            ax.plot(self.test_dates, preds['test_actual'], 
                   label='Actual (Test)', color='darkblue', linewidth=3, marker='o', markersize=2)
            ax.plot(self.test_dates, preds['test'], 
                   label='Predicted (Test)', color='darkred', linewidth=3, marker='s', markersize=2)
            
            # Add vertical line to separate train/test
            if len(self.train_dates) > 0 and len(self.test_dates) > 0:
                ax.axvline(x=self.test_dates[0], color='green', linestyle='--', alpha=0.7, 
                          label='Train/Test Split')
            
            # Calculate test accuracy metrics for display
            test_mae = mean_absolute_error(preds['test_actual'], preds['test'])
            test_mse = mean_squared_error(preds['test_actual'], preds['test'])
            test_r2 = r2_score(preds['test_actual'], preds['test'])
            
            ax.set_title(f'{model_name} - {self.symbol} Stock Price Prediction\n'
                        f'Test MAE: ${test_mae:.2f} | Test MSE: ${test_mse:.2f} | Test R²: {test_r2:.3f}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add annotation for test period
            ax.text(0.02, 0.98, f'Test Period: {len(self.test_dates)} days', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Additional focused plot for test data only
        self.plot_test_data_only()
    
    def plot_test_data_only(self):
        """Create a focused plot showing only test data comparison"""
        print("\n🎯 Generating Test Data Only Comparison...")
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(n_models, 1, figsize=(14, 6*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, preds) in enumerate(self.predictions.items()):
            ax = axes[i]
            
            # Plot only test data for clearer comparison
            ax.plot(self.test_dates, preds['test_actual'], 
                   label='Actual Price', color='darkblue', linewidth=3, 
                   marker='o', markersize=4, alpha=0.8)
            ax.plot(self.test_dates, preds['test'], 
                   label='Predicted Price', color='darkred', linewidth=3, 
                   marker='s', markersize=4, alpha=0.8)
            
            # Fill area between actual and predicted
            ax.fill_between(self.test_dates, preds['test_actual'], preds['test'], 
                           alpha=0.3, color='gray', label='Prediction Error')
            
            # Calculate metrics
            test_mae = mean_absolute_error(preds['test_actual'], preds['test'])
            test_mape = np.mean(np.abs((preds['test_actual'] - preds['test']) / preds['test_actual'])) * 100
            
            ax.set_title(f'{model_name} - TEST DATA ONLY\n'
                        f'Mean Absolute Error: ${test_mae:.2f} | MAPE: {test_mape:.2f}%',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest"""
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_names = self.features.columns
            importances = rf_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance (Random Forest)')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of features"""
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.features.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def run_full_analysis(self):
        """Run complete stock prediction analysis"""
        print(f"🚀 Starting Stock Prediction Analysis for {self.symbol}")
        print("="*60)
        
        # Step 1: Fetch data
        if not self.fetch_data():
            return
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Create features
        self.create_features()
        
        # Step 4: Prepare data
        self.prepare_data()
        
        # Step 4: Train models
        self.train_linear_regression()
        self.train_random_forest()
        
        # Step 5: Evaluate models
        results = self.evaluate_models()
        
        # Step 6: Visualizations
        self.plot_predictions()
        self.plot_feature_importance()
        
        print(f"\n🎯 Analysis completed for {self.symbol}!")
        return results

# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Apple Stock (AAPL)
    print("Example 1: Apple Stock (AAPL)")
    predictor_aapl = StockPredictor('AAPL', '2y')
    results_aapl = predictor_aapl.run_full_analysis()
    
    # Example 2: Google Stock (GOOGL)
    print("\n" + "="*60)
    print("Example 2: Google Stock (GOOGL)")
    predictor_googl = StockPredictor('GOOGL', '2y')
    results_googl = predictor_googl.run_full_analysis()
    
    # Example 3: TCS Stock (for Indian market)
    print("\n" + "="*60)
    print("Example 3: TCS Stock (TCS.NS)")
    predictor_tcs = StockPredictor('TCS.NS', '2y')
    results_tcs = predictor_tcs.run_full_analysis()

# Additional utility functions
def compare_multiple_stocks(symbols=['AAPL', 'GOOGL', 'MSFT'], period='1y'):
    """Compare predictions across multiple stocks"""
    results = {}
    
    for symbol in symbols:
        predictor = StockPredictor(symbol, period)
        if predictor.fetch_data():
            predictor.create_features()
            predictor.prepare_data()
            predictor.train_linear_regression()
            predictor.train_random_forest()
            results[symbol] = predictor.evaluate_models()
    
    return results

def predict_next_day(predictor, model_name='Random Forest'):
    """Predict next day's closing price"""
    if model_name not in predictor.models:
        print(f"Model {model_name} not found!")
        return None
    
    # Get the latest features
    latest_features = predictor.features.iloc[-1:].values
    latest_features_scaled = predictor.scalers['features'].transform(latest_features)
    
    # Make prediction
    model = predictor.models[model_name]
    prediction = model.predict(latest_features_scaled)[0]
    
    current_price = predictor.target.iloc[-1]
    price_change = prediction - current_price
    percentage_change = (price_change / current_price) * 100
    
    print(f"\n📈 Next Day Prediction for {predictor.symbol}:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price: ${prediction:.2f}")
    print(f"Expected Change: ${price_change:.2f} ({percentage_change:.2f}%)")
    
    return prediction
