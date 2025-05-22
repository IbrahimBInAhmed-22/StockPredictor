"""
Stock Chart Generator and Data Processor
Handles data download, chart generation, and labeling
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class StockDataProcessor:
    def _init_(self, symbol='AAPL', years=10):
        self.symbol = symbol
        self.years = years
        
        # Create directories
        os.makedirs('charts', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
    def download_stock_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Downloading {self.symbol} data for {self.years} years...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years * 365)
        
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {self.symbol}")
        
        print(f"Downloaded {len(data)} days of data")
        
        # Save raw data
        data.to_csv(f'data/{self.symbol}_raw_data.csv')
        print(f"Raw data saved to data/{self.symbol}_raw_data.csv")
        
        return data
    
    def add_technical_indicators(self, data):
        """Add technical indicators to the data"""
        print("Adding technical indicators...")
        
        # Moving averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Remove NaN values
        data = data.dropna()
        
        # Save processed data
        data.to_csv(f'data/{self.symbol}_processed_data.csv')
        print(f"Processed data saved to data/{self.symbol}_processed_data.csv")
        
        return data
    
    def generate_labels(self, data, window=30, future_days=10, threshold=0.10):
        """Generate labels based on future price movement"""
        print("Generating labels...")
        
        labels = []
        chart_indices = []
        label_details = []
        
        for i in range(window, len(data) - future_days):
            current_price = data.iloc[i]['Close']
            current_date = data.index[i]
            
            # Look at future prices
            future_prices = data.iloc[i+1:i+1+future_days]['Close']
            max_future_price = future_prices.max()
            min_future_price = future_prices.min()
            
            # Calculate percentage changes
            max_gain = (max_future_price - current_price) / current_price
            max_loss = (current_price - min_future_price) / current_price
            
            # Label logic: Buy/Sell/Hold
            if max_gain >= threshold:  # 10% gain possible
                label = 0  # Buy
                label_name = "Buy"
            elif max_loss >= threshold:  # 10% loss possible
                label = 1  # Sell
                label_name = "Sell"
            else:
                label = 2  # Hold
                label_name = "Hold"
            
            labels.append(label)
            chart_indices.append(i)
            
            # Store detailed information
            label_details.append({
                'index': i,
                'date': str(current_date.date()),
                'label': label,
                'label_name': label_name,
                'current_price': current_price,
                'max_gain': max_gain,
                'max_loss': max_loss
            })
        
        print(f"Generated {len(labels)} labels")
        label_counts = pd.Series(labels).value_counts()
        print(f"Distribution - Buy: {label_counts.get(0, 0)}, Sell: {label_counts.get(1, 0)}, Hold: {label_counts.get(2, 0)}")
        
        # Save label information
        labels_df = pd.DataFrame(label_details)
        labels_df.to_csv(f'data/{self.symbol}_labels.csv', index=False)
        print(f"Labels saved to data/{self.symbol}_labels.csv")
        
        return labels, chart_indices, label_details
    
    def create_candlestick_charts(self, data, labels, chart_indices, label_details, window=30):
        """Generate candlestick charts with volume and technical indicators"""
        print("Generating candlestick charts...")
        
        chart_info = []
        
        for idx, (label, chart_idx, details) in enumerate(tqdm(zip(labels, chart_indices, label_details), 
                                                              total=len(labels), 
                                                              desc="Creating charts")):
            # Get data window
            start_idx = chart_idx - window
            chart_data = data.iloc[start_idx:chart_idx].copy()
            
            if len(chart_data) < window:
                continue
            
            # Create chart filename
            chart_filename = f"charts/chart_{idx:05d}{label}{details['label_name']}.png"
            
            # Create the plot
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), 
                                   gridspec_kw={'height_ratios': [3, 1]}, 
                                   facecolor='white')
            
            # Main candlestick chart
            ax1 = axes[0]
            
            # Plot candlesticks
            for i, (date, row) in enumerate(chart_data.iterrows()):
                color = 'green' if row['Close'] >= row['Open'] else 'red'
                
                # Candlestick body
                ax1.bar(i, abs(row['Close'] - row['Open']), 
                       bottom=min(row['Close'], row['Open']),
                       color=color, alpha=0.8, width=0.8)
                
                # Wicks
                ax1.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
            
            # Add moving averages
            if 'MA20' in chart_data.columns:
                valid_ma20 = chart_data['MA20'].dropna()
                if len(valid_ma20) > 0:
                    ax1.plot(range(len(chart_data)-len(valid_ma20), len(chart_data)), 
                            valid_ma20, color='blue', linewidth=1, label='MA20', alpha=0.7)
            
            if 'MA50' in chart_data.columns:
                valid_ma50 = chart_data['MA50'].dropna()
                if len(valid_ma50) > 0:
                    ax1.plot(range(len(chart_data)-len(valid_ma50), len(chart_data)), 
                            valid_ma50, color='orange', linewidth=1, label='MA50', alpha=0.7)
            
            # Chart formatting
            ax1.set_title(f'{self.symbol} - {details["date"]} - Label: {details["label_name"]} '
                         f'(Gain: {details["max_gain"]:.2%}, Loss: {details["max_loss"]:.2%})')
            ax1.set_ylabel('Price ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Set y-axis limits for better visualization
            price_min = chart_data[['Low', 'MA20', 'MA50']].min().min()
            price_max = chart_data[['High', 'MA20', 'MA50']].max().max()
            if not (pd.isna(price_min) or pd.isna(price_max)):
                margin = (price_max - price_min) * 0.05
                ax1.set_ylim(price_min - margin, price_max + margin)
            
            # Volume chart
            ax2 = axes[1]
            colors = ['green' if chart_data.iloc[i]['Close'] >= chart_data.iloc[i]['Open'] 
                     else 'red' for i in range(len(chart_data))]
            ax2.bar(range(len(chart_data)), chart_data['Volume'], color=colors, alpha=0.6)
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Days (30-day window)')
            ax2.grid(True, alpha=0.3)
            
            # Format volume axis
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            plt.savefig(chart_filename, dpi=75, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Store chart information
            chart_info.append({
                'chart_id': idx,
                'filename': chart_filename,
                'label': label,
                'label_name': details['label_name'],
                'date': details['date'],
                'data_index': chart_idx,
                'current_price': details['current_price'],
                'max_gain': details['max_gain'],
                'max_loss': details['max_loss']
            })
        
        print(f"Generated {len(chart_info)} chart images")
        
        # Save chart information
        chart_df = pd.DataFrame(chart_info)
        chart_df.to_csv(f'data/{self.symbol}_chart_info.csv', index=False)
        print(f"Chart information saved to data/{self.symbol}_chart_info.csv")
        
        return chart_info
    
    def load_existing_data(self):
        """Load previously processed data if it exists"""
        processed_file = f'data/{self.symbol}_processed_data.csv'
        labels_file = f'data/{self.symbol}_labels.csv'
        chart_info_file = f'data/{self.symbol}_chart_info.csv'
        
        if all(os.path.exists(f) for f in [processed_file, labels_file, chart_info_file]):
            print("Loading existing processed data...")
            
            data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
            labels_df = pd.read_csv(labels_file)
            chart_info_df = pd.read_csv(chart_info_file)
            
            print(f"Loaded {len(data)} data points, {len(labels_df)} labels, {len(chart_info_df)} charts")
            return data, labels_df, chart_info_df
        
        return None, None, None
    
    def process_complete_dataset(self, force_regenerate=False):
        """Process complete dataset: download, label, and create charts"""
        
        # Check if data already exists
        if not force_regenerate:
            existing_data, existing_labels, existing_charts = self.load_existing_data()
            if existing_data is not None:
                print("Using existing processed data. Set force_regenerate=True to recreate.")
                return existing_data, existing_labels, existing_charts
        
        # Step 1: Download data
        data = self.download_stock_data()
        
        # Step 2: Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Step 3: Generate labels
        labels, chart_indices, label_details = self.generate_labels(data)
        
        # Step 4: Create charts
        chart_info = self.create_candlestick_charts(data, labels, chart_indices, label_details)
        
        # Convert to DataFrames for consistency
        labels_df = pd.DataFrame(label_details)
        chart_info_df = pd.DataFrame(chart_info)
        
        print(f"\nDataset Processing Complete!")
        print(f"Symbol: {self.symbol}")
        print(f"Data points: {len(data)}")
        print(f"Charts generated: {len(chart_info)}")
        print(f"Label distribution:")
        label_counts = labels_df['label'].value_counts()
        for i, name in enumerate(['Buy', 'Sell', 'Hold']):
            print(f"  {name}: {label_counts.get(i, 0)}")
        
        return data, labels_df, chart_info_df
    
    def get_dataset_summary(self):
        """Get summary of the processed dataset"""
        try:
            _, labels_df, chart_info_df = self.load_existing_data()
            if labels_df is not None:
                print(f"\nDataset Summary for {self.symbol}:")
                print(f"Total charts: {len(chart_info_df)}")
                
                label_counts = labels_df['label'].value_counts().sort_index()
                for i, name in enumerate(['Buy', 'Sell', 'Hold']):
                    count = label_counts.get(i, 0)
                    percentage = (count / len(labels_df)) * 100
                    print(f"{name}: {count} ({percentage:.1f}%)")
                
                return len(chart_info_df), label_counts.to_dict()
            else:
                print("No processed dataset found. Run process_complete_dataset() first.")
                return 0, {}
        except Exception as e:
            print(f"Error getting dataset summary: {e}")
            return 0, {}

# Usage Example
if _name_ == "_main_":
    # Initialize the data processor
    processor = StockDataProcessor(symbol='AAPL', years=10)
    
    # Process the complete dataset
    data, labels_df, chart_info_df = processor.process_complete_dataset(force_regenerate=False)
    
    # Get summary
    processor.get_dataset_summary()
    
    print("\nData processing completed! You can now use train_model.py to train the neural network.")