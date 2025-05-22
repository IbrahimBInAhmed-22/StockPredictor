import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance import plot as mpf
import mplfinance as mpf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
from PIL import Image
import os
import glob
from tqdm import tqdm
import warnings
import shutil
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class StockChartDataset(Dataset):
    def __init__(self, chart_paths, labels, transform=None):
        self.chart_paths = chart_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.chart_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.chart_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class StockTradingCNN:
    def __init__(self, symbol='AAPL', years=10):
        self.symbol = symbol
        self.years = years
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_accuracy = 0.0
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs('charts', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
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
        return data
    
    def add_technical_indicators(self, data):
        """Add technical indicators to the data"""
        # Moving averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Remove NaN values
        data = data.dropna()
        return data
    
    def generate_labels(self, data, window=30, future_days=10, threshold=0.10):
        """Generate labels based on future price movement"""
        labels = []
        chart_indices = []
        
        for i in range(window, len(data) - future_days):
            current_price = data.iloc[i]['Close']
            
            # Look at future prices
            future_prices = data.iloc[i+1:i+1+future_days]['Close']
            max_future_price = future_prices.max()
            min_future_price = future_prices.min()
            
            # Calculate percentage changes
            max_gain = (max_future_price - current_price) / current_price
            max_loss = (current_price - min_future_price) / current_price
            
            # Label logic: Buy/Sell/Hold
            if max_gain >= threshold:  # 10% gain possible
                labels.append(0)  # Buy
            elif max_loss >= threshold:  # 10% loss possible
                labels.append(1)  # Sell
            else:
                labels.append(2)  # Hold
            
            chart_indices.append(i)
        
        print(f"Generated {len(labels)} labels")
        label_counts = pd.Series(labels).value_counts()
        print(f"Buy: {label_counts.get(0, 0)}, Sell: {label_counts.get(1, 0)}, Hold: {label_counts.get(2, 0)}")
        
        return labels, chart_indices
    
    def create_candlestick_charts(self, data, labels, chart_indices, window=30):
        """Generate candlestick charts with volume and technical indicators"""
        print("Generating candlestick charts...")
        chart_paths = []
        
        for idx, (label, chart_idx) in enumerate(tqdm(zip(labels, chart_indices), total=len(labels))):
            # Get data window
            start_idx = chart_idx - window
            chart_data = data.iloc[start_idx:chart_idx].copy()
            
            if len(chart_data) < window:
                continue
            
            # Create chart filename
            chart_filename = f"charts/chart_{idx}_{label}.png"
            
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
                ax1.plot(range(len(chart_data)), chart_data['MA20'], 
                        color='blue', linewidth=1, label='MA20', alpha=0.7)
            if 'MA50' in chart_data.columns:
                ax1.plot(range(len(chart_data)), chart_data['MA50'], 
                        color='orange', linewidth=1, label='MA50', alpha=0.7)
            
            ax1.set_title(f'{self.symbol} - Label: {["Buy", "Sell", "Hold"][label]}')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Volume chart
            ax2 = axes[1]
            colors = ['green' if chart_data.iloc[i]['Close'] >= chart_data.iloc[i]['Open'] 
                     else 'red' for i in range(len(chart_data))]
            ax2.bar(range(len(chart_data)), chart_data['Volume'], color=colors, alpha=0.6)
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Days')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(chart_filename, dpi=75, bbox_inches='tight')  # 224x224 approx
            plt.close()
            
            chart_paths.append(chart_filename)
        
        print(f"Generated {len(chart_paths)} chart images")
        return chart_paths
    
    def create_model(self, num_classes=3):
        """Create ResNet18-based model for stock prediction"""
        # Load pre-trained ResNet18
        model = models.resnet18(pretrained=True)
        
        # Modify the final layer for our 3 classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        return model.to(self.device)
    
    def load_pretrained_model(self):
        """Check for and load any existing pretrained model"""
        model_path = f"models/best_model_{self.symbol}.pth"
        if os.path.exists(model_path):
            print(f"Found pretrained model: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_accuracy = checkpoint.get('accuracy', 0.0)
            print(f"Loaded model with accuracy: {self.best_accuracy:.4f}")
            return True
        return False
    
    def save_model(self, epoch, accuracy, loss):
        """Save model checkpoint if it's better than previous best"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'accuracy': accuracy,
                'loss': loss,
            }
            torch.save(checkpoint, f"models/best_model_{self.symbol}.pth")
            print(f"New best model saved! Accuracy: {accuracy:.4f}")
    
    def train_model(self, train_loader, val_loader, epochs=150, lr=0.03):
        """Train the CNN model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        print("Starting training...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct_train/total_train:.2f}%'
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{val_loss/(len(val_loader)):.4f}',
                        'Acc': f'{100.*correct_val/total_val:.2f}%'
                    })
            
            val_loss /= len(val_loader)
            val_acc = correct_val / total_val
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print('-' * 50)
            
            # Save best model
            self.save_model(epoch, val_acc, val_loss)
            
            # Update learning rate
            scheduler.step()
        
        return train_losses, train_accuracies, val_losses, val_accuracies
    
    def evaluate_model(self, test_loader):
        """Evaluate model and generate confusion matrix"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        print("\nFinal Test Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Buy', 'Sell', 'Hold'],
                    yticklabels=['Buy', 'Sell', 'Hold'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=['Buy', 'Sell', 'Hold']))
    
    def plot_training_history(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        # Step 1: Download data
        data = self.download_stock_data()
        
        # Step 2: Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Step 3: Generate labels
        labels, chart_indices = self.generate_labels(data)
        
        # Step 4: Create charts
        chart_paths = self.create_candlestick_charts(data, labels, chart_indices)
        
        # Step 5: Prepare data loaders
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Split data 70/30
        X_train, X_test, y_train, y_test = train_test_split(
            chart_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = StockChartDataset(X_train, y_train, transform=transform)
        test_dataset = StockChartDataset(X_test, y_test, transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # Step 6: Create and load model
        self.model = self.create_model()
        self.load_pretrained_model()  # Load if exists
        
        # Step 7: Train model
        train_losses, train_accuracies, val_losses, val_accuracies = self.train_model(
            train_loader, test_loader, epochs=150, lr=0.03
        )
        
        # Step 8: Plot training history
        self.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        # Step 9: Final evaluation
        self.evaluate_model(test_loader)
        
        print("Training completed successfully!")

# Usage Example
if __name__ == "__main__":
    # Initialize the trading CNN
    trader = StockTradingCNN(symbol='AAPL', years=10)
    
    # Run the complete pipeline
    trader.run_complete_pipeline()