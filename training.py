"""
Stock Trading CNN Model Training
Uses pre-generated charts to train the neural network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class StockChartDataset(Dataset):
    def _init_(self, chart_info_df, transform=None):
        self.chart_info = chart_info_df.reset_index(drop=True)
        self.transform = transform
        
        # Filter out charts that don't exist
        existing_charts = []
        for idx, row in self.chart_info.iterrows():
            if os.path.exists(row['filename']):
                existing_charts.append(idx)
        
        self.chart_info = self.chart_info.loc[existing_charts].reset_index(drop=True)
        print(f"Dataset initialized with {len(self.chart_info)} valid charts")
    
    def _len_(self):
        return len(self.chart_info)
    
    def _getitem_(self, idx):
        row = self.chart_info.iloc[idx]
        image_path = row['filename']
        label = int(row['label'])
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if there's an error
            image = torch.zeros((3, 224, 224))
        
        return image, label

class StockTradingCNN:
    def _init_(self, symbol='AAPL'):
        self.symbol = symbol
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_accuracy = 0.0
        print(f"Using device: {self.device}")
        
        # Create model directory
        os.makedirs('models', exist_ok=True)
        
    def load_chart_data(self):
        """Load chart information from processed data"""
        chart_info_file = f'data/{self.symbol}_chart_info.csv'
        
        if not os.path.exists(chart_info_file):
            raise FileNotFoundError(f"Chart info file not found: {chart_info_file}")
        
        chart_info_df = pd.read_csv(chart_info_file)
        print(f"Loaded chart information for {len(chart_info_df)} charts")
        
        # Display label distribution
        label_counts = chart_info_df['label'].value_counts().sort_index()
        print("Label distribution:")
        for i, name in enumerate(['Buy', 'Sell', 'Hold']):
            count = label_counts.get(i, 0)
            percentage = (count / len(chart_info_df)) * 100 if len(chart_info_df) > 0 else 0
            print(f"  {name}: {count} ({percentage:.1f}%)")
        
        return chart_info_df
    
    def create_data_loaders(self, chart_info_df, batch_size=32, test_size=0.3):
        """Create train and validation data loaders"""
        
        # Data transformations
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),  # Light augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Split data 70/30 while maintaining class distribution
        train_df, val_df = train_test_split(
            chart_info_df, 
            test_size=test_size, 
            random_state=42, 
            stratify=chart_info_df['label']
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        
        # Create datasets
        train_dataset = StockChartDataset(train_df, transform=train_transform)
        val_dataset = StockChartDataset(val_df, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
    
    def create_model(self, num_classes=3):
        """Create ResNet18-based model for stock prediction"""
        print("Creating ResNet18-based model...")
        
        # Load pre-trained ResNet18
        model = models.resnet18(pretrained=True)
        
        # Freeze early layers (optional - for faster training)
        # for param in model.parameters():
        #     param.requires_grad = False
        
        # Modify the final layer for our 3 classes
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        return model.to(self.device)
    
    def load_pretrained_model(self):
        """Check for and load any existing pretrained model"""
        model_path = f"models/best_model_{self.symbol}.pth"
        
        if os.path.exists(model_path):
            print(f"Found pretrained model: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.best_accuracy = checkpoint.get('accuracy', 0.0)
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Loaded model from epoch {start_epoch} with accuracy: {self.best_accuracy:.4f}")
                return start_epoch + 1
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Starting training from scratch...")
                return 0
        else:
            print("No pretrained model found. Starting from scratch...")
            return 0
    
    def save_model(self, epoch, accuracy, loss, optimizer):
        """Save model checkpoint if it's better than previous best"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': loss,
                'symbol': self.symbol
            }
            
            model_path = f"models/best_model_{self.symbol}.pth"
            torch.save(checkpoint, model_path)
            print(f"✓ New best model saved! Accuracy: {accuracy:.4f}")
            return True
        return False
    
    def train_model(self, train_loader, val_loader, epochs=150, lr=0.03, start_epoch=0):
        """Train the CNN model"""
        print(f"Starting training from epoch {start_epoch}...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Load optimizer state if resuming
        if start_epoch > 0:
            model_path = f"models/best_model_{self.symbol}.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        print(f"Training for {epochs - start_epoch} epochs...")
        print("=" * 60)
        
        for epoch in range(start_epoch, epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # Training progress bar
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{epochs} [TRAIN]', 
                            leave=False, ncols=100)
            
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
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct_train / total_train
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.1f}%'
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1:3d}/{epochs} [VAL]  ', 
                              leave=False, ncols=100)
                
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
                    
                    # Update progress bar
                    current_val_loss = val_loss / len(val_loader)
                    current_val_acc = 100. * correct_val / total_val
                    val_pbar.set_postfix({
                        'Loss': f'{current_val_loss:.4f}',
                        'Acc': f'{current_val_acc:.1f}%'
                    })
            
            val_loss /= len(val_loader)
            val_acc = correct_val / total_val
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f'Epoch {epoch+1:3d}/{epochs} | '
                  f'Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | '
                  f'Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}')
            
            # Save best model
            saved = self.save_model(epoch, val_acc, val_loss, optimizer)
            if saved:
                print(f'             | Best model updated! Previous best: {self.best_accuracy:.4f}')
            
            # Update learning rate
            scheduler.step()
            
            print("-" * 60)
        
        print("Training completed!")
        return train_losses, train_accuracies, val_losses, val_accuracies
    
    def evaluate_model(self, val_loader):
        """Evaluate model and generate confusion matrix"""
        print("Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        print("\n" + "="*50)
        print("FINAL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Buy', 'Sell', 'Hold'],
                    yticklabels=['Buy', 'Sell', 'Hold'],
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.symbol} Stock Prediction\nAccuracy: {accuracy:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j+0.5, i+0.7, f'{cm_percent[i, j]:.1%}', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig(f'models/{self.symbol}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(all_targets, all_predictions, target_names=['Buy', 'Sell', 'Hold']))
        
        return accuracy, precision, recall, f1
    
    def plot_training_history(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{self.symbol} Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title(f'{self.symbol} Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'models/{self.symbol}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_training_pipeline(self, batch_size=32, epochs=150, lr=0.03):
        """Run the complete training pipeline"""
        print("="*60)
        print(f"STOCK TRADING CNN - {self.symbol}")
        print("="*60)
        
        # Step 1: Load chart data
        chart_info_df = self.load_chart_data()
        
        # Step 2: Create data loaders
        train_loader, val_loader = self.create_data_loaders(chart_info_df, batch_size)
        
        # Step 3: Create model
        self.model = self.create_model()
        
        # Step 4: Load pretrained model if exists
        start_epoch = self.load_pretrained_model()
        
        # Step 5: Train model
        train_losses, train_accuracies, val_losses, val_accuracies = self.train_model(
            train_loader, val_loader, epochs=epochs, lr=lr, start_epoch=start_epoch
        )
        
        # Step 6: Plot training history
        if len(train_losses) > 0:
            self.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        # Step 7: Final evaluation
        metrics = self.evaluate_model(val_loader)
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model saved with accuracy: {self.best_accuracy:.4f}")
        print(f"Model location: models/best_model_{self.symbol}.pth")
        print(f"Training history: models/{self.symbol}_training_history.png")
        print(f"Confusion matrix: models/{self.symbol}_confusion_matrix.png")
        
        return metrics

class ModelInference:
    """Class for making predictions with trained model"""
    
    def _init_(self, symbol='AAPL'):
        self.symbol = symbol
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_trained_model(self):
        """Load the trained model for inference"""
        model_path = f"models/best_model_{self.symbol}.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        # Create model architecture
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.model = model.to(self.device)
        print(f"Loaded trained model with accuracy: {checkpoint.get('accuracy', 'Unknown'):.4f}")
    
    def predict_single_chart(self, image_path):
        """Make prediction for a single chart image"""
        if self.model is None:
            self.load_trained_model()
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
            
            class_names = ['Buy', 'Sell', 'Hold']
            confidence = probabilities[0][predicted_class].item()
            
            return {
                'prediction': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'Buy': probabilities[0][0].item(),
                    'Sell': probabilities[0][1].item(),
                    'Hold': probabilities[0][2].item()
                }
            }
        
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

# Usage Example and Main Execution
if __name__ == "__main__":
    # Initialize the model trainer
    trainer = StockTradingCNN(symbol='AAPL')
    
    # Run the complete training pipeline
    try:
        metrics = trainer.run_training_pipeline(
            batch_size=32,
            epochs=150,
            lr=0.03
        )
        
        print("\nTraining completed successfully!")
        print("You can now use the trained model for predictions.")
        
        # Example of how to use the model for inference
        print("\nExample: Loading model for inference...")
        inference = ModelInference(symbol='AAPL')
        inference.load_trained_model()
        print("Model ready for predictions!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run data_processor.py first to generate the chart dataset.")
        print("Example:")
        print("python data_processor.py")
        print("python train_model.py")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please check your setup and try again.")
