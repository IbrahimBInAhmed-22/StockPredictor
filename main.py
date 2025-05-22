"""
Main Runner for Stock Trading CNN Project
Coordinates data processing and model training
"""

import os
import sys
from data_processor import StockDataProcessor
from train_model import StockTradingCNN, ModelInference

def print_banner():
    """Print project banner"""
    print("="*70)
    print("              STOCK TRADING CNN PREDICTION SYSTEM")
    print("="*70)
    print("This system downloads stock data, generates labeled charts,")
    print("and trains a CNN to predict Buy/Sell/Hold signals.")
    print("="*70)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'matplotlib', 
        'torch', 'torchvision', 'scikit-learn', 
        'seaborn', 'tqdm', 'pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    """Main execution function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Configuration
    SYMBOL = 'AAPL'  # Change this to any stock symbol
    YEARS = 10       # Years of historical data
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 0.03
    FORCE_REGENERATE = False  # Set to True to regenerate charts
    
    print(f"\nConfiguration:")
    print(f"  Stock Symbol: {SYMBOL}")
    print(f"  Historical Data: {YEARS} years")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Force Regenerate Charts: {FORCE_REGENERATE}")
    
    try:
        # Step 1: Data Processing
        print("\n" + "="*50)
        print("STEP 1: DATA PROCESSING & CHART GENERATION")
        print("="*50)
        
        processor = StockDataProcessor(symbol=SYMBOL, years=YEARS)
        
        # Check if data already exists
        total_charts, label_dist = processor.get_dataset_summary()
        
        if total_charts == 0 or FORCE_REGENERATE:
            print("Processing stock data and generating charts...")
            data, labels_df, chart_info_df = processor.process_complete_dataset(
                force_regenerate=FORCE_REGENERATE
            )
        else:
            print("Using existing processed data and charts.")
            print("Set FORCE_REGENERATE=True to recreate charts.")
        
        # Show final summary
        total_charts, label_dist = processor.get_dataset_summary()
        
        if total_charts == 0:
            print("❌ No charts generated. Please check the data processing step.")
            return
        
        print(f"✅ Dataset ready with {total_charts} charts")
        
        # Step 2: Model Training
        print("\n" + "="*50)
        print("STEP 2: CNN MODEL TRAINING")
        print("="*50)
        
        trainer = StockTradingCNN(symbol=SYMBOL)
        
        # Run training pipeline
        metrics = trainer.run_training_pipeline(
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )
        
        # Step 3: Model Testing
        print("\n" + "="*50)
        print("STEP 3: MODEL TESTING")
        print("="*50)
        
        # Test the trained model
        inference = ModelInference(symbol=SYMBOL)
        inference.load_trained_model()
        
        # Find a sample chart to test
        chart_files = [f for f in os.listdir('charts') if f.endswith('.png')]
        if chart_files:
            sample_chart = os.path.join('charts', chart_files[0])
            print(f"Testing model with sample chart: {chart_files[0]}")
            
            result = inference.predict_single_chart(sample_chart)
            if result:
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"  {class_name}: {prob:.4f}")
        
        print("\n" + "="*70)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Files generated:")
        print(f"  📊 Charts: charts/ directory ({total_charts} images)")
        print(f"  🤖 Model: models/best_model_{SYMBOL}.pth")
        print(f"  📈 Training History: models/{SYMBOL}_training_history.png")
        print(f"  📊 Confusion Matrix: models/{SYMBOL}_confusion_matrix.png")
        print(f"  📄 Data Files: data/ directory")
        print("\nYour CNN model is now ready to predict stock trading signals!")
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("Please ensure all required files are in the correct directories.")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Please check your setup and try again.")

def run_data_processing_only():
    """Run only the data processing step"""
    print("Running data processing only...")
    processor = StockDataProcessor(symbol='AAPL', years=10)
    processor.process_complete_dataset(force_regenerate=True)
    print("Data processing completed!")

def run_training_only():
    """Run only the model training step"""
    print("Running model training only...")
    trainer = StockTradingCNN(symbol='AAPL')
    trainer.run_training_pipeline(batch_size=32, epochs=150, lr=0.03)
    print("Model training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Trading CNN System')
    parser.add_argument('--mode', choices=['full', 'data', 'train'], default='full',help='Run mode: full pipeline, data processing only, or training only')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol to analyze')
    parser.add_argument('--years', type=int, default=10, help='Years of historical data')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regenerate charts even if they exist')
    
    args = parser.parse_args()
    
    # Update global configuration
    if len(sys.argv) > 1:
        globals().update({
            'SYMBOL': args.symbol,
            'YEARS': args.years,
            'EPOCHS': args.epochs,
            'BATCH_SIZE': args.batch_size,
            'LEARNING_RATE': args.lr,
            'FORCE_REGENERATE': args.force_regenerate
        })
    
    # Run based on mode
    if args.mode == 'full':
        main()
    elif args.mode == 'data':
        run_data_processing_only()
    elif args.mode == 'train':
        run_training_only()