import os
import sys
import time
from training_lstm import train_lstm
from visualize_lstm_results import visualize_lstm_results

def main():
    """Train and visualize LSTM models"""
    print("=" * 50)
    print("Starting LSTM-based MATD3 Training")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('LSTM_details', exist_ok=True)
    os.makedirs('LSTM_details/models', exist_ok=True)
    os.makedirs('LSTM_details/visualizations', exist_ok=True)
    os.makedirs('LSTM_details/trajectories', exist_ok=True)
    os.makedirs('LSTM_details/animations', exist_ok=True)
    
    # Log start time
    start_time = time.time()
    
    # Train LSTM models
    train_lstm()
    
    # Log training time
    training_time = time.time() - start_time
    print(f"LSTM Training completed in {training_time:.2f} seconds")
    
    # Visualize results
    print("Generating final visualizations...")
    visualize_lstm_results(['best', 'final'])
    
    print("\nLSTM training and visualization complete.")
    print(f"All results saved to LSTM_details/ folder")
    print("=" * 50)

if __name__ == "__main__":
    main()