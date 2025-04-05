import numpy as np
import matplotlib.pyplot as plt
import os
from evaluation_lstm import visualize_trajectories_lstm, generate_lstm_animations

def visualize_lstm_results(episodes=None):
    """Visualize and compare results from LSTM training"""
    # Create directory if it doesn't exist
    os.makedirs('LSTM_details/visualizations', exist_ok=True)
    
    # If no episodes specified, find all model files
    if episodes is None:
        model_files = [f for f in os.listdir('LSTM_details/models') if f.startswith('lstm_actor')]
        episodes = []
        for f in model_files:
            if 'best' in f:
                episodes.append('best')
            elif 'final' in f:
                episodes.append('final')
            else:
                try:
                    ep = int(f.split('_ep_')[1].split('.')[0])
                    if ep not in episodes:
                        episodes.append(ep)
                except:
                    pass
    
    # If episodes is still empty, use default
    if not episodes:
        episodes = ['best']
    
    # Process each episode
    for episode in episodes:
        print(f"Processing LSTM results for episode {episode}")
        
        # Generate trajectory visualization
        trajectories, obstacles = visualize_trajectories_lstm(episode, display=False)
        
        # Generate animation
        generate_lstm_animations(episode)
    
    # If reward history exists, plot it
    if os.path.exists('LSTM_details/reward_history.npy'):
        rewards = np.load('LSTM_details/reward_history.npy')
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('LSTM Training Reward History')
        plt.grid(True)
        plt.savefig('LSTM_details/visualizations/lstm_reward_plot.png')
        plt.close()
    
    print("LSTM visualization complete. Results saved to LSTM_details/visualizations/")

if __name__ == "__main__":
    visualize_lstm_results()