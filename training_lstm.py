import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from collections import deque
from tqdm import tqdm
from environment.uav_env import UAVEnv
from models.actor_lstm import LSTMActor
from models.critic_lstm import LSTMCritic
from models.apf import APF
from utils.buffer import MultiAgentReplayBuffer
from utils.noise import GaussianNoise
from config import Config
from evaluation_lstm import visualize_trajectories_lstm

class MALTD3:
    def __init__(self, num_agents=Config.NUM_AGENTS):
        self.num_agents = num_agents
        self.state_dim = Config.STATE_DIM
        self.action_dim = Config.ACTION_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        
        # Initialize actors and critics
        self.actors = [LSTMActor().to(Config.DEVICE) for _ in range(num_agents)]
        self.actor_targets = [LSTMActor().to(Config.DEVICE) for _ in range(num_agents)]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]
        
        self.critics = [LSTMCritic().to(Config.DEVICE) for _ in range(num_agents)]
        self.critic_targets = [LSTMCritic().to(Config.DEVICE) for _ in range(num_agents)]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]
        
        # Initialize target networks
        for i in range(num_agents):
            self._hard_update(self.actor_targets[i], self.actors[i])
            self._hard_update(self.critic_targets[i], self.critics[i])
        
        # Replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(Config.BUFFER_SIZE, num_agents)
        
        # Noise processes
        self.noises = [GaussianNoise(self.action_dim) for _ in range(num_agents)]
        
        # Training parameters
        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.policy_noise = Config.POLICY_NOISE
        self.noise_clip = Config.NOISE_CLIP
        self.policy_freq = Config.POLICY_FREQ
        self.batch_size = Config.BATCH_SIZE
        
        self.total_iterations = 0
        
        # Create directories for saving models
        if not os.path.exists('LSTM_details/models'):
            os.makedirs('LSTM_details/models')
    
    def select_action(self, states, exploration=True):
        actions = []
        hidden_states = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(Config.DEVICE)
            action, hidden = self.actors[i](state)
            
            if exploration:
                noise = torch.FloatTensor(self.noises[i].sample()).to(Config.DEVICE)
                action = (action + noise).clamp(-1, 1)
            
            actions.append(action.squeeze(0).cpu().detach().numpy())
            hidden_states.append(hidden)
        
        return np.array(actions), hidden_states
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.total_iterations += 1
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to appropriate device
        states = states.to(Config.DEVICE)
        actions = actions.to(Config.DEVICE)
        rewards = rewards.to(Config.DEVICE)
        next_states = next_states.to(Config.DEVICE)
        dones = dones.to(Config.DEVICE)
        
        # Train each agent
        for i in range(self.num_agents):
            # Critic training
            with torch.no_grad():
                # Select action with target actor and add noise
                next_action, _ = self.actor_targets[i](next_states[:, i])
                noise = torch.randn_like(next_action) * self.policy_noise
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-1, 1)
                
                # Compute target Q values
                target_Q1, target_Q2, _, _ = self.critic_targets[i](next_states[:, i], next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards[:, i] + (1 - dones[:, i]) * self.gamma * target_Q
            
            # Get current Q estimates
            current_Q1, current_Q2, _, _ = self.critics[i](states[:, i], actions[:, i])
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Optimize critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # Delayed policy updates
            if self.total_iterations % self.policy_freq == 0:
                # Actor training
                actor_loss = -self.critics[i].Q1(states[:, i], self.actors[i](states[:, i])[0])[0].mean()
                
                # Optimize actor
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()
                
                # Update target networks
                self._soft_update(self.critic_targets[i], self.critics[i])
                self._soft_update(self.actor_targets[i], self.actors[i])
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _hard_update(self, target, source):
        target.load_state_dict(source.state_dict())
    
    def save_models(self, episode):
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), f'LSTM_details/models/lstm_actor_{i}_ep_{episode}.pth')
            torch.save(self.critics[i].state_dict(), f'LSTM_details/models/lstm_critic_{i}_ep_{episode}.pth')
    
    def load_models(self, episode):
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(torch.load(f'LSTM_details/models/lstm_actor_{i}_ep_{episode}.pth'))
            self.critics[i].load_state_dict(torch.load(f'LSTM_details/models/lstm_critic_{i}_ep_{episode}.pth'))

def train_lstm():
    # Create necessary directories
    if not os.path.exists('LSTM_details'):
        os.makedirs('LSTM_details')
    if not os.path.exists('LSTM_details/models'):
        os.makedirs('LSTM_details/models')
    if not os.path.exists('LSTM_details/visualizations'):
        os.makedirs('LSTM_details/visualizations')
    if not os.path.exists('LSTM_details/trajectories'):
        os.makedirs('LSTM_details/trajectories')
    
    # Initialize env (no visualization for training)
    env = UAVEnv(visualize=False)
    agent = MALTD3()
    
    episode_rewards = []
    best_reward = -float('inf')
    
    for episode in tqdm(range(Config.MAX_EPISODES), desc="LSTM Training"):
        state, _ = env.reset()
        episode_reward = np.zeros(Config.NUM_AGENTS)
        
        hidden_states = [None] * Config.NUM_AGENTS
        states_history = []  # Collect states during episode
        
        for step in range(Config.MAX_STEPS):
            # Select action
            actions, hidden_states = agent.select_action(state, exploration=True)
            
            # Take action in environment
            next_state, rewards, dones, truncated, _ = env.step(actions)
            
            # Store transition in replay buffer
            agent.replay_buffer.add(state, actions, rewards, next_state, dones)
            
            # Train agent
            agent.train()
            
            # Update state and reward
            state = next_state
            episode_reward += rewards
            
            # Collect state for trajectory visualization
            states_history.append(state)
            
            if all(dones) or all(truncated):
                break
        
        # Save average episode reward
        avg_episode_reward = np.mean(episode_reward)
        episode_rewards.append(avg_episode_reward)
        
        # Save best model
        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            agent.save_models('best')
        
        # Save models only once every 50 episodes
        if episode > 0 and episode % 50 == 0:
            agent.save_models(episode)
            print(f"Saved LSTM model at episode {episode}")
            
        # Print training information at visualization frequency
        if episode % Config.VISUALIZATION_FREQ == 0:
            print(f"LSTM Episode: {episode}, Reward: {avg_episode_reward:.2f}")
            
            # Run silent visualization every 10 episodes
            if episode > 0 and episode % 10 == 0:
                print(f"Generating LSTM visualization for episode {episode}...")
                try:
                    # Save visualization without displaying
                    visualize_trajectories_lstm(episode, display=False)
                except Exception as e:
                    print(f"Error in LSTM visualization: {e}")
                    
                # Save trajectory data
                trajectory_data = {
                    'episode': episode,
                    'states': states_history,
                    'obstacles': env.obstacles
                }
                np.save(f'LSTM_details/trajectories/lstm_trajectory_data_{episode}.npy', trajectory_data)
    
    env.close()
    
    # Save final models
    agent.save_models('final')
    
    # Save reward history
    np.save('LSTM_details/reward_history.npy', np.array(episode_rewards))
    
    # Create and save reward plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('LSTM Training Reward History')
    plt.grid(True)
    plt.savefig('LSTM_details/reward_plot.png')
    plt.close()
    
    return episode_rewards

if __name__ == "__main__":
    train_lstm()