import logging
import sys
import os
import torch
import time
import numpy as np
from datetime import datetime

class Logger:
    def __init__(self, name="APF-MATD3", log_level=logging.INFO, log_to_file=True):
        """Initialize the logger with formatting and output options"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            os.makedirs('logs', exist_ok=True)
            log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_filename}")
        
        # Log system info
        self._log_system_info()
    
    def _log_system_info(self):
        """Log information about the system and environment"""
        self.logger.info("=" * 50)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 50)
        
        # PyTorch and CUDA information
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            self.logger.info("Running on CPU")
        
        # NumPy information
        self.logger.info(f"NumPy version: {np.__version__}")
        
        self.logger.info("=" * 50)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def debug(self, msg):
        self.logger.debug(msg)
        
    # Training specific logging
    def log_episode_stats(self, episode, rewards, steps, collisions=0, goals_reached=0, time_taken=None):
        """Log episode statistics in a consistent format"""
        self.logger.info(
            f"Episode {episode:4d} | "
            f"Avg reward: {np.mean(rewards):8.2f} | "
            f"Min: {np.min(rewards):6.2f} | "
            f"Max: {np.max(rewards):6.2f} | "
            f"Steps: {steps:4d} | "
            f"Collisions: {collisions:2d} | "
            f"Goals: {goals_reached:2d}"
            + (f" | Time: {time_taken:.2f}s" if time_taken else "")
        )
    
    def log_training_config(self, config):
        """Log training configuration parameters"""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 50)
        
        # Convert config object to dict if it's not already
        if not isinstance(config, dict):
            config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('_') and not callable(getattr(config, attr))}
        else:
            config_dict = config
            
        # Log each config parameter
        for key, value in config_dict.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 50)

# Initialize global logger instance
global_logger = Logger()

def get_logger():
    return global_logger
