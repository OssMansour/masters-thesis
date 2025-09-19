#!/usr/bin/env python3
"""
Train DHP with live rendering enabled
"""
from train_dhp_pendulum import PendulumDHPTrainer

def main():
    print("🚀 Starting DHP training with live rendering...")
    
    # Configuration with rendering enabled
    config_with_rendering = {
        'num_episodes': 100,          # Shorter training for testing
        'render_training': True,      # Enable live rendering
        'render_frequency': 10,       # Render every 10 episodes
        'render_best_only': True,     # Only render when new best is found
        'render_episode_steps': 25,   # Render every 25 steps
        'render_realtime': False,     # Fast rendering
        'log_interval': 5,           # More frequent logging
    }
    
    # Create trainer and update config
    trainer = PendulumDHPTrainer()
    trainer.config.update(config_with_rendering)
    
    print("Training will show live pendulum animation when:")
    print("1. A new best episode is found")
    print("2. Every 10th episode")
    print("3. Every 25 steps within those episodes")
    print("\nStarting training...")
    
    trainer.train()
    
    print("Training completed! Now showing best episode...")
    trainer.visualize_best_episode()

if __name__ == "__main__":
    main()
