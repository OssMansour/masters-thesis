#!/usr/bin/env python3
"""
Quick script to view the best episode after training
"""
from train_dhp_pendulum import PendulumDHPTrainer

def main():
    print("🎥 Loading trained model and visualizing best episode...")
    
    # Create trainer (it will load existing model if available)
    trainer = PendulumDHPTrainer()
    
    # Try to load the best model
    try:
        trainer.load_best_model()
        print("✓ Best model loaded successfully!")
        
        # Show the best episode with rendering
        print("Starting best episode visualization...")
        trainer.visualize_best_episode(steps=100)
        
    except Exception as e:
        print(f"Could not load best model: {e}")
        print("Make sure you have trained a model first!")
        
        # Alternative: demonstrate with current policy
        print("Demonstrating current policy instead...")
        trainer.demonstrate_policy(gui=True, episode_length=100)

if __name__ == "__main__":
    main()
