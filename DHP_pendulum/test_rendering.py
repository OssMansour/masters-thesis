#!/usr/bin/env python3
"""
Test script for DHP pendulum rendering functionality

This script demonstrates how to:
1. Enable rendering during training
2. View the best episode
3. Use the visualization functions
"""

import sys
import os
sys.path.append('/home/osos/Mohamed_Masters_Thesis/DHP_pendulum')

from train_dhp_pendulum import PendulumDHPTrainer
from dhp_compatible_pendulum_env import create_dhp_pendulum
import numpy as np

def test_basic_rendering():
    """Test basic environment rendering"""
    print("="*60)
    print("TEST 1: Basic Environment Rendering")
    print("="*60)
    
    # Create environment
    env = create_dhp_pendulum()
    
    # Test the built-in visualization
    print("Running environment visualization test...")
    env.test_visualization(steps=30)
    print("✓ Basic rendering test completed!")

if __name__ == "__main__":
    print("DHP PENDULUM RENDERING TEST")
    print("="*60)
    test_basic_rendering()
