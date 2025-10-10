#!/usr/bin/env python3
"""
TensorBoard Launcher Script
Run this in a separate terminal while training is running
"""

import subprocess
import sys
import os
import webbrowser
import time

def launch_tensorboard():
    """Launch TensorBoard and open browser"""
    
    log_dir = "logs/tensorboard/SAC_fresh"
    
    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        print("Make sure training has started and created logs")
        return
    
    print(f"🚀 Launching TensorBoard for: {log_dir}")
    print("📊 TensorBoard will show:")
    print("   - Training rewards and losses")
    print("   - Evaluation metrics")
    print("   - Learning curves")
    print("   - Network statistics")
    
    try:
        # Launch TensorBoard
        print(f"\n🔧 Starting TensorBoard server...")
        process = subprocess.Popen([
            sys.executable, "-m", "tensorboard.main", 
            f"--logdir={log_dir}",
            "--port=6006",
            "--reload_interval=30"  # Refresh every 30 seconds
        ])
        
        # Wait a moment for server to start
        print("⏳ Waiting for TensorBoard server to start...")
        time.sleep(3)
        
        # Open browser
        url = "http://localhost:6006"
        print(f"🌐 Opening browser to: {url}")
        webbrowser.open(url)
        
        print("\n✅ TensorBoard is running!")
        print("📈 You should see training metrics in your browser")
        print("🔄 Metrics update automatically during training")
        print("\n💡 Useful tips:")
        print("   - Use SCALARS tab for loss/reward curves")
        print("   - Use HISTOGRAMS tab for network weights")
        print("   - Use IMAGES tab if you log images")
        print("\n⛔ Press Ctrl+C to stop TensorBoard")
        
        # Keep process alive
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping TensorBoard...")
            process.terminate()
            process.wait()
            print("✅ TensorBoard stopped")
            
    except Exception as e:
        print(f"❌ Error launching TensorBoard: {e}")
        print("💡 Try running manually: tensorboard --logdir=logs/tensorboard/SAC_fresh")

if __name__ == "__main__":
    launch_tensorboard()
