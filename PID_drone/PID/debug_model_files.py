#!/usr/bin/env python3
"""
Debug script to check what model files exist and their contents
"""
import os
import pickle
import pandas as pd
import glob

def check_model_files():
    """Check what model files exist and their contents"""
    
    checkpoint_dir = "/home/osos/Mohamed_Masters_Thesis/DHP_drone_updated/DHP/trained_models"
    
    print("=== DHP Model Files Debug ===")
    print(f"Checking directory: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory does not exist: {checkpoint_dir}")
        return
    
    # List all files in the directory
    all_files = os.listdir(checkpoint_dir)
    print(f"\nAll files in directory ({len(all_files)}):")
    for f in sorted(all_files):
        file_path = os.path.join(checkpoint_dir, f)
        size = os.path.getsize(file_path)
        print(f"  {f} ({size} bytes)")
    
    # Check for specific model files
    model_patterns = [
        "dhp_cf2x_best",
        "dhp_cf2x_final"
    ]
    
    for pattern in model_patterns:
        print(f"\n=== Checking {pattern} ===")
        
        # Look for all files matching pattern
        matching_files = glob.glob(os.path.join(checkpoint_dir, f"{pattern}*"))
        
        if not matching_files:
            print(f"❌ No files found matching {pattern}")
            continue
            
        print(f"Found {len(matching_files)} matching files:")
        for f in sorted(matching_files):
            basename = os.path.basename(f)
            size = os.path.getsize(f)
            print(f"  ✅ {basename} ({size} bytes)")
        
        # Check hyperparameters file specifically
        pkl_file = f"{checkpoint_dir}/{pattern}.pkl"
        if os.path.exists(pkl_file):
            print(f"\n--- Checking hyperparameters: {pattern}.pkl ---")
            try:
                hyperparams = pd.read_pickle(pkl_file)
                print("Hyperparameters loaded successfully!")
                print("Available keys:")
                for key in sorted(hyperparams.keys()):
                    try:
                        value = hyperparams[key]
                        value_type = type(value).__name__
                        if hasattr(value, '__len__') and len(str(value)) > 50:
                            value_str = f"{str(value)[:50]}... (length: {len(value)})"
                        else:
                            value_str = str(value)
                        print(f"  {key}: {value_str} ({value_type})")
                    except Exception as e:
                        print(f"  {key}: <error reading value: {e}>")
                        
            except Exception as e:
                print(f"❌ Error loading hyperparameters: {e}")
        else:
            print(f"❌ Hyperparameters file not found: {pkl_file}")
        
        # Check for TensorFlow checkpoint files
        tf_patterns = [
            f"{pattern}-*.index",
            f"{pattern}-*.meta", 
            f"{pattern}-*.data-*"
        ]
        
        print(f"\n--- TensorFlow checkpoint files for {pattern} ---")
        for tf_pattern in tf_patterns:
            tf_files = glob.glob(os.path.join(checkpoint_dir, tf_pattern))
            if tf_files:
                for tf_file in sorted(tf_files):
                    basename = os.path.basename(tf_file)
                    size = os.path.getsize(tf_file)
                    print(f"  ✅ {basename} ({size} bytes)")
            else:
                print(f"  ❌ No files found: {tf_pattern}")
    
    # Check for metadata files
    print(f"\n=== Metadata Files ===")
    metadata_files = glob.glob(os.path.join(checkpoint_dir, "*metadata*.pkl"))
    
    if metadata_files:
        for meta_file in sorted(metadata_files):
            basename = os.path.basename(meta_file)
            size = os.path.getsize(meta_file)
            print(f"  ✅ {basename} ({size} bytes)")
            
            try:
                with open(meta_file, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"     Contains: {list(metadata.keys())}")
            except Exception as e:
                print(f"     ❌ Error reading: {e}")
    else:
        print("  ❌ No metadata files found")

if __name__ == "__main__":
    check_model_files()