"""
TensorBoard Launcher Script

This script launches TensorBoard to visualize training logs.
You can specify the log directory as a command-line argument or modify the default path.

Usage:
    python tensorboardScript.py                           # Use default log directory
    python tensorboardScript.py path/to/logs              # Specify custom log directory
    python tensorboardScript.py --port 6006               # Specify custom port
    python tensorboardScript.py path/to/logs --port 6007  # Both custom directory and port
"""

import sys
import os
import subprocess
import argparse


def launch_tensorboard(log_dir, port=6006):
    """
    Launch TensorBoard with the specified log directory and port.
    
    Args:
        log_dir (str): Path to the directory containing TensorBoard logs
        port (int): Port number for TensorBoard server (default: 6006)
    """
    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' does not exist!")
        print("\nAvailable log directories:")
        
        # Try to find common log directories
        possible_dirs = [
            './ppo_logs/',
            '../highway_ppo/',
            '../highway_dqn/',
            '../highway_td3/',
            './logs/'
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                print(f"  - {os.path.abspath(dir_path)}")
        
        return False
    
    print(f"Starting TensorBoard...")
    print(f"Log directory: {os.path.abspath(log_dir)}")
    print(f"Port: {port}")
    print(f"\nOpen your browser to: http://localhost:{port}")
    print("\nPress Ctrl+C to stop TensorBoard\n")
    
    try:
        # Launch TensorBoard
        subprocess.run([
            "tensorboard",
            f"--logdir={log_dir}",
            f"--port={port}",
            "--bind_all"
        ])
    except KeyboardInterrupt:
        print("\n\nTensorBoard stopped.")
    except FileNotFoundError:
        print("\nError: TensorBoard is not installed!")
        print("Install it with: pip install tensorboard")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard to visualize training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tensorboardScript.py
  python tensorboardScript.py ../highway_ppo/logs_20251201_120000
  python tensorboardScript.py ./ppo_logs --port 6007
        """
    )
    
    parser.add_argument(
        'log_dir',
        nargs='?',
        default='./ppo_logs/',
        help='Path to the log directory (default: ./ppo_logs/)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='Port for TensorBoard server (default: 6006)'
    )
    
    args = parser.parse_args()
    
    # Launch TensorBoard
    success = launch_tensorboard(args.log_dir, args.port)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
