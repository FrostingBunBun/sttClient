# config.py
import json
import os

# Configuration file to save keybind and mic selection
CONFIG_FILE = "stt_config.json"

def load_config():
    """Load configuration from file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        # Create a default config file if it doesn't exist
        default_config = {"keybind": "space", "mic_index": 0}
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f)
        return default_config

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)