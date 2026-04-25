import os
import json

CONFIG_FILE = "config.json"

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Missing {CONFIG_FILE}. Please create it first.")

with open(CONFIG_FILE, 'r') as f:
    raw_config = json.load(f)

# Flatten the nested JSON for easy access across the app
config = {}
for section, settings in raw_config.items():
    if isinstance(settings, dict):
        for key, value in settings.items():
            config[key] = value
    else:
        config[section] = settings

# Ensure fallback defaults for list-based configs
config['video_folders'] = config.get('video_folders', ['./my_drone_videos'])
config['music_folders'] = config.get('music_folders', [])

# Ensure log directory exists
log_path = config.get('log_file', 'logs/drone_editor.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)