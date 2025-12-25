"""
Configuration loader for rap-transcriber.
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config() -> Dict[str, Any]:
    """Get default configuration."""
    return load_config()


if __name__ == "__main__":
    config = load_config()
    print("✅ Config loaded successfully!")
    print(f"   Project: {config['project']['name']}")
    print(f"   Version: {config['project']['version']}")
