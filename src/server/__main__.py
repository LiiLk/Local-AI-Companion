"""
Entry point for running the server as a module.

Usage:
    python -m src.server
"""

import uvicorn
from pathlib import Path
from ..utils.config_loader import load_yaml_config


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    return load_yaml_config(config_path)


def main():
    """Start the server."""
    config = load_config()
    server_config = config.get("server", {})
    
    uvicorn.run(
        "src.server.app:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=True,
    )


if __name__ == "__main__":
    main()
