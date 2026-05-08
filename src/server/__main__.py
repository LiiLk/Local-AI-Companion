"""
Entry point for running the server as a module.

Usage:
    python -m src.server
"""

import uvicorn
from pathlib import Path
from ..utils.config_loader import load_yaml_config
from .settings import resolve_server_host, resolve_server_port


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    return load_yaml_config(config_path)


def main():
    """Start the server."""
    config = load_config()
    
    uvicorn.run(
        "src.server.app:app",
        host=resolve_server_host(config),
        port=resolve_server_port(config),
        reload=True,
    )


if __name__ == "__main__":
    main()
