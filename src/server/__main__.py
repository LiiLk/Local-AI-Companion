"""
Entry point for running the server as a module.

Usage:
    python -m src.server
"""

import uvicorn
from pathlib import Path
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
