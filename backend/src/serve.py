from __future__ import annotations

import argparse
from typing import Optional

import uvicorn


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sandbox FastAPI backend via uvicorn."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload for development.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Uvicorn log level (default: info).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
