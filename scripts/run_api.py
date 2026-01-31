"""
Run the Rap Transcription API server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080
    python scripts/run_api.py --reload  # Development mode
"""
import sys
sys.path.insert(0, '.')

import argparse
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description='Run Rap Transcription API')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind to')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("RAP TRANSCRIPTION API SERVER")
    print("=" * 50)
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"API docs at http://localhost:{args.port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
