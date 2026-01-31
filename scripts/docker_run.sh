#!/bin/bash
# Docker helper script for Rap Transcription API

set -e

COMMAND=${1:-help}

case $COMMAND in
    build)
        echo "Building Docker image..."
        docker build -t rap-transcriber:latest .
        echo "✅ Build complete!"
        ;;
    
    run)
        echo "Starting container..."
        docker run -d \
            --name rap-transcriber-api \
            -p 8000:8000 \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/outputs:/app/outputs \
            rap-transcriber:latest
        echo "✅ Container started at http://localhost:8000"
        ;;
    
    stop)
        echo "Stopping container..."
        docker stop rap-transcriber-api || true
        docker rm rap-transcriber-api || true
        echo "✅ Container stopped"
        ;;
    
    logs)
        docker logs -f rap-transcriber-api
        ;;
    
    shell)
        docker exec -it rap-transcriber-api /bin/bash
        ;;
    
    compose-up)
        echo "Starting with Docker Compose..."
        docker-compose up -d rap-transcriber
        echo "✅ Started at http://localhost:8000"
        ;;
    
    compose-down)
        echo "Stopping Docker Compose services..."
        docker-compose down
        echo "✅ Stopped"
        ;;
    
    compose-dev)
        echo "Starting development mode..."
        docker-compose --profile dev up rap-transcriber-dev
        ;;
    
    help|*)
        echo "Rap Transcription Docker Helper"
        echo ""
        echo "Usage: ./scripts/docker_run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  build        Build Docker image"
        echo "  run          Run container"
        echo "  stop         Stop container"
        echo "  logs         View container logs"
        echo "  shell        Open shell in container"
        echo "  compose-up   Start with Docker Compose"
        echo "  compose-down Stop Docker Compose services"
        echo "  compose-dev  Start in development mode"
        echo "  help         Show this help"
        ;;
esac
