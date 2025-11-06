.PHONY: help build up down restart logs clean test health backup

# Default target
help:
	@echo "Agentic RAG Docker Commands"
	@echo "============================"
	@echo ""
	@echo "  make build      - Build Docker image"
	@echo "  make up         - Start containers in background"
	@echo "  make down       - Stop and remove containers"
	@echo "  make restart    - Restart containers"
	@echo "  make logs       - View container logs (follow mode)"
	@echo "  make health     - Check application health"
	@echo "  make clean      - Remove containers, images, and volumes"
	@echo "  make backup     - Backup databases"
	@echo "  make shell      - Open shell in running container"
	@echo "  make ingest     - Run data ingestion inside container"
	@echo ""

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker-compose build

# Start containers
up:
	@echo "Starting containers..."
	docker-compose up -d
	@echo "Application running at http://localhost:8501"

# Stop containers
down:
	@echo "Stopping containers..."
	docker-compose down

# Restart containers
restart:
	@echo "Restarting containers..."
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

# Check health
health:
	@echo "Checking application health..."
	@curl -f http://localhost:8501/_stcore/health && echo "✅ Application is healthy" || echo "❌ Application is unhealthy"

# Clean up everything
clean:
	@echo "Removing containers, images, and volumes..."
	docker-compose down -v
	docker image prune -f
	@echo "Cleanup complete"

# Backup databases
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup-$$(date +%Y%m%d-%H%M%S).tar.gz data/sqlite.db data/chroma_db/
	@echo "Backup created in backups/"

# Open shell in container
shell:
	docker-compose exec agentic-rag /bin/bash

# Run data ingestion
ingest:
	@echo "Running data ingestion..."
	docker-compose exec agentic-rag uv run python app/ingest.py

# Quick start (build + up)
start: build up
	@echo "Application started successfully!"

# Full reset (clean + build + up)
reset: clean build up
	@echo "Application reset complete!"
