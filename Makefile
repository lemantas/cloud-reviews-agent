.PHONY: help build up down restart logs clean health backup shell ingest check-data verify-data

# Default target
help:
	@echo "Agentic RAG Docker Commands"
	@echo "============================"
	@echo ""
	@echo "Container Management:"
	@echo "  make build      - Build Docker image"
	@echo "  make up         - Start containers in background"
	@echo "  make down       - Stop and remove containers"
	@echo "  make restart    - Restart containers"
	@echo "  make logs       - View container logs (follow mode)"
	@echo "  make shell      - Open shell in running container"
	@echo ""
	@echo "Data Management:"
	@echo "  make check-data - Check if data files exist (local)"
	@echo "  make verify-data - Verify data in running container"
	@echo "  make ingest     - Run data ingestion inside container"
	@echo "  make backup     - Backup databases"
	@echo ""
	@echo "Maintenance:"
	@echo "  make health     - Check application health"
	@echo "  make clean      - Remove containers, images, and volumes"
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
	docker-compose run --rm agentic-rag uv run python app/ingest.py

# Check if data exists locally (before deployment)
check-data:
	@echo "Checking local data files..."
	@echo ""
	@if [ -f data/sqlite.db ]; then \
		echo "✅ data/sqlite.db exists ($$(du -h data/sqlite.db | cut -f1))"; \
	else \
		echo "❌ data/sqlite.db NOT FOUND"; \
	fi
	@echo ""
	@if [ -d data/chroma_db ]; then \
		echo "✅ data/chroma_db/ exists ($$(du -sh data/chroma_db | cut -f1))"; \
		echo "   Files: $$(find data/chroma_db -type f | wc -l | tr -d ' ')"; \
	else \
		echo "❌ data/chroma_db/ NOT FOUND"; \
	fi
	@echo ""
	@if [ -d data/reviews ] && [ "$$(ls -A data/reviews/*.csv 2>/dev/null)" ]; then \
		echo "✅ data/reviews/ has CSV files ($$(ls data/reviews/*.csv 2>/dev/null | wc -l | tr -d ' ') files)"; \
	else \
		echo "⚠️  data/reviews/ has no CSV files (needed for ingestion)"; \
	fi
	@echo ""

# Verify data in running container
verify-data:
	@echo "Verifying data in container..."
	@echo ""
	@docker exec agentic-rag-app ls -lh /app/data/sqlite.db 2>/dev/null && echo "✅ sqlite.db mounted" || echo "❌ sqlite.db NOT FOUND in container"
	@echo ""
	@docker exec agentic-rag-app ls -la /app/data/chroma_db 2>/dev/null | head -5 && echo "✅ chroma_db mounted" || echo "❌ chroma_db NOT FOUND in container"
	@echo ""
	@echo "Database statistics:"
	@docker exec agentic-rag-app sqlite3 /app/data/sqlite.db "SELECT COUNT(*) AS total_reviews FROM reviews;" 2>/dev/null || echo "❌ Cannot query database"
	@echo ""
	@echo "Vector store:"
	@docker exec agentic-rag-app python -c "import chromadb; client = chromadb.PersistentClient(path='/app/data/chroma_db'); collection = client.get_collection('reviews'); print(f'Total vectors: {collection.count()}')" 2>/dev/null || echo "❌ Cannot access Chroma"

# Quick start (build + up)
start: build up
	@echo "Application started successfully!"

# Full reset (clean + build + up)
reset: clean build up
	@echo "Application reset complete!"
