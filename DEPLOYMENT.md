# Docker Deployment Guide

This guide covers deploying the Agentic RAG application using Docker, both locally and on a VPS.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Docker Deployment](#local-docker-deployment)
3. [VPS Deployment](#vps-deployment)
4. [Production Setup with Nginx](#production-setup-with-nginx)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
- **OpenAI API Key**: Get one at [platform.openai.com](https://platform.openai.com/api-keys)

### Verify Installation

```bash
docker --version        # Should show Docker version 20.10+
docker-compose --version # Should show Docker Compose version 2.0+
```

---

## Local Docker Deployment

### Step 1: Clone and Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/lemantas/cloud-reviews-agent.git
cd cloud-reviews-agent

# Copy environment variables template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

### Step 2: Prepare Data

**Option A: Use Existing Data (Recommended for Testing)**

If you already have ingested data:
```bash
# Ensure data files exist
ls data/sqlite.db      # SQLite database
ls data/chroma_db/     # Chroma vector store
```

**Option B: Ingest Fresh Data**

```bash
# Run ingestion locally first (outside Docker)
uv run python app/ingest.py

# This creates:
# - data/sqlite.db
# - data/chroma_db/
```

### Step 3: Build and Run

```bash
# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop when done
docker-compose down
```

### Step 4: Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

### Step 5: Verify Health

```bash
# Check container status
docker-compose ps

# Check health endpoint
curl http://localhost:8501/_stcore/health
```

---

## VPS Deployment
I'm leaving this for later. I would wanna have a reverse proxy, if I have deployed more than 1 AI app on a server. 
As of now, I'm only exposing my app to a Cherry Servers VPN network.

### Recommended VPS Providers

| Provider | Plan | Price | Specs |
|----------|------|-------|-------|
| [Cherry Servers](https://www.cherryservers.com/pricing/virtual-servers) | Cloud VPS 1 (GEN 2) | €3/mo | 1 vCPU, 1GB RAM, 20GB SSD |
| [Hetzner Cloud](https://www.hetzner.com/cloud) | CPX11 | €4.51/mo | 2 vCPU, 2GB RAM, 40GB SSD |
| [DigitalOcean](https://www.digitalocean.com/) | Basic | $6/mo | 1 vCPU, 1GB RAM, 25GB SSD |
| [Vultr](https://www.vultr.com/) | Regular Performance | $6/mo | 1 vCPU, 2GB RAM, 55GB SSD |
| [AWS Lightsail](https://aws.amazon.com/lightsail/) | $5 Plan | $5/mo | 1 vCPU, 1GB RAM, 40GB SSD |

### Step 1: Create VPS Instance

**For Hetzner (Example):**

1. Go to [Cherry Servers Console](https://portal.cherryservers.com/)
2. Create new project
3. Add server:
   - Location: Choose nearest to your users
   - Image: Ubuntu 24.04 LTS
   - Type: Cloud VPS 1 (GEN 2) (1 vCPU, 1GB RAM, 20GB SSD)
   - SSH Key: Add your public key
4. Note the server IP address

### Step 2: Initial VPS Setup

```bash
# SSH into your VPS
ssh root@YOUR_VPS_IP

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Verify installation
docker --version
docker-compose --version

# Create a non-root user (recommended)
adduser appuser
usermod -aG docker appuser
usermod -aG sudo appuser

# Switch to new user
su - appuser
```

### Step 3: Deploy Application

```bash
# Clone your repository
git clone <your-repo-url>
cd cloud-reviews-agent

# Set up environment variables
cp .env.example .env
nano .env
# Add your OPENAI_API_KEY

# Copy your pre-ingested data (if available)
# Option A: SCP from local machine
# On your local machine:
scp -r data/sqlite.db appuser@YOUR_VPS_IP:~/cloud-reviews-agent
scp -r data/chroma_db appuser@YOUR_VPS_IP:~/cloud-reviews-agent

# Option B: Run ingestion on VPS
# Make sure data/reviews/*.csv are present, then:
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/app:/app/app \
  --env-file .env \
  python:3.11-slim \
  bash -c "pip install uv && uv run python app/ingest.py"

# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Step 4: Configure Firewall

```bash
# Allow SSH, HTTP, and HTTPS
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8501/tcp  # Streamlit (temporary, will use Nginx later)
sudo ufw enable
sudo ufw status
```

### Step 5: Test Access

```
http://YOUR_VPS_IP:8501
```

---

## Production Setup with Nginx

### Why Use Nginx?

- **SSL/HTTPS**: Secure your app with Let's Encrypt certificates
- **Reverse Proxy**: Hide Streamlit port, use standard HTTP/HTTPS ports
- **Performance**: Static file caching, compression
- **Custom Domain**: Use yourdomain.com instead of IP:8501

### Prerequisites

- Domain name pointed to your VPS IP (e.g., `app.yourdomain.com`)
- VPS with Docker running

### Step 1: Install Nginx

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx -y
```

### Step 2: Configure Nginx

Create Nginx configuration:

```bash
sudo nano /etc/nginx/sites-available/agentic-rag
```

Add this configuration:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name app.yourdomain.com;  # Replace with your domain

    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # Redirect to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name app.yourdomain.com;  # Replace with your domain

    # SSL certificates (will be added by Certbot)
    # ssl_certificate /etc/letsencrypt/live/app.yourdomain.com/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/app.yourdomain.com/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Proxy settings for Streamlit
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;

        # WebSocket support for Streamlit
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;

        # Headers
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Health check endpoint
    location /_stcore/health {
        proxy_pass http://localhost:8501/_stcore/health;
        access_log off;
    }
}
```

### Step 3: Enable Site

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/agentic-rag /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### Step 4: Get SSL Certificate

```bash
# Obtain SSL certificate (interactive)
sudo certbot --nginx -d app.yourdomain.com

# Follow prompts:
# - Enter email address
# - Agree to terms
# - Choose whether to redirect HTTP to HTTPS (recommended: Yes)

# Verify auto-renewal
sudo certbot renew --dry-run
```

### Step 5: Update Firewall

```bash
# Remove direct Streamlit access
sudo ufw delete allow 8501/tcp

# Ensure HTTP/HTTPS are allowed
sudo ufw allow 'Nginx Full'
sudo ufw status
```

### Step 6: Update docker-compose.yml

Since Nginx handles external traffic, bind Streamlit to localhost only:

```yaml
services:
  agentic-rag:
    ports:
      - "127.0.0.1:8501:8501"  # Only accessible from localhost
```

Rebuild and restart:

```bash
docker-compose down
docker-compose up -d
```

### Step 7: Access Your App

```
https://app.yourdomain.com
```

---

## Managing the Deployment

### Start/Stop Application

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild image
docker-compose build

# Restart with new image
docker-compose up -d

# Clean up old images
docker image prune -f
```

### Monitor Resources

```bash
# Container stats
docker stats

# Disk usage
docker system df

# Detailed container info
docker-compose ps
docker inspect agentic-rag-app
```

### Backup Data

```bash
# Backup databases
tar -czf backup-$(date +%Y%m%d).tar.gz data/sqlite.db data/chroma_db/

# Transfer to local machine
scp appuser@YOUR_VPS_IP:~/backup-*.tar.gz ./backups/
```

### Auto-Start on Reboot

Docker Compose with `restart: unless-stopped` ensures containers restart automatically. Verify:

```bash
# Reboot VPS
sudo reboot

# After reboot, check status
docker-compose ps
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs

# Common issues:
# 1. Missing OPENAI_API_KEY in .env
# 2. Port 8501 already in use
# 3. Permission issues with data/ directory

# Check port usage
sudo netstat -tulpn | grep 8501

# Fix permissions
sudo chown -R $(whoami):$(whoami) data/
```

### Cannot Access Application

```bash
# Check if container is running
docker-compose ps

# Check if port is accessible
curl http://localhost:8501/_stcore/health

# Check firewall
sudo ufw status

# Check Nginx (if using)
sudo nginx -t
sudo systemctl status nginx
```

### High Memory Usage

```bash
# Check resource usage
docker stats

# Limit memory in docker-compose.yml:
services:
  agentic-rag:
    deploy:
      resources:
        limits:
          memory: 2G
```

### SSL Certificate Issues

```bash
# Check certificate expiry
sudo certbot certificates

# Renew manually
sudo certbot renew

# Check Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### Data Persistence Issues

```bash
# Verify volume mounts
docker-compose config

# Check if data exists in container
docker-compose exec agentic-rag ls -la /app/data/

# Re-mount volumes if needed
docker-compose down
docker-compose up -d
```

---

## Performance Optimization

### 1. Enable Gzip in Nginx

Add to Nginx server block:

```nginx
gzip on;
gzip_vary on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
```

### 2. Limit Container Resources

```yaml
services:
  agentic-rag:
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 2G
        reservations:
          memory: 1G
```

### 3. Use Docker BuildKit

```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker-compose build
```

---

## Security Best Practices

1. **Keep secrets in .env, never commit**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Regular updates**
   ```bash
   sudo apt update && sudo apt upgrade -y
   docker-compose pull
   ```

3. **Use non-root user** (already covered in VPS setup)

4. **Enable firewall** (already covered)

5. **Monitor logs**
   ```bash
   docker-compose logs --tail=100 -f
   ```

6. **Backup regularly**
   ```bash
   # Weekly cron job
   crontab -e
   # Add:
   0 2 * * 0 tar -czf ~/backups/backup-$(date +\%Y\%m\%d).tar.gz ~/mlevin-AE.3.5/data/
   ```