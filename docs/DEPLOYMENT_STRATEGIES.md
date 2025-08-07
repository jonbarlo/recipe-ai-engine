# Recipe AI Engine - Deployment Strategies with Ollama

## Overview

When deploying your recipe generation service to the web, you need to handle the Ollama connection. Here are the main strategies:

## Option 1: Self-Hosted Ollama (Recommended)

### Architecture
```
Internet → Web Server (FastAPI) → Ollama (Same Server)
```

### Pros:
- ✅ **Full Privacy** - No data leaves your server
- ✅ **No API Costs** - No external API fees
- ✅ **Complete Control** - Customize models and prompts
- ✅ **Offline Capable** - Works without internet

### Cons:
- ❌ **Resource Intensive** - Requires GPU/CPU for model inference
- ❌ **Maintenance** - Need to manage Ollama updates
- ❌ **Scaling** - Limited by server resources

### Implementation:

#### 1. Server Setup
```bash
# Install Ollama on your server
curl -fsSL https://ollama.ai/install.sh | sh

# Download your model
ollama pull llama2:7b

# Start Ollama service
ollama serve
```

#### 2. Environment Configuration
```python
# .env file
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama2:7b
```

#### 3. Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy your application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Download model (optional - can be done at runtime)
RUN ollama pull llama2:7b

# Expose ports
EXPOSE 8000 11434

# Start both Ollama and your app
CMD ["sh", "-c", "ollama serve & sleep 10 && uvicorn main:app --host 0.0.0.0 --port 8000"]
```

## Option 2: Separate Ollama Server

### Architecture
```
Internet → Web Server → Network → Ollama Server
```

### Implementation:

#### 1. Dedicated Ollama Server
```bash
# On a separate server with GPU
ollama serve --host 0.0.0.0:11434
```

#### 2. Web Server Configuration
```python
# .env file
OLLAMA_BASE_URL=http://your-ollama-server:11434
DEFAULT_MODEL=llama2:7b
```

#### 3. Network Security
```bash
# Firewall rules
ufw allow from web-server-ip to ollama-server-ip port 11434
```

## Option 3: Cloud Ollama Services

### Services Available:
- **Ollama Cloud** (if available)
- **RunPod** - GPU instances with Ollama
- **Vast.ai** - GPU rentals
- **Google Cloud/AWS** - Custom GPU instances

### Implementation:
```python
# .env file
OLLAMA_BASE_URL=https://your-ollama-cloud-instance.com
DEFAULT_MODEL=llama2:7b
API_KEY=your-api-key
```

## Option 4: Hybrid Approach

### Architecture
```
Internet → Load Balancer → Multiple Web Servers → Ollama Cluster
```

### Implementation:
```python
# Multiple Ollama instances
OLLAMA_INSTANCES=[
    "http://ollama-1:11434",
    "http://ollama-2:11434", 
    "http://ollama-3:11434"
]
```

## Production Deployment Example

### 1. FastAPI Application
```python
# main.py
from fastapi import FastAPI, HTTPException
from recipe_ai_engine import RecipeRequest, RecipeGenerator
import os

app = FastAPI(title="Recipe AI Engine")

# Initialize recipe generator
generator = RecipeGenerator()

@app.post("/recipes/generate")
async def generate_recipe(request: RecipeRequest):
    try:
        recipe = generator.generate_recipe(request)
        return recipe
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    # Check if Ollama is accessible
    if generator.test_connection():
        return {"status": "healthy", "ollama": "connected"}
    else:
        return {"status": "unhealthy", "ollama": "disconnected"}
```

### 2. Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

### 3. Environment Variables
```bash
# .env
OLLAMA_BASE_URL=http://ollama:11434
DEFAULT_MODEL=llama2:7b
TEMPERATURE=0.3
MAX_TOKENS=800
REQUEST_TIMEOUT=60
```

## Scaling Considerations

### 1. Load Balancing
```python
# Multiple Ollama instances
class LoadBalancedRecipeGenerator:
    def __init__(self):
        self.ollama_instances = [
            "http://ollama-1:11434",
            "http://ollama-2:11434",
            "http://ollama-3:11434"
        ]
        self.current_instance = 0
    
    def get_next_instance(self):
        instance = self.ollama_instances[self.current_instance]
        self.current_instance = (self.current_instance + 1) % len(self.ollama_instances)
        return instance
```

### 2. Caching
```python
# Redis caching for generated recipes
import redis
import hashlib
import json

class CachedRecipeGenerator:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.generator = RecipeGenerator()
    
    def generate_recipe(self, request: RecipeRequest):
        # Create cache key
        cache_key = hashlib.md5(
            json.dumps(request.dict(), sort_keys=True).encode()
        ).hexdigest()
        
        # Check cache
        cached = self.redis_client.get(cache_key)
        if cached:
            return RecipeResponse(**json.loads(cached))
        
        # Generate new recipe
        recipe = self.generator.generate_recipe(request)
        
        # Cache for 1 hour
        self.redis_client.setex(
            cache_key, 
            3600, 
            json.dumps(recipe.dict())
        )
        
        return recipe
```

## Security Considerations

### 1. Network Security
```bash
# Firewall rules
ufw allow 8000/tcp  # Web server
ufw allow 11434/tcp # Ollama (if external)
ufw deny 11434/tcp  # Ollama (if internal only)
```

### 2. API Authentication
```python
# Add authentication to your API
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    if not is_valid_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return token.credentials

@app.post("/recipes/generate")
async def generate_recipe(
    request: RecipeRequest, 
    token: str = Depends(verify_token)
):
    # Your recipe generation logic
    pass
```

## Monitoring and Logging

### 1. Health Checks
```python
@app.get("/health/ollama")
async def ollama_health():
    try:
        response = requests.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=5
        )
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### 2. Metrics
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

recipe_generation_counter = Counter(
    'recipe_generation_total', 
    'Total recipe generations'
)

recipe_generation_duration = Histogram(
    'recipe_generation_duration_seconds',
    'Recipe generation duration'
)

@app.post("/recipes/generate")
async def generate_recipe(request: RecipeRequest):
    with recipe_generation_duration.time():
        recipe = generator.generate_recipe(request)
        recipe_generation_counter.inc()
        return recipe
```

## Recommended Deployment Strategy

### For Production:
1. **Use Docker Compose** for easy deployment
2. **Separate Ollama server** with GPU for performance
3. **Add Redis caching** for frequently requested recipes
4. **Implement authentication** for API access
5. **Add monitoring** and health checks
6. **Use load balancer** for multiple instances

### For Development:
1. **Local Ollama** for testing
2. **Docker setup** for consistency
3. **Environment variables** for configuration

This approach gives you full control over your AI model while maintaining privacy and avoiding external API costs. 