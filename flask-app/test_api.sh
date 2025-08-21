#!/bin/bash
set -e

echo "=== Flask Egg Counter API Test ==="

# 1. Copy model if it exists
if [ -f "/home/ewan/omlet-server-egg-det/egg_detection_model.onnx" ]; then
    mkdir -p /home/ewan/omlet-server-egg-det/flask-app/models
    cp -f /home/ewan/omlet-server-egg-det/egg_detection_model.onnx /home/ewan/omlet-server-egg-det/flask-app/models/rtdetr_eggs.onnx
    echo "✓ Model copied to flask-app/models/"
else
    echo "⚠ Model not found at /home/ewan/omlet-server-egg-det/egg_detection_model.onnx"
fi

# 2. Start Flask app (choose one method below)
cd /home/ewan/omlet-server-egg-det/flask-app

echo ""
echo "Choose how to run the Flask app:"
echo "1. Docker (recommended)"
echo "2. Local venv"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    # Docker method
    echo "Building Docker image..."
    docker build -t egg-counter-api:latest .
    
    # Stop existing container if running
    docker rm -f egg-counter-test 2>/dev/null || true
    
    echo "Starting Docker container on port 5000..."
    docker run -d --name egg-counter-test -p 5000:5000 egg-counter-api:latest
    
    echo "Waiting for container to start..."
    sleep 3
    
elif [ "$choice" = "2" ]; then
    # Local venv method
    echo "Setting up local venv..."
    bash scripts/setup_venv.sh
    
    echo "Starting Flask app locally..."
    bash scripts/run_server.sh &
    FLASK_PID=$!
    
    echo "Waiting for Flask to start..."
    sleep 3
else
    echo "Invalid choice"
    exit 1
fi

# 3. Test health endpoint
echo ""
echo "=== Testing Health Endpoint ==="
curl -s http://localhost:5000/health | python3 -m json.tool || echo "Health check failed"

# 4. Test with your local images
echo ""
echo "=== Testing Egg Detection ==="

test_images=(
    "/home/ewan/omlet-server-egg-det/eggs_1.jpg"
    "/home/ewan/omlet-server-egg-det/eggs_2.jpg" 
    "/home/ewan/omlet-server-egg-det/no_eggs.jpg"
)

for img in "${test_images[@]}"; do
    if [ -f "$img" ]; then
        echo ""
        echo "Testing with: $(basename "$img")"
        curl -s -X POST http://localhost:5000/count_eggs \
            -H 'Content-Type: application/json' \
            -d "{\"image_path\":\"$img\"}" | python3 -m json.tool || echo "Request failed"
    else
        echo "⚠ Image not found: $img"
    fi
done

# 5. Show logs if Docker
if [ "$choice" = "1" ]; then
    echo ""
    echo "=== Recent Docker Logs ==="
    docker logs --tail 20 egg-counter-test 2>/dev/null || true
fi

echo ""
echo "=== Test Complete ==="
echo "To stop:"
if [ "$choice" = "1" ]; then
    echo "  docker rm -f egg-counter-test"
else
    echo "  kill $FLASK_PID"
fi
