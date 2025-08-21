#!/bin/bash
set -e

echo "=== Docker Flask Test with Local Images ==="

# 1. Ensure we have the model
if [ ! -f "models/rtdetr_eggs.onnx" ]; then
    mkdir -p models
    if [ -f "/home/ewan/omlet-server-egg-det/egg_detection_model.onnx" ]; then
        cp /home/ewan/omlet-server-egg-det/egg_detection_model.onnx models/rtdetr_eggs.onnx
        echo "✓ Model copied"
    else
        echo "❌ Model not found"
        exit 1
    fi
fi

# 2. Copy test images to flask-app directory
mkdir -p test-images
for img in eggs_1.jpg eggs_2.jpg no_eggs.jpg; do
    if [ -f "/home/ewan/omlet-server-egg-det/$img" ]; then
        cp "/home/ewan/omlet-server-egg-det/$img" "test-images/$img"
        echo "✓ Copied $img"
    else
        echo "⚠ $img not found"
    fi
done

# 3. Stop existing container and start new one
docker rm -f egg-counter-test 2>/dev/null || true

echo "Starting Docker container..."
docker run -d --name egg-counter-test -p 5000:5000 egg-counter-api:latest

# 4. Wait and test
sleep 5

echo -e "\n=== Testing Health ==="
curl -s http://localhost:5000/health | python3 -m json.tool 2>/dev/null || echo "Health failed"

echo -e "\n=== Testing Egg Detection ==="

# Test with images that are now inside the container
for img in eggs_1.jpg eggs_2.jpg no_eggs.jpg; do
    if [ -f "test-images/$img" ]; then
        echo -e "\nTesting $img:"
        curl -s -X POST http://localhost:5000/count_eggs \
            -H 'Content-Type: application/json' \
            -d "{\"image_path\":\"/app/test-images/$img\"}" | python3 -m json.tool 2>/dev/null || echo "Failed"
    fi
done

echo -e "\n=== Container Logs ==="
docker logs --tail 30 egg-counter-test 2>/dev/null || true

echo -e "\nTo stop: docker rm -f egg-counter-test"
