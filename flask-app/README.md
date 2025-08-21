# Flask Egg Counting API

A production-ready Flask service that counts eggs using an RT-DETR ONNX model. The API detects eggs in images and returns count + bounding box coordinates.

## Quick Start (Docker)

### 1. Build the Docker Image
```bash
cd flask-app
docker build -t egg-counter-api .
```

### 2. Run the Container
```bash
# Basic run (for URLs or images copied into container)
docker run -d --name egg-counter -p 5000:5000 egg-counter-api

# With volume mount for local images
docker run -d --name egg-counter -p 5000:5000 \
  -v /path/to/your/images:/test-images:ro \
  egg-counter-api
```

### 3. Test the API
```bash
# Health check
curl http://localhost:5000/health

# Count eggs in an image
curl -X POST http://localhost:5000/count_eggs \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"/test-images/your-image.jpg"}'

# Or use a URL
curl -X POST http://localhost:5000/count_eggs \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"https://example.com/eggs.jpg"}'
```

## API Endpoints

### GET /health
Returns API status.

**Response:**
```json
{"status": "ok"}
```

### POST /count_eggs
Detects and counts eggs in an image.

**Request:**
```json
{
  "image_path": "/path/to/image.jpg"  // Local path or HTTP/HTTPS URL
}
```

**Response (Success):**
```json
{
  "egg_count": 5,
  "detections": [
    {
      "egg_id": 1,
      "bounding_box": [681.08, 419.08, 818.73, 542.16],
      "confidence": 0.962
    },
    {
      "egg_id": 2,
      "bounding_box": [412.46, 555.46, 531.22, 670.57],
      "confidence": 0.960
    }
    // ... more detections
  ]
}
```

**Response (No Eggs):**
```json
{
  "egg_count": 0,
  "detections": [],
  "info": "No eggs detected"
}
```

**Response (Error):**
```json
{
  "error": "Error message"
}
```

## Important Notes

### Image Processing
- **Images are automatically resized to 640x640** for inference
- **Bounding box coordinates are scaled back to original image dimensions**
- Bounding boxes are in format: `[x1, y1, x2, y2]` (top-left, bottom-right corners)
- Coordinates are in pixels relative to the original image size

### File Access
- For local images, use Docker volume mounts: `-v /host/path:/container/path:ro`
- The API supports HTTP/HTTPS URLs and will download images automatically
- Images are cached in `/app/downloads/` within the container

### Configuration
Set environment variables when running the container:

```bash
docker run -d --name egg-counter -p 5000:5000 \
  -e DETECTION_THRESHOLD=0.3 \
  -e ENABLE_CORS=true \
  -e DEBUG=false \
  egg-counter-api
```

**Environment Variables:**
- `DETECTION_THRESHOLD`: Confidence threshold (default: 0.3)
- `ENABLE_CORS`: Enable CORS for browser clients (default: true)
- `DEBUG`: Enable Flask debug mode (default: false)
- `DETECTION_ONNX_MODEL`: Path to ONNX model (default: `/app/models/rtdetr_eggs.onnx`)

## Example Usage

### Test with Sample Images
```bash
# Start container with volume mount
docker run -d --name egg-counter -p 5000:5000 \
  -v /home/user/images:/test-images:ro \
  egg-counter-api

# Test multiple images
for img in eggs_1.jpg eggs_2.jpg no_eggs.jpg; do
  echo "Testing $img:"
  curl -s -X POST http://localhost:5000/count_eggs \
    -H 'Content-Type: application/json' \
    -d "{\"image_path\":\"/test-images/$img\"}" | \
    python3 -m json.tool
done
```

### Production Deployment
```bash
# Run with resource limits and restart policy
docker run -d --name egg-counter-prod \
  -p 5000:5000 \
  --restart=unless-stopped \
  --memory=2g \
  --cpus=2 \
  -v /data/images:/images:ro \
  -e DETECTION_THRESHOLD=0.4 \
  egg-counter-api
```

## Cleanup
```bash
# Stop and remove container
docker rm -f egg-counter

# Remove image
docker rmi egg-counter-api
```

## Technical Details
- **Model**: RT-DETR (Real-Time DETR) for object detection
- **Backend**: ONNXRuntime (CPU optimized)
- **Server**: Gunicorn with 4 workers + 4 threads
- **Base Image**: python:3.11-slim
- **Security**: Runs as non-root user (`appuser`)

## Troubleshooting

### Common Issues
1. **"Model not found"**: Ensure `models/rtdetr_eggs.onnx` exists in the build context
2. **"File not found"**: Use volume mounts for local files or check file paths
3. **Memory issues**: Increase Docker memory limits for large images
4. **Permission denied**: Check file permissions on mounted volumes

### Debug Mode
```bash
# Run with debug logs
docker run -it --rm -p 5000:5000 \
  -e DEBUG=true \
  egg-counter-api
```
