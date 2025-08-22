from flask import Flask, request, jsonify
from config import Config
from utils.detection import run_detection
from utils.utils import timer, download_file_if_needed, validate_session
import onnxruntime as ort
import os

# Global session placeholder
detection_session = None

# Start detection session so we don't load it every time someone makes a request.
def initialize_sessions():
    global detection_session
    
    try:
        if detection_session is None and os.path.exists(app.config['DETECTION_ONNX_MODEL']):
            providers = ort.get_available_providers()
            detection_session = ort.InferenceSession(app.config['DETECTION_ONNX_MODEL'], providers=providers)
            print("RT-DETR egg detection model loaded successfully")
    except Exception as e:
        print(f"Error initializing model session: {str(e)}")

app = Flask(__name__)
app.config.from_object(Config)

# Optional: CORS for browser clients
try:
    if app.config.get('ENABLE_CORS', False):
        from flask_cors import CORS
        CORS(app)
except Exception as _:
    pass

# Initialize sessions on startup
with app.app_context():
    initialize_sessions()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/count_eggs', methods=['POST'])
@timer
def count_eggs():
    """
    Count eggs in an image using RT-DETR object detection model.
    
    This endpoint performs egg detection in the provided image using an ONNX RT-DETR model
    and returns the count and locations of detected eggs.
    
    Expected JSON payload:
        {
            "image_path": "path/to/image.jpg"
        }
    
    Returns:
        JSON response with detection results:
        - 200: Success with results or info message
        - 400: Bad request (missing image_path)
        - 500: Internal server error during processing
    """
    # Load image
    data = request.get_json()
    image_path = data.get('image_path') if isinstance(data, dict) else None
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400
    
    image_path = download_file_if_needed(image_path)

    global detection_session
    if detection_session is None:
        initialize_sessions()

    # Run detection
    detection_session = validate_session(detection_session, app.config['DETECTION_ONNX_MODEL'])
    try:
        detection_result, labels = run_detection(image_path, app.config['DETECTION_THRESHOLD'], detection_session)
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    # Handle no detections
    if detection_result is None:
        return jsonify({
            "egg_count": 0,
            "detections": [],
            "info": "No eggs detected"
        }), 200
    
    # Prepare detailed response with egg detections
    detections = []
    for i, box in enumerate(detection_result.boxes):
        detections.append({
            "egg_id": i + 1,
            "bounding_box": box.tolist(),
            "confidence": float(detection_result.scores[i]) if hasattr(detection_result, 'scores') and i < len(detection_result.scores) else 1.0
        })

    return jsonify({
        "egg_count": len(detection_result.boxes),
        "detections": detections
    }), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=app.config['DEBUG'])