import os


class Config:
    # Flask
    DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

    # Model & inference
    DETECTION_ONNX_MODEL = os.environ.get(
        "DETECTION_ONNX_MODEL",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "rtdetr_eggs.onnx")),
    )
    DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", 0.3))

    # CORS
    ENABLE_CORS = os.environ.get("ENABLE_CORS", "true").lower() == "true"
