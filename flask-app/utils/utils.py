import os
import time
import urllib.parse
import urllib.request
from functools import wraps
from typing import Optional


def timer(func):
    """Simple timing decorator that prints the execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            duration = (time.time() - start) * 1000.0
            print(f"{func.__name__} took {duration:.1f} ms")
    return wrapper


def download_file_if_needed(path_or_url: str, downloads_dir: Optional[str] = None) -> str:
    """Return a local file path. If given an http/https URL, download to a cache folder once.

    Args:
        path_or_url: local path or URL.
        downloads_dir: optional directory to store downloads (defaults to ./flask-app/downloads)
    """
    if not path_or_url:
        raise ValueError("Empty image path provided")

    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        # Prepare downloads dir
        base_dir = downloads_dir or os.path.join(os.path.dirname(__file__), "..", "downloads")
        base_dir = os.path.abspath(base_dir)
        os.makedirs(base_dir, exist_ok=True)

        # Use filename from URL
        filename = os.path.basename(parsed.path) or "image.jpg"
        local_path = os.path.join(base_dir, filename)

        if not os.path.exists(local_path):
            print(f"Downloading {path_or_url} -> {local_path}")
            urllib.request.urlretrieve(path_or_url, local_path)
        return local_path
    else:
        # Assume local file path
        return path_or_url


def validate_session(session, model_path: str):
    """Create an ONNXRuntime session if needed. Return the existing one otherwise."""
    import onnxruntime as ort

    if session is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")
        providers = ort.get_available_providers()
        print(f"Creating ONNX Runtime session with providers: {providers}")
        return ort.InferenceSession(model_path, providers=providers)
    return session
