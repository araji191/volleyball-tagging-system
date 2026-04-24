"""
Flask REST API for volleyball action recognition.
Accepts raw video uploads, runs pose-based inference, and returns
annotated video URL and a list of detected action events.

Authors: Abiola Raji, Patrick Dang
"""

from flask import Flask, request, jsonify, send_from_directory, url_for
import os
import uuid
from werkzeug.utils import secure_filename

from video_pipeline.inference import run_video_inference

app = Flask(__name__)

UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "static_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/outputs/<filename>', methods=['GET'])
def serve_video(filename):
    """Serves annotated output videos from the static outputs directory."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/api/v1/analyze', methods=['POST'])
def analyze_video():
    """Accepts a raw video file, runs inference, and returns events + annotated video URL."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    unique_id = str(uuid.uuid4())
    safe_filename = secure_filename(file.filename)
    input_filepath = os.path.join(UPLOAD_DIR, f"{unique_id}_{safe_filename}")
    output_filename = f"annotated_{unique_id}.mp4"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    file.save(input_filepath)

    try:
        events = run_video_inference(input_filepath, output_filepath)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    finally:
        if os.path.exists(input_filepath):
            os.remove(input_filepath)

    video_url = request.host_url.rstrip('/') + url_for('serve_video', filename=output_filename)

    return jsonify({
        "status": "success",
        "video_url": video_url,
        "events": events
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)