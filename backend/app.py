from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename

# Import the inference pipeline
from video_pipeline.inference import run_video_inference

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 

# Create directories for temporary uploads and processed outputs
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "static_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/outputs/<filename>', methods=['GET'])
def serve_video(filename):
    """Serves the generated output videos for download."""
    return send_from_directory(OUTPUT_DIR, filename, mimetype="video/mp4")

@app.route('/api/v1/analyze', methods=['POST'])
def analyze_video():
    """Handles the raw video upload, runs inference, and returns data + video URL."""
    # 1. Validate the incoming request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided in the request."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # 2. Secure the filename and generate unique paths
    if file:
        unique_id = str(uuid.uuid4())
        safe_filename = secure_filename(file.filename)
        input_filename = f"{unique_id}_{safe_filename}"
        output_filename = f"annotated_{unique_id}.mp4"
        
        input_filepath = os.path.join(UPLOAD_DIR, input_filename)
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        # 3. Save the raw video locally
        file.save(input_filepath)
        
        # 4. Run the inference
        try:
            events = run_video_inference(input_filepath, output_filepath)
        except Exception as e:
            return jsonify({"error": f"Inference failed: {str(e)}"}), 500
        finally:
            # 5. Clean up the raw upload to save space
            if os.path.exists(input_filepath):
                os.remove(input_filepath)
                
        # 6. Generate the full download URL for the annotated video
        # request.host_url automatically captures your domain (e.g., http://localhost:5000/)
        # Use ONLY url_for with _external=True. Flask handles the localhost:5000 part for you.
        video_url = url_for('serve_video', filename=output_filename, _external=True)
        
        # 7. Return the combined JSON response
        return jsonify({
            "status": "success",
            "video_url": video_url,
            "events": events
        })

if __name__ == '__main__':
    # Run the server on all available IPs on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)