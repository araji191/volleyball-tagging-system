import { useState } from 'react';
import axios from 'axios';
import './App.css';
import volleyballIcon from './assets/volleyball.jpg';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a video file first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file); // Matches your Flask request.files['file']

    try {
      const response = await axios.post('http://localhost:5000/api/v1/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred during inference.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div id="root">
      <section id="center">
        <h1 className="karasuno-title">Volleyball Action Tracker</h1>
        <p>Upload a video to detect and tag player actions.</p>

        <div className="upload-container">
          <input 
            type="file" 
            accept="video/*" 
            onChange={handleFileChange} 
            className="file-input"
          />
          <button 
            className="counter" 
            onClick={handleUpload} 
            disabled={loading}
          >
            {loading ? "Processing Video..." : "Run Inference"}
          </button>
        </div>

        {error && <p style={{ color: '#ff4d4d' }}>{error}</p>}

        {loading && (
          <div className="loader-box">
            <img src={volleyballIcon} className="spinner-image" alt="Loading..." />
            <p>Our model is analyzing the frames... This may take a minute.</p>
          </div>
        )}

        {result && (
          <div className="results-grid">
            <div className="video-section">
              <h3>Annotated Video</h3>
              <video controls src={result.video_url} className="output-video" />
            </div>
            
            <div className="events-section">
              <h3>Detected Actions</h3>
              <div className="events-list">
                {result.events.length > 0 ? (
                  result.events.map((event, index) => (
                    <div key={index} className="event-card">
                      <span className="action-tag">{event.action}</span>
                      <span className="timestamp">{event.start_ts}s - {event.end_ts}s</span>
                    </div>
                  ))
                ) : (
                  <p>No specific actions detected.</p>
                )}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;