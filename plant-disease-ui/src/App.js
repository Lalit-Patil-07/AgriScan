import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';

const API_URL = "http://localhost:8000";

const ResultDisplay = ({ result }) => {
  const [activeTab, setActiveTab] = useState('symptoms');
  const { disease_name, confidence, details } = result;
  const confidencePercent = (confidence * 100).toFixed(1);

  const getGaugeColor = (conf) => {
    if (conf > 90) return '#4c956c';
    if (conf > 75) return '#f0ad4e';
    return '#d9534f';
  };
  const gaugeColor = getGaugeColor(confidencePercent);

  return (
    <div className="result-container">
      <h2>Analysis Complete</h2>
      
      <div className="result-main">
        <span className="result-disease">{disease_name}</span>
        <div className="confidence-gauge">
          <div 
            className="confidence-gauge-fill" 
            style={{ width: `${confidencePercent}%`, backgroundColor: gaugeColor }}
          />
          <span className="confidence-gauge-text">{confidencePercent}% Confidence</span>
        </div>
      </div>

      <div className="result-section">
        <p className="result-description">{details.description}</p>
      </div>
      
      <div className="result-tabs">
        <button 
          className={`tab-button ${activeTab === 'symptoms' ? 'active' : ''}`}
          onClick={() => setActiveTab('symptoms')}
        >
          Symptoms
        </button>
        <button 
          className={`tab-button ${activeTab === 'prevention' ? 'active' : ''}`}
          onClick={() => setActiveTab('prevention')}
        >
          Treatment & Prevention
        </button>
      </div>

      <div className="result-tab-content">
        {activeTab === 'symptoms' && (
          <ul>
            {details.symptoms.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        )}
        {activeTab === 'prevention' && (
          <ul>
            {details.prevention.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false
  });

  const handlePredict = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Prediction failed.");
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      console.error("Prediction error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="container">
      <header>
        <h1>Agri-Detect 🌿</h1>
        <p>Upload a plant leaf image to detect diseases in real-time.</p>
      </header>
      
      <main>
        {!preview && (
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            <p>Drag & drop an image here, or click to select</p>
            <em>(Supports .jpg, .jpeg, .png)</em>
          </div>
        )}

        {preview && (
          <div className="preview-container">
            <img src={preview} alt="Selected leaf" className="preview-image" />
          </div>
        )}

        <div className="button-group">
          <button onClick={handlePredict} disabled={!selectedFile || isLoading}>
            {isLoading ? "Analyzing..." : "Analyze Disease"}
          </button>
          <button onClick={handleClear} disabled={!selectedFile && !result} className="secondary">
            Clear
          </button>
        </div>

        {isLoading && <div className="loader"></div>}
        
        {error && <div className="error-message"><strong>Error:</strong> {error}</div>}

        {result && <ResultDisplay result={result} />}
      </main>
    </div>
  );
}

export default App;