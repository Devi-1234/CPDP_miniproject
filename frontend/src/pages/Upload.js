import React, { useState } from "react";
import Navbar from "../components/Navbar";

// Configuration
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:5000";
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB

export default function Upload() {
  const [selectedAdaptation, setSelectedAdaptation] = useState("");
  const [selectedMLModel, setSelectedMLModel] = useState("");
  const [targetFile, setTargetFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const validateFile = (file) => {
    if (!file) return "Please select a file";
    if (!file.name.endsWith('.csv')) return "Only CSV files are allowed";
    if (file.size > MAX_FILE_SIZE) return "File size exceeds 16MB limit";
    return null;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    
    // Validation
    if (!selectedAdaptation || !selectedMLModel) {
      setError("Please select both adaptation method and ML model");
      setIsLoading(false);
      return;
    }
    
    const targetError = validateFile(targetFile);
    const testError = validateFile(testFile);
    
    if (targetError || testError) {
      setError(targetError || testError);
      setIsLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("targetFile", targetFile);
    formData.append("testFile", testFile);
    formData.append("modelAdaptation", selectedAdaptation);
    formData.append("mlModel", selectedMLModel);

    try {
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process files");
      }

      const result = await response.json();
      setResults(result);
    } catch (error) {
      setError(error.message || "Error uploading files");
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatResults = (results) => {
    if (!results) return null;
    
    return (
      <div className="mt-4">
        <h2>Results</h2>
        <div className="card">
          <div className="card-body">
            {Object.entries(results).map(([key, value]) => (
              <div key={key} className="mb-2">
                <strong>{key}:</strong> {typeof value === 'number' ? value.toFixed(4) : value}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div>
      <Navbar />
      <div className="container mt-5">
        <h1 className="text-center">Upload Datasets</h1>
        {error && (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        )}
        <form className="mt-4" onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label">Select Model Adaptation</label>
            <select 
              className="form-select" 
              onChange={(e) => setSelectedAdaptation(e.target.value)}
              value={selectedAdaptation}
              required
            >
              <option value="">Select Adaptation</option>
              <option value="TCA">TCA</option>
              <option value="CORAL">CORAL</option>
              <option value="MMD">MMD</option>
              <option value="HISSN">HISSN</option>
              <option value="TCA_CORAL">TCA + CORAL</option>
            </select>
          </div>
          <div className="mb-3">
            <label className="form-label">Select ML Model</label>
            <select 
              className="form-select" 
              onChange={(e) => setSelectedMLModel(e.target.value)}
              value={selectedMLModel}
              required
            >
              <option value="">Select Model</option>
              <option value="RandomForest">Random Forest</option>
              <option value="LogisticRegression">Logistic Regression</option>
              <option value="XGBoost">XGBoost</option>
            </select>
          </div>
          <div className="mb-3">
            <label className="form-label">Upload Target File (.csv)</label>
            <input 
              type="file" 
              className="form-control" 
              accept=".csv" 
              onChange={(e) => setTargetFile(e.target.files[0])}
              required 
            />
            {targetFile && (
              <small className="text-muted">
                File size: {(targetFile.size / 1024 / 1024).toFixed(2)} MB
              </small>
            )}
          </div>
          <div className="mb-3">
            <label className="form-label">Upload Test File (.csv)</label>
            <input 
              type="file" 
              className="form-control" 
              accept=".csv" 
              onChange={(e) => setTestFile(e.target.files[0])}
              required 
            />
            {testFile && (
              <small className="text-muted">
                File size: {(testFile.size / 1024 / 1024).toFixed(2)} MB
              </small>
            )}
          </div>
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Processing...
              </>
            ) : (
              'Submit'
            )}
          </button>
        </form>
        {formatResults(results)}
      </div>
    </div>
  );
}
