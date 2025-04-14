import React from "react";
import Navbar from "../components/Navbar";

export default function Home() {
  return (
    <div>
      <Navbar />
      <div className="container mt-5">
        <h1 className="text-center">Cross-Project Defect Prediction (CPDP)</h1>
        <div className="card mt-4">
          <div className="card-body">
            <h2 className="card-title">What is CPDP?</h2>
            <p className="card-text">
              Cross-Project Defect Prediction (CPDP) is a machine learning-based approach to predict software defects in a target project using knowledge transferred from other source projects.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}