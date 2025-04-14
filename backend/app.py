from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from werkzeug.utils import secure_filename
from models.tca_model import run_tca
from models.coral_model import run_coral
from models.mmd_model import run_mmd
from models.hissn_model import run_hissn
from models.tca_coral_model import run_tca_coral

app = Flask(__name__)
CORS(app)

# Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Define Upload Folder (using absolute path)
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define Dataset Folder (using absolute path)
DATASET_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))

# Ensure Upload Directory Exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Ensure both files are provided
        if 'targetFile' not in request.files or 'testFile' not in request.files:
            return jsonify({"error": "Target and test files are required"}), 400

        target_file = request.files['targetFile']
        test_file = request.files['testFile']
        model_adaptation = request.form.get('modelAdaptation')
        ml_model = request.form.get('mlModel')

        if not model_adaptation or not ml_model:
            return jsonify({"error": "Model adaptation and ML model are required"}), 400

        # Ensure files are valid
        if target_file.filename == '' or test_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not (target_file.filename.endswith('.csv') and test_file.filename.endswith('.csv')):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        # Check file sizes
        target_file.seek(0, 2)  # Seek to end
        test_file.seek(0, 2)
        if target_file.tell() > MAX_CONTENT_LENGTH or test_file.tell() > MAX_CONTENT_LENGTH:
            return jsonify({"error": "File size exceeds 16MB limit"}), 400
        target_file.seek(0)  # Reset file pointer
        test_file.seek(0)

        # Secure filenames and save
        target_filename = secure_filename(target_file.filename)
        test_filename = secure_filename(test_file.filename)

        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)

        target_file.save(target_path)
        test_file.save(test_path)

        print(f"Uploaded: {target_filename}, {test_filename}")
        print(f"Selected Model Adaptation: {model_adaptation}, ML Model: {ml_model}")

        # Define Source Datasets
        source_files = ['EQ.csv', 'PDE.csv', 'LC.csv', 'ML.csv', 'JDT.csv']
        data_path = DATASET_FOLDER

        # Run selected model
        model_mapping = {
            "TCA": run_tca,
            "CORAL": run_coral,
            "MMD": run_mmd,
            "HISSN": run_hissn,
            "TCA_CORAL": run_tca_coral
        }

        if model_adaptation not in model_mapping:
            return jsonify({"error": "Unsupported model adaptation technique"}), 400

        results = model_mapping[model_adaptation](source_files, target_filename, test_filename, data_path, ml_model)
        return jsonify(results)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
