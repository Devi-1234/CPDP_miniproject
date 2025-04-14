from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return jsonify({"message": "File uploaded successfully", "filename": filename}), 200
    
    return jsonify({"error": "Invalid file type. Only CSV allowed"}), 400
