from flask import Blueprint

upload_bp = Blueprint("upload", __name__)
inference_bp = Blueprint("inference", __name__)

from . import upload, inference
