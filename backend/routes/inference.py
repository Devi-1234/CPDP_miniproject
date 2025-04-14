from flask import Blueprint, request, jsonify
import pandas as pd
from models.tca_model import run_tca
#from models.dtb import run_dtb
from models.hissn_model import run_hissn
from models.coral_model import run_coral
from models.mmd_model import run_mmd

inference_bp = Blueprint("inference", __name__)

@inference_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        X_source = pd.DataFrame(data["X_source"])
        y_source = data["y_source"]
        X_target = pd.DataFrame(data["X_target"])
        X_test = pd.DataFrame(data["X_test"])
        y_test = data["y_test"]
        model_adaptation = data["model_adaptation"]
        ml_model = data["ml_model"]

        if model_adaptation == "TCA":
            results = run_tca(X_source, y_source, X_target, X_test, y_test, ml_model)
        # elif model_adaptation == "DTB":
        #     results = run_dtb(X_source, y_source, X_target, X_test, y_test, ml_model)
        elif model_adaptation == "HISSN":
            results = run_hissn(X_source, y_source, X_target, X_test, y_test, ml_model)
        elif model_adaptation == "CORAL":
            results = run_coral(X_source, y_source, X_target, X_test, y_test, ml_model)
        elif model_adaptation == "MMD":
            results = run_mmd(X_source, y_source, X_target, X_test, y_test, ml_model)
        else:
            return jsonify({"error": "Invalid model adaptation selected"}), 400

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
