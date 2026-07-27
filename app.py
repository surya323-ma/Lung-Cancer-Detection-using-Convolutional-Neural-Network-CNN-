import os
import uuid
import json
import datetime as dt
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from model.predict import predict, is_real_model_loaded

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
HISTORY_FILE = os.path.join(BASE_DIR, "data", "history.json")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-200:], f, indent=2)  # keep last 200 scans


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/system-status")
def system_status():
    return jsonify({
        "model_mode": "trained_model" if is_real_model_loaded() else "demo_heuristic",
        "server_time": dt.datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under field 'image'."}), 400

    file = request.files["image"]
    patient_ref = request.form.get("patient_ref", "").strip() or "Unlabeled sample"

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, BMP or TIFF."}), 400

    scan_id = uuid.uuid4().hex[:10]
    filename = secure_filename(f"{scan_id}_{file.filename}")
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        result = predict(save_path)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    record = {
        "scan_id": scan_id,
        "patient_ref": patient_ref,
        "filename": filename,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        **result,
    }

    history = load_history()
    history.append(record)
    save_history(history)

    return jsonify(record)


@app.route("/api/history")
def api_history():
    history = load_history()
    limit = int(request.args.get("limit", 25))
    return jsonify(list(reversed(history))[:limit])


@app.route("/api/stats")
def api_stats():
    history = load_history()
    total = len(history)
    benign = sum(1 for h in history if h.get("predicted_class") == "lung_benign_tissue")
    adeno = sum(1 for h in history if h.get("predicted_class") == "lung_adenocarcinoma")
    squamous = sum(1 for h in history if h.get("predicted_class") == "lung_squamous_cell_carcinoma")
    avg_conf = round(sum(h.get("confidence", 0) for h in history) / total, 2) if total else 0
    high_risk = sum(1 for h in history if h.get("malignant_probability", 0) >= 60)

    trend = {}
    for h in history:
        day = h["timestamp"][:10]
        trend[day] = trend.get(day, 0) + 1
    trend_sorted = sorted(trend.items())[-14:]

    return jsonify({
        "total_scans": total,
        "benign_count": benign,
        "adenocarcinoma_count": adeno,
        "squamous_count": squamous,
        "average_confidence": avg_conf,
        "high_risk_flags": high_risk,
        "scans_per_day": [{"date": d, "count": c} for d, c in trend_sorted],
        "model_mode": "trained_model" if is_real_model_loaded() else "demo_heuristic",
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
