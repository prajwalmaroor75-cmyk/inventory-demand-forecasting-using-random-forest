from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load model components
try:
    model = joblib.load("backend/model.pkl")
    scaler = joblib.load("backend/scaler.pkl")
    max_demand = joblib.load("backend/max_demand.pkl")
    valid_ids = joblib.load("backend/valid_ids.pkl")
    r2_score_val = joblib.load("backend/r2_score.pkl")  # ✅ added line
    print("✅ All model components loaded successfully.")
except Exception as e:
    print("❌ Error loading model components:", e)
    model = scaler = max_demand = valid_ids = r2_score_val = None

# Frontend home route
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Endpoint to return model R² score
@app.route("/accuracy", methods=["GET"])
def accuracy():
    if r2_score_val is None:
        return jsonify({"error": "⚠️ Model accuracy not available."}), 500
    return jsonify({"r2_score": round(r2_score_val, 2)})

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if None in [model, scaler, max_demand, valid_ids]:
        return jsonify({"error": "⚠️ Server not ready. Missing model components."}), 500

    try:
        data = request.json
        df = pd.DataFrame([data])

        required = ["store_id", "sku_id", "total_price", "base_price", "is_featured_sku", "is_display_sku"]

        # Validate required fields
        for col in required:
            if col not in df or not pd.api.types.is_numeric_dtype(df[col]):
                return jsonify({"error": f"Invalid or missing value for: {col}"}), 400

        # Validate store_id and sku_id
        if df["store_id"].iloc[0] not in valid_ids["store_ids"]:
            return jsonify({"error": "❌ Invalid Store ID."}), 400
        if df["sku_id"].iloc[0] not in valid_ids["sku_ids"]:
            return jsonify({"error": "❌ Invalid SKU ID."}), 400

        # Validate binary fields
        if df["is_featured_sku"].iloc[0] not in [0, 1]:
            return jsonify({"error": "⚠️ 'Featured SKU' must be 0 or 1."}), 400
        if df["is_display_sku"].iloc[0] not in [0, 1]:
            return jsonify({"error": "⚠️ 'Display SKU' must be 0 or 1."}), 400

        # Feature engineering: derive price_per_unit
        df["price_per_unit"] = df["total_price"] / (df["base_price"] + 1)  # avoid division by zero

        features = required + ["price_per_unit"]
        scaled_input = scaler.transform(df[features])

        # Predict units sold
        predicted_units = model.predict(scaled_input)[0]
        predicted_percentage = (predicted_units / max_demand) * 100

        return jsonify({
            "predicted_units": round(predicted_units, 2),
            "predicted_demand_percentage": round(predicted_percentage, 2),
            "is_demand_high": "Yes" if predicted_percentage > 50 else "No"
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": "⚠️ Prediction failed. Check your input values."}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
