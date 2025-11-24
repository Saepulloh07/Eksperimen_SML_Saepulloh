import os
import joblib
import pickle
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
import time

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total', 
    'Total number of predictions',
    ['model_version', 'prediction_class']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Time spent processing prediction'
)

ERROR_COUNTER = Counter(
    'model_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Load model and preprocessor
print("🚀 Loading model and preprocessor...")
try:
    model = joblib.load('../Membangun_model/wine_quality_model.pkl')
    scaler = joblib.load('../preprocessing/data/output/scaler.pkl')
    
    with open('../preprocessing/data/output/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"✅ Model loaded successfully")
    print(f"✅ Expected features: {len(feature_names)}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Model info
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0')
MODEL_NAME = "wine_quality_classifier"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME,
        'version': MODEL_VERSION,
        'timestamp': time.time()
    }), 200


@app.route('/info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({
        'model_name': MODEL_NAME,
        'version': MODEL_VERSION,
        'features': feature_names,
        'n_features': len(feature_names),
        'model_type': 'RandomForestClassifier'
    }), 200


@app.route('/predict', methods=['POST'])
@PREDICTION_LATENCY.time()
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()
        
        if 'features' not in data:
            ERROR_COUNTER.labels(error_type='missing_features').inc()
            return jsonify({
                'error': 'Missing features in request',
                'required_features': feature_names
            }), 400
        
        # Extract features
        features = data['features']
        
        # Validate feature count
        if len(features) != len(feature_names):
            ERROR_COUNTER.labels(error_type='invalid_feature_count').inc()
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(features)}',
                'required_features': feature_names
            }), 400
        
        # Prepare input
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][prediction])
        
        # Update metrics
        PREDICTION_COUNTER.labels(
            model_version=MODEL_VERSION,
            prediction_class=prediction
        ).inc()
        
        # Prepare response
        response = {
            'prediction': prediction,
            'prediction_label': 'Good Wine' if prediction == 1 else 'Bad Wine',
            'probability': probability,
            'model_version': MODEL_VERSION,
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type='prediction_error').inc()
        return jsonify({
            'error': str(e),
            'model_version': MODEL_VERSION
        }), 500


@app.route('/predict/batch', methods=['POST'])
@PREDICTION_LATENCY.time()
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if 'features' not in data:
            ERROR_COUNTER.labels(error_type='missing_features').inc()
            return jsonify({
                'error': 'Missing features in request'
            }), 400
        
        # Extract features
        features_list = data['features']
        
        if not isinstance(features_list, list):
            ERROR_COUNTER.labels(error_type='invalid_input_format').inc()
            return jsonify({
                'error': 'Features must be a list of feature arrays'
            }), 400
        
        # Prepare input
        X = np.array(features_list)
        
        # Validate shape
        if X.shape[1] != len(feature_names):
            ERROR_COUNTER.labels(error_type='invalid_feature_count').inc()
            return jsonify({
                'error': f'Expected {len(feature_names)} features per sample',
                'required_features': feature_names
            }), 400
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled).tolist()
        probabilities = model.predict_proba(X_scaled).tolist()
        
        # Update metrics
        for pred in predictions:
            PREDICTION_COUNTER.labels(
                model_version=MODEL_VERSION,
                prediction_class=pred
            ).inc()
        
        # Prepare response
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                'sample_id': i,
                'prediction': pred,
                'prediction_label': 'Good Wine' if pred == 1 else 'Bad Wine',
                'probability': proba[pred]
            })
        
        response = {
            'predictions': results,
            'count': len(results),
            'model_version': MODEL_VERSION,
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type='batch_prediction_error').inc()
        return jsonify({
            'error': str(e),
            'model_version': MODEL_VERSION
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"🚀 Starting {MODEL_NAME} v{MODEL_VERSION}")
    print(f"{'='*60}")
    print(f"📍 Server running on port {port}")
    print(f"📊 Health check: http://localhost:{port}/health")
    print(f"📊 Model info: http://localhost:{port}/info")
    print(f"🔮 Prediction: http://localhost:{port}/predict")
    print(f"📈 Metrics: http://localhost:{port}/metrics")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
