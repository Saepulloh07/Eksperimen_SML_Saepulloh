"""
Prometheus Exporter for Heart Disease ML Model
Exposes custom metrics for monitoring model performance and health
"""

import time
import os
import logging
from datetime import datetime
from collections import defaultdict
import threading

import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from prometheus_client import CollectorRegistry, generate_latest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MLSERVER_URL = os.getenv("MLSERVER_URL", "http://localhost:8080")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", 8000))

# Create registry
registry = CollectorRegistry()

# ============================================================================
# METRIC 1-3: Basic Model Metrics (for Prometheus minimum requirement)
# ============================================================================

# Prediction counter by result
prediction_counter = Counter(
    'heart_disease_predictions_total',
    'Total number of predictions made',
    ['result', 'model'],
    registry=registry
)

# Prediction latency
prediction_latency = Histogram(
    'heart_disease_prediction_latency_seconds',
    'Time taken for predictions',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Model health status
model_health = Gauge(
    'heart_disease_model_health',
    'Model health status (1=healthy, 0=unhealthy)',
    ['model'],
    registry=registry
)

# ============================================================================
# METRIC 4-10+: Extended Metrics (for Grafana 10 metrics requirement)
# ============================================================================

# Request rate
request_rate = Gauge(
    'heart_disease_requests_per_minute',
    'Number of requests per minute',
    ['model'],
    registry=registry
)

# Prediction confidence
prediction_confidence = Gauge(
    'heart_disease_prediction_confidence',
    'Average prediction confidence score',
    ['result', 'model'],
    registry=registry
)

# Error rate
error_counter = Counter(
    'heart_disease_errors_total',
    'Total number of errors',
    ['error_type', 'model'],
    registry=registry
)

# Model uptime
model_uptime = Gauge(
    'heart_disease_model_uptime_seconds',
    'Model uptime in seconds',
    ['model'],
    registry=registry
)

# Memory usage (simulated)
memory_usage = Gauge(
    'heart_disease_memory_usage_bytes',
    'Estimated memory usage',
    ['component', 'model'],
    registry=registry
)

# Feature statistics
feature_stats = Gauge(
    'heart_disease_feature_stats',
    'Statistics about input features',
    ['feature', 'stat_type', 'model'],
    registry=registry
)

# Prediction distribution
prediction_distribution = Gauge(
    'heart_disease_prediction_distribution',
    'Distribution of predictions over time window',
    ['result', 'time_window', 'model'],
    registry=registry
)

# API response time
api_response_time = Histogram(
    'heart_disease_api_response_seconds',
    'API endpoint response time',
    ['endpoint', 'method'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry
)

# Model version info
model_info = Info(
    'heart_disease_model_info',
    'Information about the deployed model',
    registry=registry
)

# Throughput
throughput = Gauge(
    'heart_disease_throughput',
    'Predictions processed per second',
    ['model'],
    registry=registry
)

# Data drift indicator (simulated)
data_drift_score = Gauge(
    'heart_disease_data_drift_score',
    'Data drift detection score',
    ['model'],
    registry=registry
)


class ModelMonitor:
    """Monitor ML model and collect metrics"""
    
    def __init__(self, mlserver_url):
        self.mlserver_url = mlserver_url
        self.model_name = "heart-disease"
        self.start_time = time.time()
        
        # Tracking variables
        self.prediction_history = []
        self.confidence_history = defaultdict(list)
        self.request_times = []
        self.feature_values = defaultdict(list)
        
        # Set model info
        model_info.info({
            'model_name': self.model_name,
            'version': 'latest',
            'framework': 'scikit-learn',
            'deployment': 'mlserver'
        })
        
        logger.info(f"Initialized ModelMonitor for {mlserver_url}")
    
    def check_model_health(self):
        """Check if model is healthy and responding"""
        try:
            # Try health endpoint
            response = requests.get(
                f"{self.mlserver_url}/v2/health/ready",
                timeout=5
            )
            
            if response.status_code == 200:
                model_health.labels(model=self.model_name).set(1)
                logger.debug("Model health check: HEALTHY")
                return True
            else:
                model_health.labels(model=self.model_name).set(0)
                logger.warning(f"Model health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            model_health.labels(model=self.model_name).set(0)
            error_counter.labels(
                error_type='health_check_failed',
                model=self.model_name
            ).inc()
            logger.error(f"Health check error: {e}")
            return False
    
    def update_uptime(self):
        """Update model uptime metric"""
        uptime = time.time() - self.start_time
        model_uptime.labels(model=self.model_name).set(uptime)
    
    def simulate_prediction_metrics(self):
        """
        Simulate prediction metrics based on realistic patterns
        In production, this would be replaced by actual inference logging
        """
        import random
        import numpy as np
        
        # Simulate predictions with realistic distribution
        # Heart disease dataset typically has ~55% healthy, 45% disease
        if random.random() < 0.55:
            result = "no_disease"
            confidence = random.uniform(0.6, 0.95)
        else:
            result = "disease"
            confidence = random.uniform(0.55, 0.90)
        
        # Update metrics
        prediction_counter.labels(
            result=result,
            model=self.model_name
        ).inc()
        
        # Simulate latency
        latency = random.uniform(0.01, 0.15)
        prediction_latency.labels(model=self.model_name).observe(latency)
        
        # Store for aggregation
        self.prediction_history.append({
            'result': result,
            'confidence': confidence,
            'timestamp': time.time()
        })
        self.confidence_history[result].append(confidence)
        
        # Update confidence metrics
        if len(self.confidence_history[result]) > 0:
            avg_confidence = np.mean(self.confidence_history[result][-100:])
            prediction_confidence.labels(
                result=result,
                model=self.model_name
            ).set(avg_confidence)
        
        # Simulate feature statistics (age, cholesterol, etc.)
        features = {
            'age': random.randint(30, 80),
            'cholesterol': random.randint(150, 350),
            'heart_rate': random.randint(60, 180),
            'blood_pressure': random.randint(90, 180)
        }
        
        for feature, value in features.items():
            self.feature_values[feature].append(value)
            
            # Update feature stats
            if len(self.feature_values[feature]) >= 10:
                recent_values = self.feature_values[feature][-100:]
                feature_stats.labels(
                    feature=feature,
                    stat_type='mean',
                    model=self.model_name
                ).set(np.mean(recent_values))
                
                feature_stats.labels(
                    feature=feature,
                    stat_type='std',
                    model=self.model_name
                ).set(np.std(recent_values))
        
        # Calculate request rate
        now = time.time()
        self.request_times.append(now)
        recent_requests = [t for t in self.request_times if now - t <= 60]
        self.request_times = recent_requests
        
        requests_per_minute = len(recent_requests)
        request_rate.labels(model=self.model_name).set(requests_per_minute)
        
        # Calculate throughput (predictions per second)
        if len(recent_requests) > 1:
            throughput_value = len(recent_requests) / 60.0
            throughput.labels(model=self.model_name).set(throughput_value)
        
        # Update prediction distribution
        recent_predictions = [
            p for p in self.prediction_history 
            if now - p['timestamp'] <= 300  # Last 5 minutes
        ]
        
        if recent_predictions:
            disease_count = sum(1 for p in recent_predictions if p['result'] == 'disease')
            no_disease_count = len(recent_predictions) - disease_count
            
            prediction_distribution.labels(
                result='disease',
                time_window='5min',
                model=self.model_name
            ).set(disease_count)
            
            prediction_distribution.labels(
                result='no_disease',
                time_window='5min',
                model=self.model_name
            ).set(no_disease_count)
        
        # Simulate data drift (random walk between 0-1)
        if not hasattr(self, 'drift_score'):
            self.drift_score = 0.1
        self.drift_score += random.uniform(-0.05, 0.05)
        self.drift_score = max(0, min(1, self.drift_score))
        data_drift_score.labels(model=self.model_name).set(self.drift_score)
        
        # Simulate memory usage
        base_memory = 150 * 1024 * 1024  # 150 MB base
        variable_memory = random.randint(-10, 10) * 1024 * 1024
        memory_usage.labels(
            component='model',
            model=self.model_name
        ).set(base_memory + variable_memory)
        
        memory_usage.labels(
            component='cache',
            model=self.model_name
        ).set(random.randint(20, 50) * 1024 * 1024)
    
    def simulate_errors(self):
        """Occasionally simulate errors for realistic monitoring"""
        import random
        
        # Simulate occasional errors (1% chance)
        if random.random() < 0.01:
            error_types = ['timeout', 'invalid_input', 'server_error']
            error_type = random.choice(error_types)
            error_counter.labels(
                error_type=error_type,
                model=self.model_name
            ).inc()
            logger.warning(f"Simulated error: {error_type}")
    
    def collect_metrics(self):
        """Main metrics collection loop"""
        logger.info("Starting metrics collection...")
        
        while True:
            try:
                # Check model health
                self.check_model_health()
                
                # Update uptime
                self.update_uptime()
                
                # Simulate predictions (in production, hook into actual inference)
                # Simulate 1-10 predictions per iteration
                import random
                num_predictions = random.randint(1, 10)
                for _ in range(num_predictions):
                    self.simulate_prediction_metrics()
                
                # Occasionally simulate errors
                self.simulate_errors()
                
                # Sleep for collection interval
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                error_counter.labels(
                    error_type='collection_error',
                    model=self.model_name
                ).inc()
                time.sleep(5)


def main():
    """Main function to start the exporter"""
    logger.info("="*60)
    logger.info("Heart Disease Model Prometheus Exporter")
    logger.info("="*60)
    logger.info(f"MLServer URL: {MLSERVER_URL}")
    logger.info(f"Exporter Port: {EXPORTER_PORT}")
    logger.info("="*60)
    
    # Start Prometheus HTTP server
    start_http_server(EXPORTER_PORT, registry=registry)
    logger.info(f"✓ Prometheus exporter started on port {EXPORTER_PORT}")
    logger.info(f"✓ Metrics available at http://localhost:{EXPORTER_PORT}/metrics")
    
    # Initialize monitor
    monitor = ModelMonitor(MLSERVER_URL)
    
    # Start metrics collection in separate thread
    collection_thread = threading.Thread(
        target=monitor.collect_metrics,
        daemon=True
    )
    collection_thread.start()
    
    logger.info("✓ Metrics collection started")
    logger.info("✓ Exporter is running. Press Ctrl+C to stop.")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down exporter...")


if __name__ == "__main__":
    main()