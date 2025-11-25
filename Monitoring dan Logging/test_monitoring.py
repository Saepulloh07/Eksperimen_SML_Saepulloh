"""
test_monitoring.py - Test script untuk verify monitoring stack
"""

import requests
import time
import sys


def test_service(name, url, expected_status=200):
    """Test if a service is accessible"""
    try:
        print(f"Testing {name}...", end=" ")
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            print("✓ OK")
            return True
        else:
            print(f"✗ FAILED (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ FAILED ({str(e)})")
        return False


def test_prometheus_metrics():
    """Test if Prometheus metrics are being collected"""
    print("\nTesting Prometheus Metrics...")
    
    try:
        # Get metrics from exporter
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        metrics = response.text
        
        # Check for key metrics
        required_metrics = [
            "heart_disease_model_health",
            "heart_disease_predictions_total",
            "heart_disease_prediction_latency_seconds",
            "heart_disease_requests_per_minute",
            "heart_disease_prediction_confidence"
        ]
        
        all_found = True
        for metric in required_metrics:
            if metric in metrics:
                print(f"  ✓ {metric}")
            else:
                print(f"  ✗ {metric} NOT FOUND")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"  ✗ Error checking metrics: {e}")
        return False


def test_grafana_datasource():
    """Test if Grafana can connect to Prometheus"""
    print("\nTesting Grafana Datasource...")
    
    try:
        # Login to Grafana
        login_data = {
            "user": "admin",
            "password": "admin123"
        }
        session = requests.Session()
        response = session.post(
            "http://localhost:3000/login",
            json=login_data,
            timeout=5
        )
        
        # Get datasources
        response = session.get(
            "http://localhost:3000/api/datasources",
            timeout=5
        )
        
        if response.status_code == 200:
            datasources = response.json()
            for ds in datasources:
                if ds.get('type') == 'prometheus':
                    print(f"  ✓ Prometheus datasource found: {ds.get('name')}")
                    return True
            
            print("  ✗ Prometheus datasource not found")
            return False
        else:
            print(f"  ✗ Failed to get datasources (status: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"  ✗ Error checking Grafana: {e}")
        return False


def main():
    """Main test function"""
    print("="*60)
    print("Heart Disease Model - Monitoring Stack Test")
    print("="*60)
    print()
    
    tests = []
    
    # Test basic services
    print("Testing Services:")
    print("-"*60)
    tests.append(test_service(
        "MLServer Health", 
        "http://localhost:8080/v2/health/ready"
    ))
    tests.append(test_service(
        "Prometheus", 
        "http://localhost:9090/-/healthy"
    ))
    tests.append(test_service(
        "Grafana", 
        "http://localhost:3000/api/health"
    ))
    tests.append(test_service(
        "Prometheus Exporter", 
        "http://localhost:8000/metrics"
    ))
    tests.append(test_service(
        "Node Exporter", 
        "http://localhost:9100/metrics"
    ))
    
    # Test Prometheus metrics
    print()
    tests.append(test_prometheus_metrics())
    
    # Test Grafana datasource
    print()
    tests.append(test_grafana_datasource())
    
    # Test model inference
    print("\nTesting Model Inference...")
    try:
        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 13],
                    "datatype": "FP32",
                    "data": [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
                }
            ]
        }
        response = requests.post(
            "http://localhost:8080/v2/models/heart-disease/infer",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['outputs'][0]['data'][0]
            print(f"  ✓ Inference successful (prediction: {prediction})")
            tests.append(True)
        else:
            print(f"  ✗ Inference failed (status: {response.status_code})")
            tests.append(False)
            
    except Exception as e:
        print(f"  ✗ Inference error: {e}")
        tests.append(False)
    
    # Summary
    print()
    print("="*60)
    passed = sum(tests)
    total = len(tests)
    print(f"Test Results: {passed}/{total} passed")
    print("="*60)
    
    if passed == total:
        print("✓ All tests passed! Monitoring stack is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()