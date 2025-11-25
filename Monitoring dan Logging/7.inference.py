"""
inference.py - Send predictions to deployed model
Used for testing and generating monitoring data
"""

import requests
import json
import time
import numpy as np
import argparse
from datetime import datetime


class HeartDiseaseInference:
    """Client for making predictions against deployed model"""
    
    def __init__(self, mlserver_url="http://localhost:8080"):
        self.mlserver_url = mlserver_url
        self.model_name = "heart-disease"
        
    def check_health(self):
        """Check if model server is healthy"""
        try:
            response = requests.get(f"{self.mlserver_url}/v2/health/ready", timeout=5)
            if response.status_code == 200:
                print("✓ Model server is healthy and ready")
                return True
            else:
                print(f"✗ Model server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to model server: {e}")
            return False
    
    def generate_sample_data(self, n_samples=1):
        """
        Generate sample heart disease data for testing
        Using Indonesian column names as expected by the model
        """
        samples = []
        for _ in range(n_samples):
            sample = {
                'usia': float(np.random.randint(30, 80)),
                'jenis_kelamin': float(np.random.choice([0, 1])),
                'tipe_nyeri_dada': float(np.random.choice([0, 1, 2, 3])),
                'tekanan_darah_istirahat': float(np.random.randint(90, 200)),
                'kolesterol': float(np.random.randint(150, 400)),
                'gula_darah_puasa': float(np.random.choice([0, 1])),
                'hasil_ecg_istirahat': float(np.random.choice([0, 1, 2])),
                'thalch': float(np.random.randint(80, 200)),
                'angina_olahraga': float(np.random.choice([0, 1])),
                'depresi_st': float(round(np.random.uniform(0, 6), 1)),
                'kemiringan_st': float(np.random.choice([0, 1, 2])),
                'jumlah_pembuluh_darah': float(np.random.choice([0, 1, 2, 3])),
                'thalassemia': float(np.random.choice([0, 1, 2, 3]))
            }
            samples.append(sample)
        return samples
    
    def predict(self, data):
        """
        Make prediction using MLServer v2 inference protocol
        Model expects named columns in Indonesian
        
        Args:
            data: List of dictionaries with feature values
        """
        # Column names in exact order as model expects
        column_names = [
            'usia', 'jenis_kelamin', 'tipe_nyeri_dada', 'tekanan_darah_istirahat',
            'kolesterol', 'gula_darah_puasa', 'hasil_ecg_istirahat', 'thalch',
            'angina_olahraga', 'depresi_st', 'kemiringan_st', 'jumlah_pembuluh_darah',
            'thalassemia'
        ]
        
        # Build inputs with named columns
        inputs = []
        for col_name in column_names:
            column_data = [sample[col_name] for sample in data]
            inputs.append({
                "name": col_name,
                "shape": [len(data), 1],
                "datatype": "FP64",
                "data": column_data
            })
        
        payload = {"inputs": inputs}
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.mlserver_url}/v2/models/{self.model_name}/infer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['outputs'][0]['data']
                
                return {
                    'success': True,
                    'predictions': predictions,
                    'latency': latency,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'latency': latency
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency': None
            }
    
    def run_load_test(self, n_requests=100, delay=0.1):
        """
        Run load test to generate monitoring data
        
        Args:
            n_requests: Number of prediction requests to make
            delay: Delay between requests in seconds
        """
        print(f"\n{'='*60}")
        print(f"Running Load Test")
        print(f"{'='*60}")
        print(f"Requests: {n_requests}")
        print(f"Delay: {delay}s between requests")
        print(f"{'='*60}\n")
        
        results = {
            'success': 0,
            'failed': 0,
            'total_latency': 0,
            'predictions': {'disease': 0, 'no_disease': 0}
        }
        
        for i in range(n_requests):
            # Generate random sample
            sample_data = self.generate_sample_data(n_samples=1)
            
            # Make prediction
            result = self.predict(sample_data)
            
            if result['success']:
                results['success'] += 1
                results['total_latency'] += result['latency']
                
                # Parse prediction (assuming 0=no disease, 1=disease)
                pred = result['predictions'][0]
                if pred == 1:
                    results['predictions']['disease'] += 1
                else:
                    results['predictions']['no_disease'] += 1
                
                print(f"Request {i+1}/{n_requests}: Success | "
                      f"Prediction: {'Disease' if pred == 1 else 'No Disease'} | "
                      f"Latency: {result['latency']:.3f}s")
            else:
                results['failed'] += 1
                print(f"Request {i+1}/{n_requests}: Failed | Error: {result['error']}")
            
            time.sleep(delay)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Load Test Summary")
        print(f"{'='*60}")
        print(f"Total Requests: {n_requests}")
        print(f"Successful: {results['success']} ({results['success']/n_requests*100:.1f}%)")
        print(f"Failed: {results['failed']} ({results['failed']/n_requests*100:.1f}%)")
        
        if results['success'] > 0:
            avg_latency = results['total_latency'] / results['success']
            print(f"Average Latency: {avg_latency:.3f}s")
            print(f"\nPredictions:")
            print(f"  Disease: {results['predictions']['disease']}")
            print(f"  No Disease: {results['predictions']['no_disease']}")
        
        print(f"{'='*60}\n")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Heart Disease Model Inference Client')
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8080',
        help='MLServer URL (default: http://localhost:8080)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'load-test'],
        default='single',
        help='Inference mode (default: single)'
    )
    parser.add_argument(
        '--requests',
        type=int,
        default=100,
        help='Number of requests for load test (default: 100)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between requests in seconds (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = HeartDiseaseInference(mlserver_url=args.url)
    
    # Check health
    print(f"\n{'='*60}")
    print("Heart Disease Model Inference Client")
    print(f"{'='*60}")
    print(f"MLServer URL: {args.url}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    if not client.check_health():
        print("\n✗ Model server is not available. Exiting.")
        return
    
    if args.mode == 'single':
        # Single prediction example
        print("\n--- Single Prediction Example ---\n")
        
        sample_data = client.generate_sample_data(n_samples=1)
        print("Input features:")
        for key, value in sample_data[0].items():
            print(f"  {key}: {value}")
        
        print("\nMaking prediction...")
        result = client.predict(sample_data)
        
        if result['success']:
            print(f"\n✓ Prediction successful!")
            print(f"  Result: {'Disease' if result['predictions'][0] == 1 else 'No Disease'}")
            print(f"  Latency: {result['latency']:.3f}s")
            print(f"  Timestamp: {result['timestamp']}")
        else:
            print(f"\n✗ Prediction failed: {result['error']}")
    
    elif args.mode == 'load-test':
        # Load test
        client.run_load_test(n_requests=args.requests, delay=args.delay)
    
    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()