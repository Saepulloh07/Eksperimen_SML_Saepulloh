"""
capture_screenshots.py - Helper script to capture monitoring screenshots
Note: This is a reference guide. Actual screenshots should be taken manually
      using browser screenshot tools or automated tools like Selenium.
"""

import os
import time
from datetime import datetime


class ScreenshotGuide:
    """Guide for capturing monitoring screenshots"""
    
    def __init__(self):
        self.base_dir = "Monitoring dan Logging"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def print_header(self, title):
        print("\n" + "="*70)
        print(f" {title}")
        print("="*70)
    
    def print_step(self, number, description):
        print(f"\n[{number}] {description}")
    
    def print_url(self, url):
        print(f"    URL: {url}")
    
    def print_save_location(self, location):
        print(f"    Save to: {location}")
    
    def guide_serving_screenshots(self):
        """Guide for taking serving screenshots"""
        self.print_header("1. BUKTI SERVING MODEL")
        
        self.print_step(1, "Health Check Screenshot")
        self.print_url("http://localhost:8080/v2/health/ready")
        print("    Or use: curl http://localhost:8080/v2/health/ready")
        self.print_save_location("1.bukti_serving/1.health_check.png")
        
        self.print_step(2, "Inference Test Screenshot")
        print("    Run: python 7.inference.py --mode single")
        print("    Take screenshot of the output showing:")
        print("    - Input features")
        print("    - Prediction result")
        print("    - Latency")
        self.print_save_location("1.bukti_serving/2.inference_test.png")
        
        print("\n" + "-"*70)
        input("Press Enter when done...")
    
    def guide_prometheus_screenshots(self):
        """Guide for taking Prometheus screenshots"""
        self.print_header("2. MONITORING PROMETHEUS (Minimum 3)")
        
        self.print_url("http://localhost:9090")
        
        queries = [
            ("Health Status", "heart_disease_model_health", "1.monitoring_health_status.png"),
            ("Prediction Rate", "rate(heart_disease_predictions_total[5m])", "2.monitoring_prediction_rate.png"),
            ("Prediction Latency", "histogram_quantile(0.95, rate(heart_disease_prediction_latency_seconds_bucket[5m]))", "3.monitoring_latency.png"),
        ]
        
        for i, (name, query, filename) in enumerate(queries, 1):
            self.print_step(i, f"{name}")
            print(f"    Query: {query}")
            print("    Steps:")
            print("    1. Go to Graph tab in Prometheus")
            print("    2. Enter the query above")
            print("    3. Click 'Execute'")
            print("    4. Switch to 'Graph' view")
            print("    5. Take screenshot")
            self.print_save_location(f"4.bukti monitoring Prometheus/{filename}")
            print()
        
        print("\n" + "-"*70)
        input("Press Enter when done...")
    
    def guide_grafana_screenshots(self):
        """Guide for taking Grafana screenshots"""
        self.print_header("3. MONITORING GRAFANA (Minimum 10)")
        
        self.print_url("http://localhost:3000")
        print("    Login: admin / admin123")
        print("    Dashboard: Heart Disease Model Monitoring")
        
        panels = [
            "Dashboard Overview (full page)",
            "Model Health Status panel",
            "Total Predictions panel",
            "Request Rate gauge",
            "Model Uptime panel",
            "Prediction Rate Over Time graph",
            "Prediction Latency graph",
            "Prediction Distribution pie chart",
            "Prediction Confidence graph",
            "Error Rate graph",
            "Memory Usage graph",
            "Data Drift Score graph",
            "Throughput stat panel",
        ]
        
        for i, panel in enumerate(panels, 1):
            self.print_step(i, panel)
            filename = f"{i}.monitoring_{panel.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            self.print_save_location(f"5.bukti monitoring Grafana/{filename}")
        
        print("\n    Tips:")
        print("    - Click on panel title to expand")
        print("    - Use 'Last 1 hour' or 'Last 30 minutes' time range")
        print("    - Make sure data is being generated (run load test)")
        
        print("\n" + "-"*70)
        input("Press Enter when done...")
    
    def guide_alerting_screenshots(self):
        """Guide for taking alerting screenshots"""
        self.print_header("4. ALERTING GRAFANA (3 Alerts = 6 Screenshots)")
        
        self.print_url("http://localhost:3000/alerting/list")
        
        alerts = [
            ("Model Health Failed", "1.rules_model_health.png", "2.notifikasi_model_health.png"),
            ("High Error Rate", "3.rules_error_rate.png", "4.notifikasi_error_rate.png"),
            ("High Prediction Latency", "5.rules_latency.png", "6.notifikasi_latency.png"),
        ]
        
        for i, (alert_name, rule_file, notif_file) in enumerate(alerts, 1):
            self.print_step(i, f"Alert: {alert_name}")
            
            print(f"    A. Alert Rule Screenshot:")
            print(f"       1. Go to Alerting → Alert rules")
            print(f"       2. Find '{alert_name}' rule")
            print(f"       3. Click to view details")
            print(f"       4. Take screenshot showing:")
            print(f"          - Alert condition")
            print(f"          - Threshold values")
            print(f"          - Evaluation interval")
            self.print_save_location(f"6.bukti alerting Grafana/{rule_file}")
            
            print(f"\n    B. Alert Notification Screenshot:")
            print(f"       Option 1 - If alert is firing:")
            print(f"       1. Check 'Firing' tab in alert rules")
            print(f"       2. Take screenshot of active alert")
            print(f"\n       Option 2 - Manually trigger alert:")
            
            if "Health" in alert_name:
                print(f"       1. Run: docker-compose stop mlserver")
                print(f"       2. Wait 2-3 minutes")
                print(f"       3. Check alert is firing")
                print(f"       4. Take screenshot")
                print(f"       5. Run: docker-compose start mlserver")
            
            self.print_save_location(f"6.bukti alerting Grafana/{notif_file}")
            print()
        
        print("\n    Additional Tips:")
        print("    - Contact points: Go to Alerting → Contact points")
        print("    - Notification policies: Go to Alerting → Notification policies")
        print("    - Alert history: Check 'State history' tab")
        
        print("\n" + "-"*70)
        input("Press Enter when done...")
    
    def check_directories(self):
        """Check and create necessary directories"""
        self.print_header("CHECKING DIRECTORIES")
        
        dirs = [
            "Monitoring dan Logging/1.bukti_serving",
            "Monitoring dan Logging/4.bukti monitoring Prometheus",
            "Monitoring dan Logging/5.bukti monitoring Grafana",
            "Monitoring dan Logging/6.bukti alerting Grafana",
        ]
        
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"✓ Created: {dir_path}")
            else:
                print(f"✓ Exists: {dir_path}")
    
    def generate_traffic(self):
        """Generate traffic for monitoring"""
        self.print_header("GENERATING TRAFFIC FOR MONITORING")
        
        print("\nTo generate monitoring data, run:")
        print("\n  python 7.inference.py --mode load-test --requests 200 --delay 0.05")
        print("\nThis will:")
        print("  - Make 200 predictions")
        print("  - Generate metrics for all panels")
        print("  - Take approximately 10-15 seconds")
        print("\nRun this BEFORE taking screenshots to ensure data is visible!")
        
        print("\n" + "-"*70)
        response = input("Run load test now? (y/n): ")
        
        if response.lower() == 'y':
            print("\nRunning load test...")
            os.system("python 7.inference.py --mode load-test --requests 200 --delay 0.05")
            print("\n✓ Load test completed!")
            print("  Wait 10-20 seconds for metrics to be scraped by Prometheus")
            time.sleep(5)
    
    def run_guide(self):
        """Run the complete screenshot guide"""
        print("\n")
        print("="*70)
        print(" HEART DISEASE MODEL - SCREENSHOT CAPTURE GUIDE")
        print("="*70)
        print("\nThis guide will walk you through capturing all required screenshots")
        print("for the monitoring and logging submission.")
        print("\n" + "="*70)
        
        input("\nPress Enter to start...")
        
        # Check directories
        self.check_directories()
        
        # Generate traffic
        self.generate_traffic()
        
        # Guide for each section
        self.guide_serving_screenshots()
        self.guide_prometheus_screenshots()
        self.guide_grafana_screenshots()
        self.guide_alerting_screenshots()
        
        # Final summary
        self.print_header("SCREENSHOT CHECKLIST")
        
        checklist = {
            "Serving (2)": [
                "1.bukti_serving/1.health_check.png",
                "1.bukti_serving/2.inference_test.png",
            ],
            "Prometheus (3)": [
                "4.bukti monitoring Prometheus/1.monitoring_health_status.png",
                "4.bukti monitoring Prometheus/2.monitoring_prediction_rate.png",
                "4.bukti monitoring Prometheus/3.monitoring_latency.png",
            ],
            "Grafana (10+)": [
                "5.bukti monitoring Grafana/1.monitoring_*.png (x10 or more)",
            ],
            "Alerting (6)": [
                "6.bukti alerting Grafana/1.rules_model_health.png",
                "6.bukti alerting Grafana/2.notifikasi_model_health.png",
                "6.bukti alerting Grafana/3.rules_error_rate.png",
                "6.bukti alerting Grafana/4.notifikasi_error_rate.png",
                "6.bukti alerting Grafana/5.rules_latency.png",
                "6.bukti alerting Grafana/6.notifikasi_latency.png",
            ]
        }
        
        print("\nScreenshots needed:")
        for category, files in checklist.items():
            print(f"\n{category}:")
            for file in files:
                print(f"  □ {file}")
        
        print("\n" + "="*70)
        print("✓ Guide completed!")
        print("\nNext steps:")
        print("  1. Verify all screenshots are captured")
        print("  2. Check image quality and readability")
        print("  3. Organize files in correct folders")
        print("  4. Review README_MONITORING.md for details")
        print("="*70)


def main():
    guide = ScreenshotGuide()
    guide.run_guide()


if __name__ == "__main__":
    main()