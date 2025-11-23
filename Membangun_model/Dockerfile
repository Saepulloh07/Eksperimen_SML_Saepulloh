FROM python:3.10-slim

LABEL maintainer="saepulloh@mlops.com"
LABEL description="Wine Quality Prediction Model"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY wine_quality_model.pkl .
COPY ../preprosessing/data/output/scaler.pkl ./scaler.pkl
COPY ../preprosessing/data/output/feature_names.pkl ./feature_names.pkl

# Copy serving script (we'll create this next)
COPY serve_model.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "serve_model.py"]