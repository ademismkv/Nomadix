# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with retries and timeout
RUN pip install --no-cache-dir --timeout 100 --retries 3 -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run the application with host 0.0.0.0
CMD ["uvicorn", "ornament_analyzer:app", "--host", "0.0.0.0", "--port", "8000"] 