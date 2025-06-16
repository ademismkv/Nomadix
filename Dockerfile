FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with better retry logic
RUN pip install --no-cache-dir -r requirements.txt \
    --timeout 300 \
    --retries 10

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p weights

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "ornament_analyzer:app", "--host", "0.0.0.0", "--port", "8000"] 