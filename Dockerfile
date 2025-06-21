FROM python:3.11-slim-bullseye

# Set work directory
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
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --timeout 300 --retries 10

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p weights

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8000

# Use environment variables for secrets/config (Railway best practice)
# Do NOT copy .env by default; set env vars in Railway dashboard

# Command to run the application
CMD ["uvicorn", "ornament_analyzer:app", "--host", "0.0.0.0", "--port", "8000"] 