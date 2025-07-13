# Use a stable Python image
FROM python:3.10.16

# Creating Application Source Code Directory
RUN mkdir -p /usr/src/app

# Setting Home Directory for the container
WORKDIR /usr/src/app

# Upgrade pip to avoid dependency resolution issues
RUN pip install --upgrade pip setuptools wheel

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    gcc \
    g++ \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt /usr/src/app/

# Install dependencies with pre-built wheels to avoid compilation issues
RUN pip install --no-cache-dir --prefer-binary spacy

# Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /usr/src/app

# Application Environment variables
ENV PORT=4000

# Exposing Ports
EXPOSE $PORT

# Setting Persistent data
VOLUME ["/app-data"]

# Run the Python application with Gunicorn
CMD ["gunicorn", "--workers", "2", "--worker-class", "gevent", "--worker-connections", "20", "--timeout", "140", "--bind", "0.0.0.0:4000", "flask_app:app"]