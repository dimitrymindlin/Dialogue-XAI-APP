# Use Python 3.10 image
FROM python:3.10-slim

WORKDIR /usr/src/app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Cython to ensure C-level compatibility for Pandas
RUN pip install --no-cache-dir cython

# Fix NumPy and Pandas with force-reinstall
RUN pip install --no-cache-dir --force-reinstall "numpy<2.0" "pandas<2.0" "scikit-learn>=1.1.0,<1.3.0" "scipy>=1.7.3,<1.10.0" "llvmlite>=0.39.0"

# Install all dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set and expose port
ENV PORT=5000
EXPOSE ${PORT}

VOLUME ["/app-data"]

# Start with Gunicorn
CMD ["gunicorn", "--timeout", "0", "-b", "0.0.0.0:5000", "flask_app:app", "--log-level", "debug"]
