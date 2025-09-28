# Use a minimal Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies and create a user first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && useradd -m -u 1000 user \
    && rm -rf /var/lib/apt/lists/*

# Copy all files into /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working dir for the app
WORKDIR $HOME/app

# Copy source code again as user (if needed for ownership)
COPY --chown=user . $HOME/app

# Default command to launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
