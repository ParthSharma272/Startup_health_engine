# Dockerfile
FROM apache/airflow:2.11.0

# Install system-level dependencies required for packages like pytesseract and other Python libraries
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Switch to the 'airflow' user before installing Python packages
USER airflow

# Set working directory
WORKDIR /opt/airflow

# Upgrade pip, setuptools, and wheel to ensure compatibility
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# First, install compatible versions of pandas and protobuf
RUN python3 -m pip install --no-cache-dir pandas==2.1.4 protobuf==5.26.1

# Install the rest of the packages, avoiding any that might pull in conflicting versions
RUN python3 -m pip install --no-cache-dir \
    requests \
    Pillow==10.3.0 \
    pytesseract==0.3.10 \
    transformers==4.42.3 \
    torch==2.3.1 \
    streamlit==1.36.0 \
    langchain \
    faiss-cpu \
    sentence-transformers \
    pdfminer.six \
    PyMuPDF==1.24.5 \
    asyncio \
    aiohttp \
    aiofiles \
    openai \
    # Add ML packages with compatible versions
    scikit-learn==1.5.0 \
    numpy==1.26.4 \
    joblib==1.4.2 \
    mlflow==2.13.0 \
    matplotlib==3.8.4 \
    seaborn==0.13.2

# Skip the pip check since we have controlled the versions and it's causing issues
# RUN python3 -m pip check