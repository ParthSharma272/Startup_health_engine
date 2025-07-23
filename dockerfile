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

# Upgrade pip to ensure it's the latest version
USER airflow
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir \
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
    asyncio \
    aiohttp \
    aiofiles \
    openai # <--- THIS IS THE KEY PACKAGE THAT WAS MISSING
