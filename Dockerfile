FROM python:3.13-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir .

# Copy application files
COPY app.py .
COPY .env .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
