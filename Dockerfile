FROM python:3.11.11-slim-bookworm

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
# COPY flask_app.py ./flask_app.py
# COPY models/vectorizer.pkl ./models/vectorizer.pkl

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet omw-1.4
# Install the package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 5000

#run the application in local machine
CMD ["python", "app.py"]

# Run the application using Gunicorn
# gunicorn is a WSGI server for Python web applications
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"] 
