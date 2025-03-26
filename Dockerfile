FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Declare build-time arguments
ARG BOT_URL

# Set runtime environment variables (use ENV to persist them)
ENV BOT_URL=${BOT_URL}

# Install dependencies efficiently
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application after installing dependencies (better caching)
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Use ENTRYPOINT to allow overriding CMD easily
ENTRYPOINT ["streamlit", "run", "client.py"]
CMD ["--server.port=8501", "--server.enableCORS=false", "--server.address=0.0.0.0"]
