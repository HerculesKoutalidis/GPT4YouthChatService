# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code and images
COPY . .

# Expose the Streamlit port
EXPOSE 8505

# Command to run the app
CMD ["streamlit", "run", "chatbot_backend.py", "--server.port", "8505", "--server.address", "0.0.0.0"]