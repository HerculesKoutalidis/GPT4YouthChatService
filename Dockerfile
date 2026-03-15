# Use a lightweight Python base image
FROM public.ecr.aws/docker/library/python:3.10-slim

# Set the working directory inside the container
WORKDIR /app    
# Set PYTHONPATH so python can find the 'src' module from /app
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code and images
COPY . .

# Expose the Streamlit port
EXPOSE 8505

# Command to run the app
CMD ["streamlit", "run", "src/ui/chatbot_backend.py", "--server.port", "8505", "--server.address", "0.0.0.0"]