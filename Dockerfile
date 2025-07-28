# Step 1: Use a stable, lightweight, and AMD64-compatible base image
# Explicitly specify the platform for cross-platform build compatibility
FROM --platform=linux/amd64 python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies if any are needed
# For example, if you use libraries that need poppler-utils for PDF processing:
# RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file and install dependencies
# This leverages Docker's layer caching. Dependencies are only re-installed if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the application source code and models
COPY ./src ./src
COPY ./models ./models
COPY main.py .

# Step 6: Define the command to run the application
# This will execute main.py when the container starts.
CMD ["python", "main.py"] 