# Use the specific Python version as base image
FROM python:3.10-slim-buster

# Set the maintainer label
LABEL maintainer="fernandochafim@gmail.com"

# Set environment variables for Python
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory
WORKDIR /home/iaeml

# Update and install system dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential
# Add any other system libraries that are necessary for your project

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install poetry for managing Python dependencies
RUN pip install poetry

# Copy only the pyproject.toml first to leverage Docker cache
COPY ./pyproject.toml /home/iaeml/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Now, copy the entire project to the container
COPY . /home/iaeml/

# Set default command to bash shell
CMD ["/bin/bash"]
