# Docker file adapted from this tutorial https://github.com/bennzhang/docker-demo-with-simple-python-app
FROM python:3.9.15

# Creating Application Source Code Directory
RUN mkdir -p /usr/src/app

# Setting Home Directory for containers
WORKDIR /usr/src/app

# Installing python dependencies
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

# Copying source code to Container
COPY . /usr/src/app

# Application Environment variables
ENV PORT 4000

# Exposing Ports
EXPOSE $PORT

# Setting Persistent data
VOLUME ["/app-data"]

# Running Python Application
CMD gunicorn --timeout 0 -b 0.0.0.0:$PORT flask_app:app