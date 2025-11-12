FROM python:3.10-slim

ENV PORT=10000
WORKDIR /app

# system deps for some python packages
RUN apt-get update && apt-get install -y build-essential wget curl && rm -rf /var/lib/apt/lists/*

# install pip deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

# start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 
CMD ["/app/start.sh"]
