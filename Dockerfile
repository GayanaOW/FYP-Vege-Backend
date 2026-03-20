FROM node:18

# Install Python + venv
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip

WORKDIR /app

# Copy files
COPY . .

# Create virtual environment
RUN python3 -m venv /opt/venv

# Activate venv (set PATH)
ENV PATH="/opt/venv/bin:$PATH"

# Install Node deps
RUN npm install

# Install Python deps inside venv
RUN pip install -r requirements1.txt

# Start app
CMD ["node", "src/server.js"]