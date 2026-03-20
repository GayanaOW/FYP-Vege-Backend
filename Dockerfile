FROM node:18

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

# Copy all files
COPY . .

# Install Node dependencies
RUN npm install

# Install Python dependencies
RUN pip3 install -r requirements1.txt

# Start the backend
CMD ["node", "src/server.js"]