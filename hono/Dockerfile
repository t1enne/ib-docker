FROM oven/bun:latest

RUN apt-get update && apt-get install -y watchman && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN bun install

COPY . .

EXPOSE 3000
