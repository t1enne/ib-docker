# Interactive Brokers Client Portal Gateway Docker Container
FROM eclipse-temurin:11-jre-alpine

# Set working directory
WORKDIR /app

# Install curl for downloading gateway and other utilities
RUN apk add --no-cache curl unzip

# Create gateway directory
RUN mkdir -p /app/gateway

# Download and extract Client Portal Gateway
# Note: You may need to update this URL to the latest version
# RUN curl -L -o clientportal.gw.zip https://download2.interactivebrokers.com/portal/clientportal.gw.zip && \
#     unzip clientportal.gw.zip -d /app/gateway && \
#     rm clientportal.gw.zip

COPY gateway /app/gateway
# Set correct permissions
RUN chmod +x /app/gateway/bin/run.sh

# Create directories for configuration and certificates
RUN mkdir -p /app/certs /app/config

# Certificates
RUN keytool -genkey -keyalg RSA -alias selfsigned -keystore cacert.jks -storepass abc123 -validity 730 -keysize 2048 -dname CN=localhost
RUN keytool -importkeystore -srckeystore cacert.jks -destkeystore cacert.p12 -srcstoretype jks -deststoretype pkcs12 -srcstorepass abc123 -deststorepass abc123
RUN openssl pkcs12 -in cacert.p12 -out cacert.pem -passin pass:abc123 -passout pass:abc123
RUN cp cacert.pem /app/gateway/root/cacert.pem
RUN cp cacert.jks /app/gateway/root/cacert.jks

# Copy configuration files (will be created separately)
COPY conf.yaml /app/gateway/root/conf.yaml
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose the gateway port
EXPOSE 5000

# Set environment variables
ENV GATEWAY_DIR="/app/gateway"
ENV JAVA_OPTS="-Xmx512m"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -fk https://localhost:5000/v1/api/tickle || exit 1

# Run the gateway
ENTRYPOINT ["/app/entrypoint.sh"]
