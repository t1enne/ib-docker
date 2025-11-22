#!/bin/sh

# Interactive Brokers Client Portal Gateway Docker Entrypoint

echo "Starting Interactive Brokers Client Portal Gateway..."
echo "Gateway Directory: $GATEWAY_DIR"
echo "Java Options: $JAVA_OPTS"

# Check if gateway directory exists
if [ ! -d "$GATEWAY_DIR" ]; then
    echo "Error: Gateway directory not found at $GATEWAY_DIR"
    exit 1
fi

# Check if run script exists
if [ ! -f "$GATEWAY_DIR/bin/run.sh" ]; then
    echo "Error: Gateway run script not found at $GATEWAY_DIR/bin/run.sh"
    exit 1
fi

# Change to gateway directory
cd "$GATEWAY_DIR" || exit 1

# Set Java options if provided
if [ -n "$JAVA_OPTS" ]; then
    export JAVA_OPTS
fi

# Display configuration file path
echo "Using configuration file: $GATEWAY_DIR/root/conf.yaml"

# Start the gateway
echo "Starting Client Portal Gateway on port 5000..."
echo "Access the login page at: https://localhost:5000"
echo ""

# Execute the gateway with configuration
exec sh "$GATEWAY_DIR/bin/run.sh" root/conf.yaml
